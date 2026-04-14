"""
app.py  –  Emotion-Adaptive Music  |  Flask Backend
────────────────────────────────────────────────────
Receives a video clip from the browser, runs your exact feature extraction
pipeline (audio + MediaPipe facial landmarks), feeds the features into the
trained GradientBoosting models, then generates and returns a MIDI file.

Setup
─────
  pip install flask flask-cors librosa opencv-python mediapipe midiutil
              scikit-learn pandas numpy joblib

Train & save models first (run once):
  python train_and_save.py           ← generated alongside this file

Run:
  python app.py
  # → http://localhost:5000

Endpoints
─────────
  POST /predict   multipart/form-data  { video: <file> }
                  → JSON { valence, arousal, scale, tempo_bpm, emotion_label }

  POST /generate  multipart/form-data  { video: <file> }
                  → MIDI file (audio/midi)

  GET  /health    → JSON { status: "ok" }
"""

import io
import os
import random
import tempfile
import traceback

import cv2
import joblib
import librosa
import mediapipe as mp
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (FaceLandmarker,
                                            FaceLandmarkerOptions,
                                            RunningMode)
from midiutil import MIDIFile

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR        = os.path.join(os.path.dirname(__file__), "models")
MLP_MODEL        = os.path.join(MODEL_DIR, "va_mlp.pt")
MLP_CONFIG       = os.path.join(MODEL_DIR, "mlp_config.joblib")
FEATURE_COLS     = os.path.join(MODEL_DIR, "feature_cols.joblib")
LANDMARKER_MODEL = os.path.join(MODEL_DIR, "face_landmarker.task")

LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

app = Flask(__name__)
CORS(app)   # allow browser requests from any origin

# ─────────────────────────────────────────────────────────────────────────────
# Load models at startup
# ─────────────────────────────────────────────────────────────────────────────

def _download_landmarker():
    import urllib.request
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(LANDMARKER_MODEL):
        print("Downloading face_landmarker.task …")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_MODEL)
        print("Download complete.")

_download_landmarker()

options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
face_landmarker = FaceLandmarker.create_from_options(options)

# ── Model state (lazy-loaded on first request) ───────────────────────────────
_va_model      = None
_scaler        = None
_feature_cols  = None
_whisper_model = None

import torch
import torch.nn as nn

class VAPredictor(nn.Module):
    """Must match architecture in train_and_save.py exactly."""
    def __init__(self, input_dim, hidden, dropout=0.0):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(),
                       nn.Dropout(max(dropout, 0.0))]
            in_dim = h; dropout -= 0.1
        self.shared       = nn.Sequential(*layers)
        self.valence_head = nn.Linear(in_dim, 1)
        self.arousal_head = nn.Linear(in_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        return torch.cat([self.valence_head(h),
                          torch.sigmoid(self.arousal_head(h))], dim=1)

_torch_device = torch.device("mps"  if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")

def _load_models():
    global _va_model, _scaler, _feature_cols
    if _va_model is not None:
        return
    mlp_path = os.path.join(MODEL_DIR, "va_mlp.pt")
    cfg_path  = os.path.join(MODEL_DIR, "mlp_config.joblib")
    if not os.path.exists(mlp_path):
        raise FileNotFoundError(
            "Trained model not found. Run train_and_save.py first."
        )
    cfg           = joblib.load(cfg_path)
    model         = VAPredictor(cfg["input_dim"], cfg["hidden"], cfg["dropout"])
    model.load_state_dict(torch.load(mlp_path, map_location=_torch_device))
    model.to(_torch_device).eval()
    _va_model     = model
    _scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    _feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))
    print(f"MLP loaded ({cfg['input_dim']} features) on {_torch_device}")

def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            print("Loading Whisper 'base' model...")
            _whisper_model = whisper.load_model("base")
            print("Whisper ready.")
        except ImportError:
            print("Warning: openai-whisper not installed. Text features disabled.")
            _whisper_model = False
    return _whisper_model if _whisper_model else None


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction  (mirrors video_feature_extraction.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LEFT_EYE_TOP,  LEFT_EYE_BOTTOM  = 159, 145
LEFT_EYE_LEFT, LEFT_EYE_RIGHT   = 33,  133
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 362, 263
LEFT_BROW_INNER  = 107
RIGHT_BROW_INNER = 336


def _euclidean(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def _landmark_features(lm):
    mar = _euclidean(lm[MOUTH_TOP],  lm[MOUTH_BOTTOM]) / (
          _euclidean(lm[MOUTH_LEFT], lm[MOUTH_RIGHT])  + 1e-6)
    l_ear = _euclidean(lm[LEFT_EYE_TOP],  lm[LEFT_EYE_BOTTOM])  / (
            _euclidean(lm[LEFT_EYE_LEFT], lm[LEFT_EYE_RIGHT])   + 1e-6)
    r_ear = _euclidean(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOTTOM]) / (
            _euclidean(lm[RIGHT_EYE_LEFT],lm[RIGHT_EYE_RIGHT])  + 1e-6)
    ear   = (l_ear + r_ear) / 2.0
    brow  = (_euclidean(lm[LEFT_BROW_INNER],  lm[LEFT_EYE_LEFT]) +
             _euclidean(lm[RIGHT_BROW_INNER], lm[RIGHT_EYE_RIGHT])) / 2.0
    return {"mar": mar, "ear": ear, "brow_raise": brow}


def extract_features(video_path: str, audio_path: str) -> dict:
    """
    Extract the exact same feature vector produced by video_feature_extraction.py.
    Returns a flat dict ready to be turned into a single-row DataFrame.
    """
    # ── Audio ──────────────────────────────────────────────────────────────
    os.system(f'ffmpeg -i "{video_path}" -q:a 0 -map a '
              f'"{audio_path}" -y -loglevel quiet')

    y, sr = librosa.load(audio_path, sr=None)
    audio_duration = len(y) / sr   # seconds — used for speaking rate

    mfccs       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean  = np.mean(mfccs, axis=1)
    mfccs_std   = np.std(mfccs,  axis=1)
    delta       = librosa.feature.delta(mfccs)
    delta2      = librosa.feature.delta(mfccs, order=2)
    delta_mean  = np.mean(delta,  axis=1)
    delta2_mean = np.mean(delta2, axis=1)

    pitch        = librosa.yin(y, fmin=50, fmax=500)
    pitch_voiced = pitch[pitch > 0]
    pitch_mean   = float(np.mean(pitch_voiced))  if len(pitch_voiced) > 0 else 0.0
    pitch_std    = float(np.std(pitch_voiced))   if len(pitch_voiced) > 0 else 0.0
    pitch_range  = float(np.max(pitch_voiced) - np.min(pitch_voiced)) if len(pitch_voiced) > 0 else 0.0
    voiced_ratio = len(pitch_voiced) / (len(pitch) + 1e-6)

    rms          = librosa.feature.rms(y=y)
    energy_mean  = float(np.mean(rms))
    energy_std   = float(np.std(rms))

    spec_centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spec_rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr            = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # ── Visual ─────────────────────────────────────────────────────────────
    cap           = cv2.VideoCapture(video_path)
    frame_feats   = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = face_landmarker.detect(mp_img)
        if result.face_landmarks:
            frame_feats.append(_landmark_features(result.face_landmarks[0]))
    cap.release()

    if frame_feats:
        ff = pd.DataFrame(frame_feats)
        face_detected  = 1.0
        face_mar_mean  = float(ff["mar"].mean())
        face_mar_std   = float(ff["mar"].std())
        face_ear_mean  = float(ff["ear"].mean())
        face_ear_std   = float(ff["ear"].std())
        face_brow_mean = float(ff["brow_raise"].mean())
        face_brow_std  = float(ff["brow_raise"].std())
    else:
        face_detected  = 0.0
        face_mar_mean  = face_mar_std  = 0.0
        face_ear_mean  = face_ear_std  = 0.0
        face_brow_mean = face_brow_std = 0.0

    return {
        **{f"mfcc_mean_{i+1}":   float(mfccs_mean[i])  for i in range(13)},
        **{f"mfcc_std_{i+1}":    float(mfccs_std[i])   for i in range(13)},
        **{f"mfcc_delta_{i+1}":  float(delta_mean[i])  for i in range(13)},
        **{f"mfcc_delta2_{i+1}": float(delta2_mean[i]) for i in range(13)},
        "pitch_mean":   pitch_mean,   "pitch_std":   pitch_std,
        "pitch_range":  pitch_range,  "voiced_ratio": voiced_ratio,
        "energy_mean":  energy_mean,  "energy_std":   energy_std,
        "spec_centroid":  spec_centroid,
        "spec_bandwidth": spec_bandwidth,
        "spec_rolloff":   spec_rolloff,
        "zcr":            zcr,
        "face_detected":  face_detected,
        "face_mar_mean":  face_mar_mean,  "face_mar_std":  face_mar_std,
        "face_ear_mean":  face_ear_mean,  "face_ear_std":  face_ear_std,
        "face_brow_mean": face_brow_mean, "face_brow_std": face_brow_std,
        # Text features — populated separately via Whisper in app routes
        # (filled with zeros here; routes call transcribe_and_extract directly)
        "text_has_transcript": 0.0, "text_word_count": 0.0,
        "text_speaking_rate": 0.0,  "text_exclamation_ratio": 0.0,
        "text_question_ratio": 0.0, "text_lex_valence": 0.0,
        "text_lex_valence_std": 0.0,"text_sentiment_words": 0.0,
        "text_avg_sent_length": 0.0,"text_caps_ratio": 0.0,
        "_audio_duration": audio_duration,   # pass-through, stripped before predict
    }


# ─────────────────────────────────────────────────────────────────────────────
# Text feature extraction  (mirrors transcribe_features.py)
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

SENTIMENT_LEXICON = {
    "good": 0.7, "great": 0.8, "happy": 0.9, "love": 0.85, "wonderful": 0.9,
    "excellent": 0.85, "fantastic": 0.9, "nice": 0.6, "beautiful": 0.8,
    "joy": 0.85, "joyful": 0.85, "pleased": 0.7, "glad": 0.65, "fine": 0.4,
    "okay": 0.2, "ok": 0.2, "yes": 0.3, "sure": 0.3, "right": 0.2,
    "bright": 0.5, "warm": 0.5, "thanks": 0.6, "thank": 0.6, "welcome": 0.5,
    "excited": 0.75, "amazing": 0.85, "brilliant": 0.8, "perfect": 0.85,
    "bad": -0.7, "sad": -0.8, "angry": -0.8, "hate": -0.9, "terrible": -0.85,
    "awful": -0.85, "horrible": -0.9, "wrong": -0.5, "no": -0.3, "not": -0.3,
    "never": -0.4, "fear": -0.75, "scared": -0.75, "afraid": -0.7,
    "disgust": -0.8, "disgusting": -0.85, "gross": -0.7, "ugly": -0.65,
    "hurt": -0.7, "pain": -0.75, "suffer": -0.8, "cry": -0.65, "dark": -0.4,
    "cold": -0.3, "lonely": -0.75, "lost": -0.5, "fail": -0.65, "failed": -0.65,
    "dead": -0.85, "die": -0.85, "kill": -0.85, "mad": -0.7,
    "furious": -0.9, "rage": -0.9, "upset": -0.65, "miserable": -0.85,
    "very": 1.3, "really": 1.2, "so": 1.15, "extremely": 1.5, "quite": 1.1,
    "absolutely": 1.4, "totally": 1.3, "completely": 1.3,
}

def extract_text_features(transcript: str, duration_seconds: float) -> dict:
    text  = transcript.strip().lower()
    words = _re.findall(r"[a-z']+", text)
    word_count   = len(words)
    speaking_rate = word_count / max(duration_seconds, 0.1)
    has_transcript = 1.0 if word_count > 0 else 0.0
    exclamation_ratio = transcript.count("!") / max(word_count, 1)
    question_ratio    = transcript.count("?") / max(word_count, 1)

    sentiment_scores = []
    intensifier = 1.0
    for word in words:
        score = SENTIMENT_LEXICON.get(word)
        if score is None:
            intensifier = 1.0; continue
        if score > 1.0:
            intensifier = score; continue
        sentiment_scores.append(score * intensifier)
        intensifier = 1.0

    lex_valence     = float(np.mean(sentiment_scores))  if sentiment_scores else 0.0
    lex_valence_std = float(np.std(sentiment_scores))   if sentiment_scores else 0.0

    sentences = [s.strip() for s in _re.split(r'[.!?]+', transcript) if s.strip()]
    avg_sent_length = word_count / max(len(sentences), 1)
    raw_words = _re.findall(r"[A-Za-z']+", transcript)
    caps_ratio = sum(1 for w in raw_words if w.isupper() and len(w) > 1) / max(len(raw_words), 1)

    return {
        "text_has_transcript":    has_transcript,
        "text_word_count":        float(word_count),
        "text_speaking_rate":     float(speaking_rate),
        "text_exclamation_ratio": float(exclamation_ratio),
        "text_question_ratio":    float(question_ratio),
        "text_lex_valence":       float(np.clip(lex_valence, -1.0, 1.0)),
        "text_lex_valence_std":   float(lex_valence_std),
        "text_sentiment_words":   float(len(sentiment_scores)),
        "text_avg_sent_length":   float(avg_sent_length),
        "text_caps_ratio":        float(caps_ratio),
    }

def transcribe_and_extract(audio_path: str, duration: float) -> dict:
    """Run Whisper on an audio file and return text features. Falls back to
    zeros if Whisper is not installed or transcription fails."""
    wmodel = _load_whisper()
    if wmodel is None:
        return extract_text_features("", duration)
    try:
        import whisper
        audio  = whisper.load_audio(audio_path)
        result = wmodel.transcribe(audio, fp16=False, language="en")
        transcript = result["text"]
        print(f"  [whisper] '{transcript.strip()[:60]}'")
        return extract_text_features(transcript, duration)
    except Exception as e:
        print(f"  [whisper] transcription failed: {e}")
        return extract_text_features("", duration)


# ─────────────────────────────────────────────────────────────────────────────
# Music generation -- phrase-based, structured, multi-track
# ─────────────────────────────────────────────────────────────────────────────

import subprocess

SOUNDFONT_PATH = "/usr/share/sounds/sf2/FluidR3_GM.sf2"  # overridden at runtime

SCALES = {
    "major":  [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "minor":  [0, 2, 3, 5, 7, 8, 10],
}

# (scale_degree, inversion) per bar -- 8 bars
PROGRESSIONS = {
    "major":  [(0,0),(3,1),(4,0),(0,0),(5,1),(3,0),(4,2),(0,0)],
    "dorian": [(0,0),(3,0),(0,1),(4,1),(0,0),(5,1),(3,0),(0,0)],
    "minor":  [(0,0),(5,1),(3,0),(4,2),(0,0),(6,0),(3,1),(0,0)],
}

# Relative scale-degree offsets for each phrase contour
CONTOURS = {
    "happy":       [ 0, 2, 4, 2,  3, 5, 4, 2],
    "content":     [ 0, 1, 2, 1,  2, 1, 0, 0],
    "hopeful":     [ 0, 2, 1, 3,  2, 4, 3, 1],
    "pensive":     [ 0,-1, 0, 1,  0,-1,-2, 0],
    "tense":       [ 0, 3,-1, 2, -2, 4,-3, 1],
    "melancholic": [ 0,-1,-2,-1, -3,-2,-4,-2],
}

def _va_to_scale(v): return "major" if v>=0.05 else ("dorian" if v>=-0.3 else "minor")

def _va_to_root(v):
    bright  = [64, 67, 62, 69]
    neutral = [60, 65]
    dark    = [58, 63, 56, 61]
    pool = bright if v>=0.2 else (neutral if v>=-0.2 else dark)
    return pool[int(abs(v) * (len(pool)-1))]

def _va_to_tempo(a):    return int(np.clip(60 + a*120, 60, 180))
def _va_to_velocity(a): return int(np.clip(50 + a*60,  50, 110))

def _emotion_region(scale, valence, arousal):
    if scale == "major":   return "happy"   if arousal >= 0.45 else "content"
    elif scale == "dorian":return "hopeful" if arousal >= 0.45 else "pensive"
    else:                  return "tense"   if arousal >= 0.45 else "melancholic"

def _build_scale(root, ivls, octaves=3):
    return [root + i + o*12 for o in range(octaves) for i in ivls]

def _build_chord_voiced(root, ivls, degree, inversion=0):
    triad_degs = [degree % 7, (degree+2) % 7, (degree+4) % 7]
    pitches    = [root + ivls[d] for d in triad_degs]
    for _ in range(inversion):
        pitches[0] += 12
        pitches.sort()
    return [p - 12 for p in pitches]

def _build_bass_note(root, ivls, degree):
    return root + ivls[degree % 7] - 24

def _make_phrase(scale_notes, contour, start_idx, beats_per_phrase, arousal, valence, rng):
    if arousal > 0.65:
        rhythms = [0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5]
    elif arousal > 0.35:
        rhythms = [1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0]
    else:
        rhythms = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0]

    events   = []
    beat     = 0.0
    base_vel = _va_to_velocity(arousal)

    for i, offset in enumerate(contour):
        if beat >= beats_per_phrase: break
        dur      = min(rhythms[i % len(rhythms)], beats_per_phrase - beat)
        idx      = int(np.clip(start_idx + offset, 0, len(scale_notes)-1))
        note     = scale_notes[idx]
        beat_pos = beat % 4
        vel_adj  = 8 if beat_pos == 0 else (4 if beat_pos == 2 else 0)
        vel      = int(np.clip(base_vel + vel_adj + rng.randint(-5,5), 30, 120))
        rest_prob= max(0.0, 0.12 - arousal * 0.1)
        if rng.random() > rest_prob:
            artic = 0.55 if arousal > 0.65 else (0.80 if arousal > 0.35 else 0.92)
            events.append((note, beat, dur * artic, vel))
        beat += dur

    return events


def generate_midi(valence: float, arousal: float,
                  num_bars: int = 8, seed: int = 42) -> bytes:
    rng    = random.Random(seed)
    scale  = _va_to_scale(valence)
    root   = _va_to_root(valence)
    ivls   = SCALES[scale]
    tempo  = _va_to_tempo(arousal)
    prog   = PROGRESSIONS[scale]
    region = _emotion_region(scale, valence, arousal)
    contour= CONTOURS[region]

    scale_notes = _build_scale(root, ivls, octaves=3)
    start_idx   = len(scale_notes) // 2

    # Instrument selection: (melody, chord pad, bass)
    if scale == "major" and arousal >= 0.5:
        mel_prog, pad_prog, bas_prog = 0,  48, 33
    elif scale == "major":
        mel_prog, pad_prog, bas_prog = 11, 52, 32
    elif scale == "dorian":
        mel_prog, pad_prog, bas_prog = 73, 48, 33
    else:
        mel_prog, pad_prog, bas_prog = 70, 44, 42

    midi = MIDIFile(numTracks=3)
    for t in range(3): midi.addTempo(t, 0, tempo)
    midi.addProgramChange(0, 0, 0, mel_prog)
    midi.addProgramChange(1, 1, 0, pad_prog)
    midi.addProgramChange(2, 2, 0, bas_prog)

    beats_per_bar    = 4
    beats_per_phrase = beats_per_bar * 4
    total_beats      = num_bars * beats_per_bar

    def section_vel_scale(bar):
        if bar < 2:               return 0.65
        elif bar >= num_bars - 2: return 0.70
        else:                     return 1.0

    # Track 0: Melody (two 4-bar phrases)
    for phrase_idx in range(num_bars // 4):
        phrase_start  = phrase_idx * beats_per_phrase
        phrase_offset = 0 if phrase_idx == 0 else (2 if valence > 0 else -2)
        p_start_idx   = int(np.clip(start_idx + phrase_offset, 0, len(scale_notes)-1))
        events = _make_phrase(scale_notes, contour, p_start_idx,
                              beats_per_phrase, arousal, valence, rng)
        for (note, beat_off, dur, vel) in events:
            bar = int((phrase_start + beat_off) // beats_per_bar)
            midi.addNote(0, 0, note, phrase_start + beat_off, dur,
                         int(vel * section_vel_scale(bar)))

    # Track 1: Chords with voice leading
    base_chord_vel = max(30, _va_to_velocity(arousal) - 20)
    for bar in range(num_bars):
        bt         = bar * beats_per_bar
        vel_scale  = section_vel_scale(bar)
        degree, inv = prog[bar % len(prog)]
        chord = _build_chord_voiced(root, ivls, degree, inv)
        cv    = int(base_chord_vel * vel_scale)
        if arousal >= 0.6:
            step = beats_per_bar / len(chord)
            for i, n in enumerate(chord):
                midi.addNote(1, 1, n, bt + i*step, step*0.85, cv)
        elif arousal >= 0.35:
            for n in chord:
                midi.addNote(1, 1, n, bt,   1.8, cv)
                midi.addNote(1, 1, n, bt+2, 1.8, int(cv*0.85))
        else:
            for n in chord:
                midi.addNote(1, 1, n, bt, beats_per_bar * 0.95, cv)

    # Track 2: Bass line
    base_bass_vel = max(40, _va_to_velocity(arousal) - 10)
    for bar in range(num_bars):
        bt         = bar * beats_per_bar
        vel_scale  = section_vel_scale(bar)
        degree, _  = prog[bar % len(prog)]
        bass_root  = _build_bass_note(root, ivls, degree)
        bass_fifth = bass_root + 7
        bv         = int(base_bass_vel * vel_scale)
        if arousal >= 0.6:
            midi.addNote(2, 2, bass_root,   bt,   0.9, bv)
            midi.addNote(2, 2, bass_root+2, bt+1, 0.9, int(bv*0.8))
            midi.addNote(2, 2, bass_fifth,  bt+2, 0.9, bv)
            midi.addNote(2, 2, bass_fifth+2,bt+3, 0.9, int(bv*0.8))
        elif arousal >= 0.35:
            midi.addNote(2, 2, bass_root,  bt,   1.8, bv)
            midi.addNote(2, 2, bass_fifth, bt+2, 1.8, int(bv*0.9))
        else:
            midi.addNote(2, 2, bass_root, bt, beats_per_bar*0.95, bv)

    buf = io.BytesIO()
    midi.writeFile(buf)
    return buf.getvalue()


def midi_to_wav(midi_bytes: bytes, soundfont: str = SOUNDFONT_PATH) -> bytes:
    """Render MIDI to WAV via FluidSynth. Returns WAV bytes or empty bytes."""
    import shutil
    if not shutil.which("fluidsynth"):
        print("FluidSynth not found -- install with: brew install fluidsynth")
        return b""
    # Search common soundfont locations if the configured path doesn't exist
    candidates = [
        soundfont,
        "/opt/homebrew/share/soundfonts/default.sf2",
        "/opt/homebrew/share/soundfonts/GeneralUser.sf2",
        os.path.expanduser("~/Downloads/GeneralUser.sf2"),
        os.path.expanduser("~/Downloads/FluidR3_GM.sf2"),
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    ]
    sf = next((p for p in candidates if p and os.path.exists(p)), None)
    if not sf:
        print("No soundfont found. Download one to ~/Downloads/ e.g. FluidR3_GM.sf2")
        return b""
    with tempfile.TemporaryDirectory() as tmp:
        mid_path = os.path.join(tmp, "out.mid")
        wav_path = os.path.join(tmp, "out.wav")
        with open(mid_path, "wb") as f:
            f.write(midi_bytes)
        result = subprocess.run(
            ["fluidsynth", "-ni", sf, mid_path, "-F", wav_path, "-r", "44100", "-q"],
            capture_output=True
        )
        if result.returncode == 0 and os.path.exists(wav_path):
            with open(wav_path, "rb") as f:
                return f.read()
        print(f"FluidSynth error: {result.stderr.decode()[:200]}")
        return b""


def emotion_label(scale: str, valence: float, arousal: float) -> str:
    labels = {
        ("major", True):  "Happy",
        ("major", False): "Content",
        ("dorian", True): "Hopeful",
        ("dorian",False): "Pensive",
        ("minor", True):  "Tense",
        ("minor", False): "Melancholic",
    }
    return labels.get((scale, arousal >= 0.45), "Neutral")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive a video clip, extract features, return VA prediction + music params.
    Body: multipart/form-data  { video: <file> }
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    try:
        _load_models()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    video_file = request.files["video"]

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "clip.webm")
        audio_path = os.path.join(tmp, "clip.wav")
        video_file.save(video_path)

        try:
            feats = extract_features(video_path, audio_path)
        except Exception as e:
            return jsonify({"error": f"Feature extraction failed: {e}",
                            "detail": traceback.format_exc()}), 500

        # Run Whisper and merge text features before predicting
        duration = feats.pop("_audio_duration", 5.0)
        text_feats = transcribe_and_extract(audio_path, duration)
        feats.update(text_feats)

        # Build feature vector aligned to training columns
        row_data = np.array([[feats.get(k, 0.0) for k in _feature_cols]],
                             dtype=np.float32)
        row_scaled = _scaler.transform(row_data)
        with torch.no_grad():
            pred = _va_model(torch.tensor(row_scaled).to(_torch_device)).cpu().numpy()[0]
        valence = float(np.clip(pred[0], -1.0, 1.0))
        arousal = float(np.clip(pred[1],  0.0, 1.0))
        scale    = _va_to_scale(valence)
        tempo    = _va_to_tempo(arousal)
        label    = emotion_label(scale, valence, arousal)

    return jsonify({
        "valence":       round(valence, 4),
        "arousal":       round(arousal, 4),
        "scale":         scale,
        "tempo_bpm":     tempo,
        "emotion_label": label,
    })


@app.route("/generate", methods=["POST"])
def generate():
    """
    Receive a video clip → predict VA → generate MIDI → return MIDI file.
    Body: multipart/form-data  { video: <file> }
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    try:
        _load_models()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    video_file = request.files["video"]

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "clip.webm")
        audio_path = os.path.join(tmp, "clip.wav")
        video_file.save(video_path)

        try:
            feats = extract_features(video_path, audio_path)
        except Exception as e:
            return jsonify({"error": f"Feature extraction failed: {e}",
                            "detail": traceback.format_exc()}), 500

        # Run Whisper and merge text features before predicting
        duration = feats.pop("_audio_duration", 5.0)
        text_feats = transcribe_and_extract(audio_path, duration)
        feats.update(text_feats)

        row_data = np.array([[feats.get(k, 0.0) for k in _feature_cols]],
                             dtype=np.float32)
        row_scaled = _scaler.transform(row_data)
        with torch.no_grad():
            pred = _va_model(torch.tensor(row_scaled).to(_torch_device)).cpu().numpy()[0]
        valence = float(np.clip(pred[0], -1.0, 1.0))
        arousal = float(np.clip(pred[1],  0.0, 1.0))

    scale = _va_to_scale(valence)
    tempo = _va_to_tempo(arousal)
    label = emotion_label(scale, valence, arousal)

    midi_bytes = generate_midi(valence, arousal, num_bars=8)
    import base64
    midi_b64 = base64.b64encode(midi_bytes).decode()

    return jsonify({
        "valence":       round(valence, 4),
        "arousal":       round(arousal, 4),
        "scale":         scale,
        "tempo_bpm":     tempo,
        "emotion_label": label,
        "midi_b64":      midi_b64,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Emotion Music API on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
