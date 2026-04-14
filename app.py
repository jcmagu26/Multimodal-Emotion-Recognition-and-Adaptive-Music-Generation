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

# Additional landmark indices for valence-sensitive features (must match
# video_feature_extraction.py exactly)
MOUTH_CORNER_LEFT  = 61
MOUTH_CORNER_RIGHT = 291
UPPER_LIP_CENTER   = 13
LOWER_LIP_CENTER   = 14
LEFT_CHEEK         = 234
RIGHT_CHEEK        = 454
NOSE_TIP           = 4


def _euclidean(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def _landmark_features(lm):
    """Mirrors compute_landmark_features() in video_feature_extraction.py exactly."""
    mouth_width = _euclidean(lm[MOUTH_LEFT], lm[MOUTH_RIGHT])

    mar = _euclidean(lm[MOUTH_TOP], lm[MOUTH_BOTTOM]) / (mouth_width + 1e-6)

    l_ear = _euclidean(lm[LEFT_EYE_TOP],  lm[LEFT_EYE_BOTTOM])  / (
            _euclidean(lm[LEFT_EYE_LEFT], lm[LEFT_EYE_RIGHT])   + 1e-6)
    r_ear = _euclidean(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOTTOM]) / (
            _euclidean(lm[RIGHT_EYE_LEFT],lm[RIGHT_EYE_RIGHT])  + 1e-6)
    ear   = (l_ear + r_ear) / 2.0

    brow  = (_euclidean(lm[LEFT_BROW_INNER],  lm[LEFT_EYE_LEFT]) +
             _euclidean(lm[RIGHT_BROW_INNER], lm[RIGHT_EYE_RIGHT])) / 2.0

    # smile_ratio: lip corners rise above lip centre in a smile
    lip_center_y = (lm[UPPER_LIP_CENTER].y + lm[LOWER_LIP_CENTER].y) / 2.0
    corner_y_avg = (lm[MOUTH_CORNER_LEFT].y + lm[MOUTH_CORNER_RIGHT].y) / 2.0
    smile_ratio  = (lip_center_y - corner_y_avg) / (mouth_width + 1e-6)

    # cheek_raise: Duchenne smile — cheeks lift above nose tip
    nose_y      = lm[NOSE_TIP].y
    cheek_y_avg = (lm[LEFT_CHEEK].y + lm[RIGHT_CHEEK].y) / 2.0
    cheek_raise = nose_y - cheek_y_avg

    # brow_furrow: inner brows converge in anger/sadness/fear
    brow_furrow = _euclidean(lm[LEFT_BROW_INNER], lm[RIGHT_BROW_INNER])

    return {
        "mar":         mar,
        "ear":         ear,
        "brow_raise":  brow,
        "smile_ratio": smile_ratio,
        "cheek_raise": cheek_raise,
        "brow_furrow": brow_furrow,
    }


def extract_features(video_path: str, audio_path: str) -> dict:
    """
    Extract the exact same feature vector produced by video_feature_extraction.py.
    Returns a flat dict ready to be turned into a single-row DataFrame.
    """
    # ── Audio ──────────────────────────────────────────────────────────────
    os.system(f'ffmpeg -i "{video_path}" -q:a 0 -map a '
              f'"{audio_path}" -y -loglevel quiet')

    y, sr = librosa.load(audio_path, sr=None)
    audio_duration = len(y) / sr

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

    rms         = librosa.feature.rms(y=y)
    energy_mean = float(np.mean(rms))
    energy_std  = float(np.std(rms))

    spec_centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spec_rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr            = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # ── NEW valence-specific audio features (must match video_feature_extraction.py) ──

    # Harmonics-to-Noise Ratio
    harmonic          = librosa.effects.harmonic(y)
    percussive        = librosa.effects.percussive(y)
    harmonic_energy   = float(np.mean(harmonic   ** 2) + 1e-10)
    percussive_energy = float(np.mean(percussive ** 2) + 1e-10)
    hnr               = float(10 * np.log10(harmonic_energy / percussive_energy))

    # Spectral flatness
    spec_flatness      = librosa.feature.spectral_flatness(y=y)
    spec_flatness_mean = float(np.mean(spec_flatness))
    spec_flatness_std  = float(np.std(spec_flatness))

    # Chroma features
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)   # 12 values
    chroma_std  = float(np.std(chroma_mean))
    chroma_max  = float(np.max(chroma_mean))
    chroma_entropy = float(
        -np.sum(chroma_mean / (chroma_mean.sum() + 1e-10) *
                np.log(chroma_mean / (chroma_mean.sum() + 1e-10) + 1e-10))
    )

    # Mel-spectrogram summary
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db   = librosa.power_to_db(mel_spec, ref=np.max)
    mel_mean = float(np.mean(mel_db))
    mel_std  = float(np.std(mel_db))
    mel_skew = float(np.mean(((mel_db - mel_mean) / (mel_std + 1e-10)) ** 3))

    # Pitch slope (rising = happy, falling = sad)
    if len(pitch_voiced) > 1:
        t           = np.linspace(0, 1, len(pitch_voiced))
        pitch_slope = float(np.polyfit(t, pitch_voiced, 1)[0])
    else:
        pitch_slope = 0.0

    # Jitter (cycle-to-cycle F0 variation)
    if len(pitch_voiced) > 2:
        jitter = float(np.mean(np.abs(np.diff(pitch_voiced))) / (pitch_mean + 1e-6))
    else:
        jitter = 0.0

    # ── Visual ─────────────────────────────────────────────────────────────
    cap         = cv2.VideoCapture(video_path)
    frame_feats = []

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
        face_detected    = 1.0
        face_mar_mean    = float(ff["mar"].mean())
        face_mar_std     = float(ff["mar"].std())
        face_ear_mean    = float(ff["ear"].mean())
        face_ear_std     = float(ff["ear"].std())
        face_brow_mean   = float(ff["brow_raise"].mean())
        face_brow_std    = float(ff["brow_raise"].std())
        face_smile_mean  = float(ff["smile_ratio"].mean())
        face_smile_std   = float(ff["smile_ratio"].std())
        face_cheek_mean  = float(ff["cheek_raise"].mean())
        face_cheek_std   = float(ff["cheek_raise"].std())
        face_furrow_mean = float(ff["brow_furrow"].mean())
        face_furrow_std  = float(ff["brow_furrow"].std())
    else:
        face_detected    = 0.0
        face_mar_mean    = face_mar_std    = 0.0
        face_ear_mean    = face_ear_std    = 0.0
        face_brow_mean   = face_brow_std   = 0.0
        face_smile_mean  = face_smile_std  = 0.0
        face_cheek_mean  = face_cheek_std  = 0.0
        face_furrow_mean = face_furrow_std = 0.0

    return {
        **{f"mfcc_mean_{i+1}":   float(mfccs_mean[i])  for i in range(13)},
        **{f"mfcc_std_{i+1}":    float(mfccs_std[i])   for i in range(13)},
        **{f"mfcc_delta_{i+1}":  float(delta_mean[i])  for i in range(13)},
        **{f"mfcc_delta2_{i+1}": float(delta2_mean[i]) for i in range(13)},
        "pitch_mean":   pitch_mean,   "pitch_std":    pitch_std,
        "pitch_range":  pitch_range,  "voiced_ratio": voiced_ratio,
        "energy_mean":  energy_mean,  "energy_std":   energy_std,
        "spec_centroid":  spec_centroid,
        "spec_bandwidth": spec_bandwidth,
        "spec_rolloff":   spec_rolloff,
        "zcr":            zcr,
        # NEW audio features
        "hnr":                hnr,
        "spec_flatness_mean": spec_flatness_mean,
        "spec_flatness_std":  spec_flatness_std,
        "chroma_std":         chroma_std,
        "chroma_max":         chroma_max,
        "chroma_entropy":     chroma_entropy,
        **{f"chroma_mean_{i+1}": float(chroma_mean[i]) for i in range(12)},
        "mel_mean":    mel_mean,
        "mel_std":     mel_std,
        "mel_skew":    mel_skew,
        "pitch_slope": pitch_slope,
        "jitter":      jitter,
        # Visual features (original)
        "face_detected":  face_detected,
        "face_mar_mean":  face_mar_mean,  "face_mar_std":  face_mar_std,
        "face_ear_mean":  face_ear_mean,  "face_ear_std":  face_ear_std,
        "face_brow_mean": face_brow_mean, "face_brow_std": face_brow_std,
        # NEW visual features
        "face_smile_mean":  face_smile_mean,  "face_smile_std":  face_smile_std,
        "face_cheek_mean":  face_cheek_mean,  "face_cheek_std":  face_cheek_std,
        "face_furrow_mean": face_furrow_mean, "face_furrow_std": face_furrow_std,
        # Text features — overwritten after Whisper runs in the route handlers
        "text_has_transcript": 0.0, "text_word_count": 0.0,
        "text_speaking_rate":  0.0, "text_exclamation_ratio": 0.0,
        "text_question_ratio": 0.0, "text_lex_valence": 0.0,
        "text_lex_valence_std":0.0, "text_sentiment_words": 0.0,
        "text_avg_sent_length":0.0, "text_caps_ratio": 0.0,
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


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based acoustic signal  (bypasses model uncertainty)
# ─────────────────────────────────────────────────────────────────────────────
# When the MLP is uncertain (outputs near its mean), raw acoustic features
# still carry clear emotional signal. We compute a lightweight rule-based
# estimate and blend it with the model output so obvious emotions always land.

# Blend weight: how much the rule-based signal contributes (0=model only, 1=rules only)
_RULE_BLEND = 0.45

def _rule_based_va(feats: dict):
    """
    Derive (valence, arousal) directly from acoustic/visual features.
    Weights calibrated against real CREMA-D clip debug output.

    Key signals confirmed from data:
      HNR:         -3.45 (angry) vs +2.59 (happy)  — strongest valence signal
      pitch_slope: -15.8 (angry) vs +5.93 (happy)  — falling=negative, rising=positive
      face_smile:   0.00 (angry) vs +0.11 (happy)  — clear visual discriminator
      energy/zcr:  slightly higher for angry        — arousal signal
    """
    energy   = feats.get("energy_mean",      0.0)
    zcr      = feats.get("zcr",              0.0)
    pitch_s  = feats.get("pitch_std",        0.0)
    pitch_sl = feats.get("pitch_slope",      0.0)
    hnr      = feats.get("hnr",              0.0)
    smile    = feats.get("face_smile_mean",  0.0)
    furrow   = feats.get("face_furrow_mean", 0.0)
    text_v   = feats.get("text_lex_valence", 0.0)

    # ── Arousal ───────────────────────────────────────────────────────────────
    e_score   = float(np.clip(energy / 0.015, 0, 1))   # 0.015 RMS ≈ loud speech
    zcr_score = float(np.clip(zcr    / 0.12,  0, 1))   # 0.12 ≈ energetic
    ps_score  = float(np.clip(pitch_s/ 80.0,  0, 1))   # 80 Hz std ≈ expressive
    raw_arousal = e_score*0.50 + zcr_score*0.30 + ps_score*0.20
    raw_arousal = 0.017 + raw_arousal * (0.74 - 0.017)

    # ── Valence ───────────────────────────────────────────────────────────────
    # HNR: ±8 dB maps to ±1  (confirmed: -3.45 angry, +2.59 happy)
    hnr_score   = float(np.clip(hnr / 8.0, -1, 1))
    # pitch_slope: ±20 Hz/s maps to ±1  (confirmed: -15.8 angry, +5.9 happy)
    slope_score = float(np.clip(pitch_sl / 20.0, -1, 1))
    # smile: 0.1 unit → 0.6 score  (confirmed: 0.0 angry, 0.11 happy)
    smile_score = float(np.clip(smile * 6, -1, 1))
    # furrow: deviation from neutral 0.048 baseline
    furrow_score= float(np.clip(-(furrow - 0.048) * 30, -1, 1))

    raw_valence = (hnr_score   * 0.40 +
                   slope_score * 0.30 +
                   smile_score * 0.20 +
                   furrow_score* 0.10)
    raw_valence = float(np.clip(raw_valence * 0.70, -0.665, 0.712))

    # Stretch to display space for label
    v = _stretch_valence(raw_valence)
    a = max(_stretch_arousal(raw_arousal), _AROUSAL_FLOOR)
    if abs(text_v) > 0.1:
        v = v * (1 - _TEXT_VALENCE_BLEND) + text_v * _TEXT_VALENCE_BLEND
    v = float(np.clip(v, -1, 1))
    a = float(np.clip(a,  0, 1))
    return raw_valence, raw_arousal, emotion_label(v, a)


# ─────────────────────────────────────────────────────────────────────────────
# Post-prediction sensitivity layer
# ─────────────────────────────────────────────────────────────────────────────
# Two different strategies for the two dimensions:
#
# VALENCE — tanh stretch centered on the true training median (-0.04).
#   Percentile ladders failed here because CREMA-D valence clusters near 0 and
#   the segment boundaries pushed the median into positive territory.
#   tanh(x/scale) gives a smooth S-curve: median→0, ±1 std→±0.65, extremes→±1.
#
# AROUSAL — percentile ladder normalisation.
#   Works well because arousal has a clear low-skewed shape we can map onto
#   equal-occupancy buckets. P75 raised to 0.42 so average clips land in MID.

_V_CENTER = -0.04   # true median of raw valence output (from training distribution)
_V_SCALE  =  0.32   # approx 1 std of raw valence; controls how fast tanh saturates

_A_PCTS = [0.017, 0.06, 0.13, 0.22, 0.42, 0.74]   # p0,p10,p25,p50,p75,p100
_A_CONTRAST = 0.92  # cubic exponent inside each segment (1.0=linear, lower=more contrast)

_AROUSAL_FLOOR = 0.15          # never report arousal below this
_TEXT_VALENCE_BLEND = 0.30     # how much Whisper sentiment shifts valence


def _stretch_valence(rv: float) -> float:
    """tanh S-curve centered on training median. Neutral raw → ~0 stretched."""
    t = (rv - _V_CENTER) / (_V_SCALE + 1e-9)
    return float(np.clip(np.tanh(t * 0.9), -1.0, 1.0))


def _stretch_arousal(ra: float) -> float:
    """Percentile ladder: maps training distribution onto equal-width [0,1] buckets."""
    bp = _A_PCTS
    n  = len(bp) - 1
    for i in range(n):
        if ra <= bp[i + 1] or i == n - 1:
            lo, hi = bp[i], bp[i + 1]
            t = float(np.clip((ra - lo) / (hi - lo + 1e-9), 0.0, 1.0))
            t = float(np.sign(t - 0.5) * (abs(t - 0.5) ** _A_CONTRAST) + 0.5)
            seg_lo = i / n
            seg_hi = (i + 1) / n
            return float(np.clip(seg_lo + t * (seg_hi - seg_lo), 0.0, 1.0))
    return 1.0


def stretch_va(raw_valence: float, raw_arousal: float,
               text_lex_valence: float = 0.0, feats: dict = None):
    v = _stretch_valence(raw_valence)
    a = max(_stretch_arousal(raw_arousal), _AROUSAL_FLOOR)

    # Blend rule-based signal to correct for model mean-regression
    if feats is not None:
        rb_v_raw, rb_a_raw, _ = _rule_based_va(feats)
        rb_v = _stretch_valence(rb_v_raw)
        rb_a = max(_stretch_arousal(rb_a_raw), _AROUSAL_FLOOR)
        v = v * (1 - _RULE_BLEND) + rb_v * _RULE_BLEND
        a = a * (1 - _RULE_BLEND) + rb_a * _RULE_BLEND

    # Blend Whisper sentiment into valence when speech content is clear
    if abs(text_lex_valence) > 0.1:
        v = v * (1 - _TEXT_VALENCE_BLEND) + text_lex_valence * _TEXT_VALENCE_BLEND

    v = float(np.clip(v, -1.0, 1.0))
    a = float(np.clip(a,  0.0,  1.0))
    print(f"  [stretch] raw=({raw_valence:+.3f}, {raw_arousal:.3f})  "
          f"text_v={text_lex_valence:+.2f}  → ({v:+.3f}, {a:.3f})")
    return v, a


# ─────────────────────────────────────────────────────────────────────────────
# Emotion label — 3×3 grid, balanced thresholds
# ─────────────────────────────────────────────────────────────────────────────

def emotion_label(valence: float, arousal: float) -> str:
    """
        arousal↑   neg-val      neu-val    pos-val
        high        Angry        Tense      Excited
        mid         Sad          Neutral    Happy
        low         Melancholic  Calm       Content
    """
    v_bin = "pos" if valence >  0.15 else ("neg" if valence < -0.15 else "neu")
    a_bin = "hi"  if arousal >  0.65 else ("lo"  if arousal <  0.32  else "mid")

    return {
        ("pos", "hi"): "Excited",     ("pos", "mid"): "Happy",   ("pos", "lo"): "Content",
        ("neu", "hi"): "Tense",       ("neu", "mid"): "Neutral",  ("neu", "lo"): "Calm",
        ("neg", "hi"): "Angry",       ("neg", "mid"): "Sad",      ("neg", "lo"): "Melancholic",
    }[(v_bin, a_bin)]


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/debug", methods=["POST"])
def debug():
    """POST a video and get raw model outputs + all key feature values."""
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400
    try:
        _load_models()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    video_file = request.files["video"]
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "clip.webm")
        audio_path = os.path.join(tmp, "clip.wav")
        video_file.save(video_path)
        feats = extract_features(video_path, audio_path)
        duration = feats.pop("_audio_duration", 5.0)
        feats.update(transcribe_and_extract(audio_path, duration))

        row = np.array([[feats.get(k, 0.0) for k in _feature_cols]], dtype=np.float32)
        row_scaled = _scaler.transform(row)

        with torch.no_grad():
            t       = torch.tensor(row_scaled).to(_torch_device)
            shared  = _va_model.shared(t)
            v_raw   = float(_va_model.valence_head(shared).cpu().item())
            a_logit = float(_va_model.arousal_head(shared).cpu().item())
            a_raw   = float(torch.sigmoid(_va_model.arousal_head(shared)).cpu().item())

        # Compute rule-based signal for comparison
        rb_v, rb_a, rb_label = _rule_based_va(feats)
        sv, sa = stretch_va(v_raw, a_raw, feats.get("text_lex_valence", 0.0))

    return jsonify({
        # Model outputs
        "raw_valence":        round(v_raw,   4),
        "raw_arousal":        round(a_raw,   4),
        "arousal_logit":      round(a_logit, 4),
        "stretched_valence":  round(sv, 4),
        "stretched_arousal":  round(sa, 4),
        "model_label":        emotion_label(sv, sa),
        # Rule-based override for comparison
        "rule_valence":       round(rb_v, 4),
        "rule_arousal":       round(rb_a, 4),
        "rule_label":         rb_label,
        # Key acoustic features — are these varying between clips?
        "energy_mean":        round(feats.get("energy_mean",      0), 6),
        "energy_std":         round(feats.get("energy_std",       0), 6),
        "pitch_mean":         round(feats.get("pitch_mean",       0), 2),
        "pitch_std":          round(feats.get("pitch_std",        0), 2),
        "zcr":                round(feats.get("zcr",              0), 5),
        "hnr":                round(feats.get("hnr",              0), 3),
        "spec_centroid":      round(feats.get("spec_centroid",    0), 1),
        "jitter":             round(feats.get("jitter",           0), 5),
        "pitch_slope":        round(feats.get("pitch_slope",      0), 3),
        # Visual features
        "face_detected":      feats.get("face_detected", 0),
        "face_smile_mean":    round(feats.get("face_smile_mean",  0), 4),
        "face_furrow_mean":   round(feats.get("face_furrow_mean", 0), 4),
        # Text
        "text_lex_valence":   round(feats.get("text_lex_valence", 0), 3),
        # Sanity
        "feature_count":      len(_feature_cols),
        "nonzero_features":   int(np.count_nonzero(row[0])),
        "audio_duration_s":   round(duration, 2),
    })


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
        raw_v = float(np.clip(pred[0], -1.0, 1.0))
        raw_a = float(np.clip(pred[1],  0.0, 1.0))
        valence, arousal = stretch_va(raw_v, raw_a, feats.get("text_lex_valence", 0.0), feats)
        tempo    = _va_to_tempo(arousal)
        label    = emotion_label(valence, arousal)

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
        raw_v = float(np.clip(pred[0], -1.0, 1.0))
        raw_a = float(np.clip(pred[1],  0.0, 1.0))
        valence, arousal = stretch_va(raw_v, raw_a, feats.get("text_lex_valence", 0.0), feats)
    scale = _va_to_scale(valence)
    tempo = _va_to_tempo(arousal)
    label = emotion_label(valence, arousal)

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
