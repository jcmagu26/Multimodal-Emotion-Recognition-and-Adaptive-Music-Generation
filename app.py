"""
app.py  –  Emotion-Adaptive Music  |  Flask Backend
────────────────────────────────────────────────────
Receives a video clip from the browser, runs your exact feature extraction
pipeline (audio + MediaPipe facial landmarks), feeds the features into the
trained MLP models, then generates and returns a MIDI file.

Setup
─────
  pip install flask flask-cors librosa opencv-python mediapipe midiutil
              scikit-learn pandas numpy joblib torch

Train & save models first (run once):
  python train_and_save.py

Run:
  python app.py
  # → http://localhost:5000

Endpoints
─────────
  POST /predict   multipart/form-data  { video: <file> }
                  → JSON { valence, arousal, scale, tempo_bpm, emotion_label }

  POST /generate  multipart/form-data  { video: <file> }
                  → JSON { valence, arousal, scale, tempo_bpm, emotion_label,
                           midi_b64, root_name }

  POST /debug     multipart/form-data  { video: <file> }
                  → JSON (full diagnostic dump)

  GET  /health    → JSON { status: "ok" }
"""

import base64
import io
import os
import random
import subprocess
import tempfile
import traceback

import cv2
import joblib
import librosa
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
SOUNDFONT_PATH   = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe setup
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

# ─────────────────────────────────────────────────────────────────────────────
# Model (lazy-loaded on first request)
# ─────────────────────────────────────────────────────────────────────────────

_va_model      = None
_scaler        = None
_feature_cols  = None
_whisper_model = None

_torch_device = torch.device("mps"  if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")


class VAPredictor(nn.Module):
    """Must match architecture in train_and_save.py exactly."""
    def __init__(self, input_dim, hidden, n_classes=6, dropout=0.0):
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
        self.emotion_head = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        h  = self.shared(x)
        va = torch.cat([self.valence_head(h),
                        torch.sigmoid(self.arousal_head(h))], dim=1)
        return va, self.emotion_head(h)


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
    cfg   = joblib.load(cfg_path)
    model = VAPredictor(cfg["input_dim"], cfg["hidden"],
                        n_classes=cfg.get("n_classes", 6), dropout=cfg["dropout"])
    ckpt        = torch.load(mlp_path, map_location=_torch_device)
    load_result = model.load_state_dict(ckpt, strict=False)
    if load_result.unexpected_keys:
        print(f"  [model] ignored extra checkpoint keys: {load_result.unexpected_keys}")
    if load_result.missing_keys:
        print(f"  [model] WARNING — missing keys (will be random): {load_result.missing_keys}")
    model.to(_torch_device).eval()
    _va_model     = model
    _scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    _feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))
    print(f"MLP loaded ({cfg['input_dim']} features, {cfg.get('n_classes',6)} classes) on {_torch_device}")


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
# Landmark indices
# ─────────────────────────────────────────────────────────────────────────────

MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LEFT_EYE_TOP,  LEFT_EYE_BOTTOM  = 159, 145
LEFT_EYE_LEFT, LEFT_EYE_RIGHT   = 33,  133
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 362, 263
LEFT_BROW_INNER  = 107
RIGHT_BROW_INNER = 336

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

    l_ear = _euclidean(lm[LEFT_EYE_TOP],  lm[LEFT_EYE_BOTTOM]) / (
            _euclidean(lm[LEFT_EYE_LEFT], lm[LEFT_EYE_RIGHT])  + 1e-6)
    r_ear = _euclidean(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOTTOM]) / (
            _euclidean(lm[RIGHT_EYE_LEFT],lm[RIGHT_EYE_RIGHT]) + 1e-6)
    ear   = (l_ear + r_ear) / 2.0

    brow  = (_euclidean(lm[LEFT_BROW_INNER],  lm[LEFT_EYE_LEFT]) +
             _euclidean(lm[RIGHT_BROW_INNER], lm[RIGHT_EYE_RIGHT])) / 2.0

    lip_center_y = (lm[UPPER_LIP_CENTER].y + lm[LOWER_LIP_CENTER].y) / 2.0
    corner_y_avg = (lm[MOUTH_CORNER_LEFT].y + lm[MOUTH_CORNER_RIGHT].y) / 2.0
    smile_ratio  = (lip_center_y - corner_y_avg) / (mouth_width + 1e-6)

    nose_y      = lm[NOSE_TIP].y
    cheek_y_avg = (lm[LEFT_CHEEK].y + lm[RIGHT_CHEEK].y) / 2.0
    cheek_raise = nose_y - cheek_y_avg

    brow_furrow = _euclidean(lm[LEFT_BROW_INNER], lm[RIGHT_BROW_INNER])

    return {
        "mar":         mar,
        "ear":         ear,
        "brow_raise":  brow,
        "smile_ratio": smile_ratio,
        "cheek_raise": cheek_raise,
        "brow_furrow": brow_furrow,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(video_path: str, audio_path: str) -> dict:
    """
    Extract the exact same feature vector produced by video_feature_extraction.py.
    Returns a flat dict ready to be turned into a single-row DataFrame.
    """
    # ── Audio ──────────────────────────────────────────────────────────────
    def _extract_wav(src, dst):
        for extra in (["-map", "0:0"], ["-map", "a:0"], ["-vn"]):
            r = subprocess.run(
                ["ffmpeg", "-i", src] + extra +
                ["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", dst, "-y", "-loglevel", "error"],
                capture_output=True, text=True
            )
            if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
                return
            if os.path.exists(dst):
                os.remove(dst)
        raise RuntimeError(f"ffmpeg failed: {r.stderr[:200]}")

    _extract_wav(video_path, audio_path)
    y, sr = librosa.load(audio_path, sr=16000)
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

    harmonic          = librosa.effects.harmonic(y)
    percussive        = librosa.effects.percussive(y)
    harmonic_energy   = float(np.mean(harmonic   ** 2) + 1e-10)
    percussive_energy = float(np.mean(percussive ** 2) + 1e-10)
    hnr               = float(10 * np.log10(harmonic_energy / percussive_energy))

    spec_flatness      = librosa.feature.spectral_flatness(y=y)
    spec_flatness_mean = float(np.mean(spec_flatness))
    spec_flatness_std  = float(np.std(spec_flatness))

    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std  = float(np.std(chroma_mean))
    chroma_max  = float(np.max(chroma_mean))
    chroma_entropy = float(
        -np.sum(chroma_mean / (chroma_mean.sum() + 1e-10) *
                np.log(chroma_mean / (chroma_mean.sum() + 1e-10) + 1e-10))
    )

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db   = librosa.power_to_db(mel_spec, ref=np.max)
    mel_mean = float(np.mean(mel_db))
    mel_std  = float(np.std(mel_db))
    mel_skew = float(np.mean(((mel_db - mel_mean) / (mel_std + 1e-10)) ** 3))

    if len(pitch_voiced) > 1:
        t           = np.linspace(0, 1, len(pitch_voiced))
        pitch_slope = float(np.polyfit(t, pitch_voiced, 1)[0])
    else:
        pitch_slope = 0.0

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
        "face_detected":  face_detected,
        "face_mar_mean":  face_mar_mean,  "face_mar_std":  face_mar_std,
        "face_ear_mean":  face_ear_mean,  "face_ear_std":  face_ear_std,
        "face_brow_mean": face_brow_mean, "face_brow_std": face_brow_std,
        "face_smile_mean":  face_smile_mean,  "face_smile_std":  face_smile_std,
        "face_cheek_mean":  face_cheek_mean,  "face_cheek_std":  face_cheek_std,
        "face_furrow_mean": face_furrow_mean, "face_furrow_std": face_furrow_std,
        # Text features — overwritten after Whisper runs in route handlers
        "text_has_transcript": 0.0, "text_word_count": 0.0,
        "text_speaking_rate":  0.0, "text_exclamation_ratio": 0.0,
        "text_question_ratio": 0.0, "text_lex_valence": 0.0,
        "text_lex_valence_std":0.0, "text_sentiment_words": 0.0,
        "text_avg_sent_length":0.0, "text_caps_ratio": 0.0,
        "_audio_duration": audio_duration,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Text feature extraction  (mirrors transcribe_features.py)
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

# Lexicon contains only UNAMBIGUOUSLY emotional words.
# Deliberately excluded: politeness/filler words that fire on casual webcam
# speech but carry no real emotional signal —
#   "okay", "ok", "sure", "yes", "right", "fine", "thanks", "thank", "welcome",
#   "bright", "warm" — these pushed neutral webcam input toward Happy.
# Words below must be things a person would say *because* they feel that emotion.
SENTIMENT_LEXICON = {
    # ── Strong positive ───────────────────────────────────────────────────────
    "happy":      0.90, "joy":       0.90, "joyful":    0.90,
    "love":       0.85, "wonderful": 0.85, "fantastic":  0.85,
    "excellent":  0.85, "amazing":   0.85, "brilliant":  0.80,
    "perfect":    0.85, "great":     0.80, "good":       0.70,
    "excited":    0.80, "glad":      0.65, "pleased":    0.70,
    "beautiful":  0.75, "nice":      0.55, "enjoy":      0.70,
    "wonderful":  0.85, "delightful":0.80, "thrilled":   0.85,
    "laugh":      0.65, "smile":     0.60,
    # ── Strong negative ───────────────────────────────────────────────────────
    "angry":     -0.85, "anger":    -0.85, "furious":   -0.90,
    "rage":      -0.90, "mad":      -0.75, "hate":      -0.90,
    "horrible":  -0.90, "terrible": -0.85, "awful":     -0.85,
    "disgusting":-0.85, "disgust":  -0.80, "disgusted": -0.85, "gross": -0.70,
    "scared":    -0.80, "afraid":   -0.75, "fear":      -0.80,
    "terrified": -0.90, "horrified":-0.90, "panic":     -0.80,
    "sad":       -0.85, "sadness":  -0.85, "miserable": -0.85,
    "depressed": -0.85, "lonely":   -0.80, "hopeless":  -0.85,
    "hurt":      -0.75, "pain":     -0.75, "suffer":    -0.80,
    "cry":       -0.70, "crying":   -0.70, "tears":     -0.65,
    "dead":      -0.85, "die":      -0.85, "kill":      -0.85,
    "ugly":      -0.65, "wrong":    -0.55, "bad":       -0.70,
    "fail":      -0.65, "failed":   -0.65, "useless":   -0.70,
    "upset":     -0.70, "anxious":  -0.70, "worried":   -0.65,
    "hate":      -0.90, "despise":  -0.85, "loathe":    -0.90,
    "never":     -0.40, "not":      -0.30,
    # ── Intensifiers (score > 1.0 flags them as multipliers, not sentiment) ──
    "very":      1.30, "really":    1.25, "so":         1.15,
    "extremely": 1.50, "absolutely":1.45, "totally":    1.30,
    "completely":1.30, "incredibly":1.40,
}


def extract_text_features(transcript: str, duration_seconds: float) -> dict:
    text  = transcript.strip().lower()
    words = _re.findall(r"[a-z']+", text)
    word_count    = len(words)
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


def transcribe_and_extract(audio_path: str, duration: float) -> tuple[dict, str]:
    """
    Run Whisper on audio and return (text_feature_dict, transcript_string).
    Falls back to (zeros_dict, "") if Whisper is unavailable.
    """
    wmodel = _load_whisper()
    if wmodel is None:
        return extract_text_features("", duration), ""
    try:
        import whisper
        audio      = whisper.load_audio(audio_path)
        result     = wmodel.transcribe(audio, fp16=False, language="en")
        transcript = result["text"].strip()
        print(f"  [whisper] '{transcript[:60]}'")
        return extract_text_features(transcript, duration), transcript
    except Exception as e:
        print(f"  [whisper] transcription failed: {e}")
        return extract_text_features("", duration), ""


# ─────────────────────────────────────────────────────────────────────────────
# Music generation
# ─────────────────────────────────────────────────────────────────────────────

SCALES = {
    "major":  [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "minor":  [0, 2, 3, 5, 7, 8, 10],
}

PROGRESSIONS = {
    "major":  [(0,0),(3,1),(4,0),(0,0),(5,1),(3,0),(4,2),(0,0)],
    "dorian": [(0,0),(3,0),(0,1),(4,1),(0,0),(5,1),(3,0),(0,0)],
    "minor":  [(0,0),(5,1),(3,0),(4,2),(0,0),(6,0),(3,1),(0,0)],
}

CONTOURS = {
    "happy":       [ 0, 2, 4, 2,  3, 5, 4, 2],
    "content":     [ 0, 1, 2, 1,  2, 1, 0, 0],
    "hopeful":     [ 0, 2, 1, 3,  2, 4, 3, 1],
    "pensive":     [ 0,-1, 0, 1,  0,-1,-2, 0],
    "tense":       [ 0, 3,-1, 2, -2, 4,-3, 1],
    "melancholic": [ 0,-1,-2,-1, -3,-2,-4,-2],
}

# Human-readable note names for each MIDI root, used in the UI
_MIDI_TO_NOTE = {
    56: "Ab", 58: "Bb", 60: "C", 61: "Db", 62: "D",
    63: "Eb", 64: "E", 65: "F", 67: "G", 69: "A",
}


def _va_to_scale(v): return "major" if v >= 0.05 else ("dorian" if v >= -0.3 else "minor")

def _va_to_root(v):
    bright  = [64, 67, 62, 69]
    neutral = [60, 65]
    dark    = [58, 63, 56, 61]
    pool = bright if v >= 0.2 else (neutral if v >= -0.2 else dark)
    return pool[int(abs(v) * (len(pool) - 1))]

def _va_to_root_name(v):
    root = _va_to_root(v)
    return _MIDI_TO_NOTE.get(root, str(root))

# Per-emotion tempo ranges — much wider spread for clear contrast
EMOTION_TEMPO = {
    "Happy":   (130, 170),   # bright, quick
    "Neutral": ( 90, 110),   # moderate
    "Angry":   ( 80, 110),   # heavy, driving but not fast
    "Fear":    ( 60,  85),   # slow, creeping dread
    "Disgust": ( 70,  90),   # slow, dragging
    "Sad":     ( 48,  68),   # very slow, sparse
}

# Per-emotion octave shift applied to ALL midi notes (+/- semitones from base)
EMOTION_PITCH_SHIFT = {
    "Happy":   +12,   # one octave up — bright, high register
    "Neutral":   0,   # no shift
    "Angry":    -5,   # lower, heavier
    "Fear":    -10,   # dark, low register
    "Disgust":  -7,   # low-mid
    "Sad":     -12,   # one octave down — deep, mournful
}

# Per-emotion velocity range (min, max)
EMOTION_VELOCITY = {
    "Happy":   (80, 115),   # loud, energetic
    "Neutral": (55,  80),
    "Angry":   (85, 120),   # hard, aggressive
    "Fear":    (30,  55),   # soft, trembling
    "Disgust": (45,  70),
    "Sad":     (25,  50),   # very soft, withdrawn
}

def _label_to_tempo(label: str, arousal: float) -> int:
    lo, hi = EMOTION_TEMPO.get(label, (80, 120))
    # arousal nudges within the label's range but cannot escape it
    t = float(np.clip(arousal, 0.0, 1.0))
    return int(lo + t * (hi - lo))

def _label_to_velocity(label: str, arousal: float) -> int:
    lo, hi = EMOTION_VELOCITY.get(label, (50, 90))
    t = float(np.clip(arousal, 0.0, 1.0))
    return int(lo + t * (hi - lo))

def _va_to_tempo(a):    return int(np.clip(60 + a * 120, 60, 180))   # kept for stretch_va compat
def _va_to_velocity(a): return int(np.clip(50 + a * 60,  50, 110))

def _emotion_region(label: str) -> str:
    """Map CREMA-D emotion label to a melodic contour region."""
    return {
        "Happy":   "happy",
        "Neutral": "content",
        "Fear":    "tense",
        "Angry":   "tense",
        "Disgust": "pensive",
        "Sad":     "melancholic",
    }.get(label, "content")

def _build_scale(root, ivls, octaves=3):
    return [root + i + o * 12 for o in range(octaves) for i in ivls]

def _build_chord_voiced(root, ivls, degree, inversion=0):
    triad_degs = [degree % 7, (degree + 2) % 7, (degree + 4) % 7]
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
        idx      = int(np.clip(start_idx + offset, 0, len(scale_notes) - 1))
        note     = scale_notes[idx]
        beat_pos = beat % 4
        vel_adj  = 8 if beat_pos == 0 else (4 if beat_pos == 2 else 0)
        vel      = int(np.clip(base_vel + vel_adj + rng.randint(-5, 5), 30, 120))
        rest_prob = max(0.0, 0.12 - arousal * 0.1)
        if rng.random() > rest_prob:
            artic = 0.55 if arousal > 0.65 else (0.80 if arousal > 0.35 else 0.92)
            events.append((note, beat, dur * artic, vel))
        beat += dur

    return events


def generate_midi(valence: float, arousal: float, label: str = "Neutral",
                  num_bars: int = 8, seed: int = 42) -> bytes:
    rng     = random.Random(seed)
    scale   = _va_to_scale(valence)
    root    = _va_to_root(valence)
    ivls    = SCALES[scale]
    prog    = PROGRESSIONS[scale]
    region  = _emotion_region(label)
    contour = CONTOURS[region]

    # ── Per-emotion tempo, velocity and pitch shift ───────────────────────────
    # These are the primary levers for making each emotion sound distinct.
    # Positive emotions: fast tempo, high register, loud.
    # Negative emotions: slow tempo, low register, soft (except Angry: loud+low).
    tempo      = _label_to_tempo(label, arousal)
    base_vel   = _label_to_velocity(label, arousal)
    pitch_shift = EMOTION_PITCH_SHIFT.get(label, 0)   # semitones added to every note

    # Per-emotion articulation: negative emotions get longer, more sustained notes
    ARTIC = {
        "Happy":   0.50,   # short, staccato — bouncy
        "Neutral": 0.80,
        "Angry":   0.65,   # clipped, punchy
        "Fear":    0.95,   # very sustained — slow, legato
        "Disgust": 0.90,
        "Sad":     0.98,   # nearly full duration — heavy, legato
    }
    artic = ARTIC.get(label, 0.80)

    # Per-emotion rhythm: how notes are subdivided per bar
    RHYTHMS = {
        "Happy":   [0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0],   # fast 8ths
        "Neutral": [1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0],
        "Angry":   [0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5],   # syncopated
        "Fear":    [2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0],   # slow, sparse
        "Disgust": [1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0],
        "Sad":     [2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 4.0, 4.0],   # very slow
    }
    rhythms = RHYTHMS.get(label, [1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0])

    # Instrument choices per emotion:
    #   channel 0 = melody, channel 1 = chord pad, channel 2 = bass
    INSTRUMENTS = {
        #              melody  pad   bass
        "Happy":   (     0,   48,   33),   # Grand Piano, Strings, Electric Bass
        "Neutral": (    11,   52,   32),   # Vibraphone, Choir, Acoustic Bass
        "Angry":   (    29,   44,   42),   # Overdriven Guitar, Tremolo Strings, Cello
        "Fear":    (    70,   44,   42),   # Bassoon, Tremolo Strings, Cello
        "Disgust": (    73,   49,   42),   # Flute, Slow Strings, Cello
        "Sad":     (    42,   49,   42),   # Cello melody, Slow Strings, Cello bass
    }
    mel_prog, pad_prog, bas_prog = INSTRUMENTS.get(label, (0, 48, 33))

    # Build scale — shift root down for dark emotions, up for bright ones
    shifted_root = root + pitch_shift
    scale_notes  = _build_scale(shifted_root, ivls, octaves=3)
    start_idx    = len(scale_notes) // 2

    midi = MIDIFile(numTracks=3)
    for t in range(3): midi.addTempo(t, 0, tempo)
    midi.addProgramChange(0, 0, 0, mel_prog)
    midi.addProgramChange(1, 1, 0, pad_prog)
    midi.addProgramChange(2, 2, 0, bas_prog)

    beats_per_bar    = 4
    beats_per_phrase = beats_per_bar * 4

    def section_vel_scale(bar):
        if bar < 2:               return 0.70
        elif bar >= num_bars - 2: return 0.75
        else:                     return 1.0

    def clamp_note(n):
        """Keep notes in a reasonable MIDI range."""
        return int(np.clip(n, 24, 108))

    # ── Track 0: Melody ───────────────────────────────────────────────────────
    for phrase_idx in range(num_bars // 4):
        phrase_start  = phrase_idx * beats_per_phrase
        phrase_offset = 0 if phrase_idx == 0 else (2 if valence > 0 else -2)
        p_start_idx   = int(np.clip(start_idx + phrase_offset, 0, len(scale_notes) - 1))

        beat = 0.0
        for i, offset in enumerate(contour):
            if beat >= beats_per_phrase: break
            dur     = min(rhythms[i % len(rhythms)], beats_per_phrase - beat)
            idx     = int(np.clip(p_start_idx + offset, 0, len(scale_notes) - 1))
            note    = clamp_note(scale_notes[idx])
            beat_pos= beat % 4
            vel_adj = 10 if beat_pos == 0 else (5 if beat_pos == 2 else 0)
            vel     = int(np.clip(base_vel + vel_adj + rng.randint(-6, 6), 20, 127))
            rest_prob = 0.05 if label in ("Happy", "Angry") else 0.15
            if rng.random() > rest_prob:
                bar = int((phrase_start + beat) // beats_per_bar)
                midi.addNote(0, 0, note,
                             phrase_start + beat,
                             dur * artic,
                             int(vel * section_vel_scale(bar)))
            beat += dur

    # ── Track 1: Chord pad ────────────────────────────────────────────────────
    chord_vel = max(20, base_vel - 25)
    for bar in range(num_bars):
        bt          = bar * beats_per_bar
        vs          = section_vel_scale(bar)
        degree, inv = prog[bar % len(prog)]
        chord = [clamp_note(n + pitch_shift)
                 for n in _build_chord_voiced(root, ivls, degree, inv)]
        cv = int(chord_vel * vs)

        if label in ("Happy", "Angry"):
            # Arpeggiated / broken chord for energy
            step = beats_per_bar / len(chord)
            for i, n in enumerate(chord):
                midi.addNote(1, 1, n, bt + i * step, step * 0.85, cv)
        elif label in ("Sad", "Fear", "Disgust"):
            # Long sustained whole-bar chord — mournful, still
            for n in chord:
                midi.addNote(1, 1, n, bt, beats_per_bar * 0.97, cv)
        else:
            # Two half-bar chords
            for n in chord:
                midi.addNote(1, 1, n, bt,   1.85, cv)
                midi.addNote(1, 1, n, bt+2, 1.85, int(cv * 0.85))

    # ── Track 2: Bass line ────────────────────────────────────────────────────
    bass_vel = max(25, base_vel - 15)
    for bar in range(num_bars):
        bt        = bar * beats_per_bar
        vs        = section_vel_scale(bar)
        degree, _ = prog[bar % len(prog)]
        bass_root  = clamp_note(_build_bass_note(root, ivls, degree) + pitch_shift)
        bass_fifth = clamp_note(bass_root + 7)
        bv         = int(bass_vel * vs)

        if label in ("Happy",):
            # Walking bass — active quarter notes
            midi.addNote(2, 2, bass_root,   bt,   0.85, bv)
            midi.addNote(2, 2, bass_root+2, bt+1, 0.85, int(bv * 0.80))
            midi.addNote(2, 2, bass_fifth,  bt+2, 0.85, bv)
            midi.addNote(2, 2, bass_fifth+2,bt+3, 0.85, int(bv * 0.80))
        elif label in ("Angry",):
            # Repeated root — heavy, relentless
            for beat_off in [0, 1, 2, 3]:
                midi.addNote(2, 2, bass_root, bt + beat_off, 0.7, int(bv * (1.0 if beat_off == 0 else 0.75)))
        elif label in ("Sad", "Fear"):
            # Single very long bass note per bar — minimal, heavy
            midi.addNote(2, 2, bass_root, bt, beats_per_bar * 0.97, bv)
        else:
            # Root + fifth, two half-bars
            midi.addNote(2, 2, bass_root,  bt,   1.85, bv)
            midi.addNote(2, 2, bass_fifth, bt+2, 1.85, int(bv * 0.90))

    buf = io.BytesIO()
    midi.writeFile(buf)
    return buf.getvalue()


def midi_to_wav(midi_bytes: bytes, soundfont: str = SOUNDFONT_PATH) -> bytes:
    """Render MIDI to WAV via FluidSynth. Returns WAV bytes or empty bytes."""
    import shutil
    if not shutil.which("fluidsynth"):
        print("FluidSynth not found — install with: brew install fluidsynth")
        return b""
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
# Rule-based acoustic signal
# ─────────────────────────────────────────────────────────────────────────────

# Separate blend weights for the two dimensions.
# Arousal: energy/ZCR/pitch_std reliably transfer from CREMA-D actors to webcam.
# Valence: smile_ratio/HNR/pitch_slope do NOT transfer well — webcam faces and
#          casual speech give systematically positive readings vs actor performances.
#          Setting valence blend to 0 means the MLP output alone drives valence.
_RULE_BLEND_AROUSAL = 0.00   # disabled — personal arousal stretch handles this
_RULE_BLEND_VALENCE = 0.00   # disable valence override — MLP is more reliable here

def _rule_based_va(feats: dict):
    energy   = feats.get("energy_mean",      0.0)
    zcr      = feats.get("zcr",              0.0)
    pitch_s  = feats.get("pitch_std",        0.0)
    pitch_sl = feats.get("pitch_slope",      0.0)
    hnr      = feats.get("hnr",              0.0)
    smile    = feats.get("face_smile_mean",  0.0)
    furrow   = feats.get("face_furrow_mean", 0.0)
    text_v   = feats.get("text_lex_valence", 0.0)

    e_score   = float(np.clip(energy / 0.015, 0, 1))
    zcr_score = float(np.clip(zcr    / 0.12,  0, 1))
    ps_score  = float(np.clip(pitch_s / 80.0, 0, 1))
    raw_arousal = e_score * 0.50 + zcr_score * 0.30 + ps_score * 0.20
    raw_arousal = 0.017 + raw_arousal * (0.74 - 0.017)

    hnr_score    = float(np.clip(hnr / 8.0,       -1, 1))
    slope_score  = float(np.clip(pitch_sl / 20.0, -1, 1))
    smile_score  = float(np.clip(smile * 6,        -1, 1))
    furrow_score = float(np.clip(-(furrow - 0.048) * 30, -1, 1))

    raw_valence = (hnr_score   * 0.40 +
                   slope_score * 0.30 +
                   smile_score * 0.20 +
                   furrow_score* 0.10)
    raw_valence = float(np.clip(raw_valence * 0.70, -0.665, 0.712))

    v = _stretch_valence(raw_valence)
    a = max(_stretch_arousal(raw_arousal), _AROUSAL_FLOOR)
    # text_v blend disabled (TEXT_VALENCE_BLEND=0.0) — see stretch_va comments
    v = float(np.clip(v, -1, 1))
    a = float(np.clip(a,  0, 1))
    return raw_valence, raw_arousal, emotion_label(v, a)


# ─────────────────────────────────────────────────────────────────────────────
# Post-prediction sensitivity layer
# ─────────────────────────────────────────────────────────────────────────────

# Webcam calibration — tuned to Jane's browser webcam raw_v/raw_a distribution
# (batch_test.py on 30 webcam clips, 5 per emotion, saved via Save Clip button)
_V_CENTER = +0.05
_V_SCALE  =  0.30

# Webcam arousal range is very compressed [0.08, 0.16]
_A_RAW_LO = 0.08
_A_RAW_HI = 0.16

_AROUSAL_FLOOR      = 0.15
# Text blend: re-enabled now that the lexicon only contains unambiguous emotion
# words. Gated in stretch_va — only applies when ≥2 sentiment words fired,
# preventing single-word false positives ("good morning" → "good" alone).
# Blend weight kept low (0.25) so it nudges rather than overrides the MLP.
# Text blend raised to 0.55: text is the dominant signal when it fires.
# With this weight, saying "I am angry" (text_v=-0.85) moves valence by ~0.36
# regardless of where the MLP sits — enough to always clear the Neutral boundary.
# The gate (≥0.75) still prevents "good morning" / filler words from firing.
_TEXT_VALENCE_BLEND = 0.55
_TEXT_MIN_WORDS     = 1      # at least 1 sentiment word must match
_TEXT_MIN_VAL       = 0.75   # score threshold: "good"=0.70 blocked, "great"/"angry" etc. pass


def _stretch_valence(rv: float) -> float:
    # Slope reduced from 0.9 → 0.65: gentler curve gives more resolution
    # in the negative range, preventing Angry clips from compressing into Fear.
    t = (rv - _V_CENTER) / (_V_SCALE + 1e-9)
    return float(np.clip(np.tanh(t * 0.65), -1.0, 1.0))


def _stretch_arousal(ra: float) -> float:
    t = float(np.clip((ra - _A_RAW_LO) / (_A_RAW_HI - _A_RAW_LO + 1e-9), 0.0, 1.0))
    return float(0.05 + t * 0.90)


def stretch_va(raw_valence: float, raw_arousal: float,
               text_lex_valence: float = 0.0, feats: dict = None):
    v = _stretch_valence(raw_valence)
    a = max(_stretch_arousal(raw_arousal), _AROUSAL_FLOOR)

    if feats is not None:
        rb_v_raw, rb_a_raw, _ = _rule_based_va(feats)
        rb_v = _stretch_valence(rb_v_raw)
        rb_a = max(_stretch_arousal(rb_a_raw), _AROUSAL_FLOOR)
        # Valence: MLP only (rule-based valence is miscalibrated for webcam input)
        # Arousal: blend MLP + rule-based (energy/ZCR transfer well across domains)
        v = v * (1 - _RULE_BLEND_VALENCE) + rb_v * _RULE_BLEND_VALENCE
        a = a * (1 - _RULE_BLEND_AROUSAL) + rb_a * _RULE_BLEND_AROUSAL

    # Gate: apply text blend only when the lexicon fired on genuinely emotional words.
    # Conditions:
    #   1. At least _TEXT_MIN_WORDS (1) sentiment words matched
    #   2. |val| >= _TEXT_MIN_VAL (0.75) — "good" alone scores 0.70, won't reach this
    #      but "great", "happy", "sad", "angry", etc. all score ≥0.75
    # This prevents "good morning" (val=0.70) and "thank you" (val=0.0) from firing
    # while still catching "I feel great" (val=0.80) and "I'm so sad" (val=0.98).
    sentiment_word_count = feats.get("text_sentiment_words", 0) if feats else 0
    if abs(text_lex_valence) >= _TEXT_MIN_VAL and sentiment_word_count >= _TEXT_MIN_WORDS:
        v = v * (1 - _TEXT_VALENCE_BLEND) + text_lex_valence * _TEXT_VALENCE_BLEND

    v = float(np.clip(v, -1.0, 1.0))
    a = float(np.clip(a,  0.0,  1.0))
    sv_mlp = _stretch_valence(raw_valence)
    sw = int(feats.get("text_sentiment_words", 0)) if feats else 0
    text_applied = abs(text_lex_valence) >= _TEXT_MIN_VAL and sw >= _TEXT_MIN_WORDS
    print(f"  [stretch] mlp_raw={raw_valence:+.4f}  mlp_sv={sv_mlp:+.4f}  "
          f"text_v={text_lex_valence:+.3f}({'applied' if text_applied else 'skipped, sw='+str(sw)})  "
          f"final=({v:+.4f}, {a:.4f})  → {emotion_label(v, a)}")
    return v, a


# ─────────────────────────────────────────────────────────────────────────────
# Emotion label
# ─────────────────────────────────────────────────────────────────────────────

def emotion_label(valence: float, arousal: float) -> str:
    """
    Map (valence, arousal) to one of CREMA-D's 6 emotion categories:
    Happy, Sad, Angry, Fear, Disgust, Neutral.

    Decision rules (ordered most to least specific):
      1. Positive valence              → Happy
      2. Near-zero valence             → Neutral
      3. Negative + very high arousal  → Fear    (most physiologically intense)
      4. Negative + high arousal       → Angry
      5. Negative + moderate valence   → Disgust  (less extreme than Sad)
      6. Negative + low arousal        → Sad
    """
    # Thresholds calibrated empirically against the tanh stretch curve
    # (V_CENTER=+0.06, V_SCALE=0.35, slope=0.65):
    #
    #   raw_v > +0.18  → sv > +0.25   → Happy
    #   raw_v in [-0.12, +0.18] → sv in [-0.35, +0.25] → Neutral  (webcam resting faces)
    #   raw_v < -0.12, high arousal   → Angry
    #   raw_v < -0.52, very high arou → Fear
    #   raw_v in [-0.52, -0.12], low  → Disgust / Sad

    # Thresholds calibrated to V_CENTER=+0.20, V_SCALE=0.30, tanh slope=0.65.
    # With these settings:
    #   raw_v=+0.35 → sv=+0.31  (Happy)
    #   raw_v=+0.20 → sv=+0.00  (center of Neutral)
    #   raw_v= 0.00 → sv=-0.41  (mid Neutral)
    #   raw_v=-0.11 → sv=-0.59  (bottom of Neutral)
    #   raw_v=-0.15 → sv=-0.64  (Angry territory)
    #   raw_v=-0.40 → sv=-0.86  (Sad territory)
    #   raw_v=-0.50 → sv=-0.91  (Fear territory)

    # With text blend=0.55, "I am angry" (text_v=-0.85) + MLP near-zero (sv≈-0.21)
    # produces fv ≈ -0.21*0.45 + -0.85*0.55 = -0.56 → must be below Neutral boundary.
    # Neutral boundary tightened to -0.45 so emotional speech always clears it.
    # Without text (CREMA-D clips), MLP must output raw_v < -0.11 to leave Neutral.

    # Webcam-calibrated thresholds (batch_test.py on 30 webcam clips, 100% on calibration set)
    if valence > 0.30:   return "Happy"
    if valence > -0.30:  return "Neutral"
    if arousal > 0.70:
        return "Fear" if valence < -0.75 else "Angry"
    if arousal < 0.25:   return "Disgust"
    if valence < -0.75:  return "Sad"
    return "Neutral"


# ─────────────────────────────────────────────────────────────────────────────
# Shared inference helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference(video_file) -> dict:
    """
    Shared by /predict, /generate, and /debug.
    Saves the uploaded file, extracts features, runs Whisper + MLP,
    applies stretch_va, and returns a dict with everything needed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Preserve the original extension so ffmpeg can detect the container.
        orig_name   = getattr(video_file, "filename", "clip.webm") or "clip.webm"
        ext         = os.path.splitext(orig_name)[1] or ".webm"
        video_path  = os.path.join(tmp, f"clip{ext}")
        audio_path  = os.path.join(tmp, "clip.wav")
        video_file.save(video_path)

        feats    = extract_features(video_path, audio_path)
        duration = feats.pop("_audio_duration", 5.0)
        text_feats, transcript = transcribe_and_extract(audio_path, duration)
        feats.update(text_feats)

        row_data   = np.array([[feats.get(k, 0.0) for k in _feature_cols]],
                               dtype=np.float32)
        row_scaled = _scaler.transform(row_data)

        _EM_CLASSES  = ["ANG","DIS","FEA","HAP","NEU","SAD"]
        _EM_TO_LABEL = {"ANG":"Angry","DIS":"Disgust","FEA":"Fear",
                        "HAP":"Happy","NEU":"Neutral","SAD":"Sad"}

        with torch.no_grad():
            t      = torch.tensor(row_scaled).to(_torch_device)
            shared = _va_model.shared(t)
            v_raw  = float(_va_model.valence_head(shared).cpu().item())
            a_logit= float(_va_model.arousal_head(shared).cpu().item())
            a_raw  = float(torch.sigmoid(_va_model.arousal_head(shared)).cpu().item())
            em_logits = _va_model.emotion_head(shared).cpu().numpy()[0]
            em_probs  = np.exp(em_logits - em_logits.max())
            em_probs /= em_probs.sum()
            em_idx        = int(em_probs.argmax())
            em_top_class  = _EM_CLASSES[em_idx]
            em_confidence = float(em_probs[em_idx])
            em_head_label = _EM_TO_LABEL[em_top_class]

        valence, arousal = stretch_va(
            float(np.clip(v_raw, -1.0, 1.0)),
            float(np.clip(a_raw,  0.0, 1.0)),
            feats.get("text_lex_valence", 0.0),
            feats,
        )

        label    = emotion_label(valence, arousal)
        label_src = "VA"
        scale     = _va_to_scale(valence)
        tempo     = _va_to_tempo(arousal)
        root_name = _va_to_root_name(valence)
        rb_v, rb_a, rb_label = _rule_based_va(feats)

        return {
            "valence":       valence,
            "arousal":       arousal,
            "scale":         scale,
            "tempo_bpm":     tempo,
            "root_name":     root_name,
            "emotion_label": label,
            "raw_valence":   v_raw,
            "raw_arousal":   a_raw,
            "arousal_logit": a_logit,
            "em_top_class":  em_top_class,
            "em_confidence": em_confidence,
            "em_head_label": em_head_label,
            "label_src":     label_src,
            "rule_valence":  rb_v,
            "rule_arousal":  rb_a,
            "rule_label":    rb_label,
            "transcript":    transcript,
            "feats":         feats,
            "row":           row_data,
            "duration":      duration,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """Return VA prediction + music parameters (no MIDI)."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    try:
        _load_models()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    try:
        r = _run_inference(request.files["video"])
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}",
                        "detail": traceback.format_exc()}), 500

    # FIX #1: scale is now always present (computed inside _run_inference)
    return jsonify({
        "valence":       round(r["valence"],   4),
        "arousal":       round(r["arousal"],   4),
        "scale":         r["scale"],
        "tempo_bpm":     r["tempo_bpm"],
        "root_name":     r["root_name"],
        "emotion_label": r["emotion_label"],
    })


@app.route("/generate", methods=["POST"])
def generate():
    """Return VA prediction + base64-encoded MIDI."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    try:
        _load_models()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    try:
        r = _run_inference(request.files["video"])
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}",
                        "detail": traceback.format_exc()}), 500

    # FIX #3: scale/tempo/label now computed inside _run_inference (not after the
    # tempfile context closes), so valence/arousal are always in scope here.
    midi_bytes = generate_midi(r["valence"], r["arousal"], r["emotion_label"], num_bars=8)
    midi_b64   = base64.b64encode(midi_bytes).decode()

    return jsonify({
        "valence":       round(r["valence"],   4),
        "arousal":       round(r["arousal"],   4),
        "scale":         r["scale"],
        "tempo_bpm":     r["tempo_bpm"],
        "root_name":     r["root_name"],
        "emotion_label": r["emotion_label"],
        "transcript":    r.get("transcript", ""),
        "midi_b64":      midi_b64,
    })


@app.route("/debug", methods=["POST"])
def debug():
    """Return full diagnostic dump of model internals and features."""
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400
    try:
        _load_models()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    try:
        r = _run_inference(request.files["video"])
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}",
                        "detail": traceback.format_exc()}), 500

    feats = r["feats"]
    row   = r["row"]

    # FIX #2: rule-based values were already computed with feats inside
    # _run_inference, so debug output now exactly matches /generate.
    return jsonify({
        "raw_valence":        round(r["raw_valence"],   4),
        "raw_arousal":        round(r["raw_arousal"],   4),
        "arousal_logit":      round(r["arousal_logit"], 4),
        "stretched_valence":  round(r["valence"], 4),
        "stretched_arousal":  round(r["arousal"], 4),
        "model_label":        emotion_label(r["valence"], r["arousal"]),
        "em_top_class":       r.get("em_top_class", "?"),
        "em_confidence":      round(r.get("em_confidence", 0), 3),
        "em_head_label":      r.get("em_head_label", "?"),
        "label_src":          r.get("label_src", "?"),
        "rule_valence":       round(r["rule_valence"], 4),
        "rule_arousal":       round(r["rule_arousal"], 4),
        "rule_label":         r["rule_label"],
        "scale":              r["scale"],
        "root_name":          r["root_name"],
        "tempo_bpm":          r["tempo_bpm"],
        "emotion_label":      r["emotion_label"],
        "energy_mean":        round(feats.get("energy_mean",      0), 6),
        "energy_std":         round(feats.get("energy_std",       0), 6),
        "pitch_mean":         round(feats.get("pitch_mean",       0), 2),
        "pitch_std":          round(feats.get("pitch_std",        0), 2),
        "zcr":                round(feats.get("zcr",              0), 5),
        "hnr":                round(feats.get("hnr",              0), 3),
        "spec_centroid":      round(feats.get("spec_centroid",    0), 1),
        "jitter":             round(feats.get("jitter",           0), 5),
        "pitch_slope":        round(feats.get("pitch_slope",      0), 3),
        "face_detected":      feats.get("face_detected", 0),
        "face_smile_mean":    round(feats.get("face_smile_mean",  0), 4),
        "face_furrow_mean":   round(feats.get("face_furrow_mean", 0), 4),
        "text_lex_valence":   round(feats.get("text_lex_valence", 0), 3),
        "feature_count":      len(_feature_cols),
        "nonzero_features":   int(np.count_nonzero(row[0])),
        "audio_duration_s":   round(r["duration"], 2),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Emotion Music API on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
