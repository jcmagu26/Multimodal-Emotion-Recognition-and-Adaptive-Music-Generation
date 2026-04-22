"""
app.py  –  Emotion-Adaptive Music  |  Flask Backend
────────────────────────────────────────────────────
Receives a video clip from the browser, runs feature extraction
(audio + MediaPipe facial landmarks + optional Whisper text), feeds
the features into a trained MLP, then generates and returns a MIDI file.

Endpoints
─────────
  POST /predict   multipart/form-data  { video: <file> }
                  → JSON { valence, arousal, scale, root_name, tempo_bpm, emotion_label }

  POST /generate  multipart/form-data  { video: <file> }
                  → JSON { valence, arousal, scale, root_name, tempo_bpm, emotion_label, midi_b64 }

  POST /debug     multipart/form-data  { video: <file> }
                  → JSON (all raw features + model internals)

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
    def __init__(self, input_dim, hidden, dropout=0.0, n_classes=6):
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
        h = self.shared(x)
        v = self.valence_head(h)
        a = torch.sigmoid(self.arousal_head(h))
        e = self.emotion_head(h)
        return torch.cat([v, a], dim=1), e

_torch_device = torch.device("mps"  if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")

# Emotion classes in the same order as LabelEncoder in train_and_save.py
_EMOTION_CLASSES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
_EMOTION_NAMES   = {"ANG": "Angry", "DIS": "Disgust", "FEA": "Fear",
                    "HAP": "Happy", "NEU": "Neutral",  "SAD": "Sad"}
_label_encoder   = None

def _load_models():
    global _va_model, _scaler, _feature_cols, _label_encoder
    if _va_model is not None:
        return
    mlp_path = os.path.join(MODEL_DIR, "va_mlp.pt")
    cfg_path  = os.path.join(MODEL_DIR, "mlp_config.joblib")
    if not os.path.exists(mlp_path):
        raise FileNotFoundError(
            "Trained model not found. Run train_and_save.py first."
        )
    cfg     = joblib.load(cfg_path)
    n_cls   = cfg.get("n_classes", 6)
    model   = VAPredictor(cfg["input_dim"], cfg["hidden"], cfg["dropout"], n_cls)
    model.load_state_dict(torch.load(mlp_path, map_location=_torch_device))
    model.to(_torch_device).eval()
    _va_model     = model
    _scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    _feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))
    le_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    if os.path.exists(le_path):
        _label_encoder = joblib.load(le_path)
    print(f"MLP loaded ({cfg['input_dim']} features, {n_cls} classes) on {_torch_device}")

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
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a",
         audio_path, "-y", "-loglevel", "error"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract audio from {video_path}:\n{result.stderr[:400]}"
        )

    y, sr = librosa.load(audio_path, sr=None)
    audio_duration = len(y) / sr

    mfccs       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean  = np.mean(mfccs, axis=1)
    mfccs_std   = np.std(mfccs,  axis=1)
    delta       = librosa.feature.delta(mfccs)
    delta2      = librosa.feature.delta(mfccs, order=2)
    delta_mean  = np.mean(delta,  axis=1)
    delta2_mean = np.mean(delta2, axis=1)
    delta_std   = np.std(delta,   axis=1)   # NEW

    pitch        = librosa.yin(y, fmin=50, fmax=500)
    pitch_voiced = pitch[pitch > 0]
    pitch_mean   = float(np.mean(pitch_voiced))  if len(pitch_voiced) > 0 else 0.0
    pitch_std    = float(np.std(pitch_voiced))   if len(pitch_voiced) > 0 else 0.0
    pitch_range  = float(np.max(pitch_voiced) - np.min(pitch_voiced)) if len(pitch_voiced) > 0 else 0.0
    voiced_ratio = len(pitch_voiced) / (len(pitch) + 1e-6)

    # F0 percentiles (NEW)
    if len(pitch_voiced) >= 4:
        pitch_p10 = float(np.percentile(pitch_voiced, 10))
        pitch_p25 = float(np.percentile(pitch_voiced, 25))
        pitch_p75 = float(np.percentile(pitch_voiced, 75))
        pitch_p90 = float(np.percentile(pitch_voiced, 90))
    else:
        pitch_p10 = pitch_p25 = pitch_p75 = pitch_p90 = pitch_mean

    if len(pitch_voiced) > 1:
        t_arr       = np.linspace(0, 1, len(pitch_voiced))
        pitch_slope = float(np.polyfit(t_arr, pitch_voiced, 1)[0])
    else:
        pitch_slope = 0.0

    jitter = float(np.mean(np.abs(np.diff(pitch_voiced))) / (pitch_mean + 1e-6)) if len(pitch_voiced) > 2 else 0.0

    # Shimmer (NEW)
    rms_fr = librosa.feature.rms(y=y, frame_length=int(sr*0.025), hop_length=int(sr*0.010))[0]
    rms_v  = rms_fr[rms_fr > rms_fr.mean() * 0.1]
    shimmer = float(np.mean(np.abs(np.diff(rms_v))) / (np.mean(rms_v) + 1e-6)) if len(rms_v) > 2 else 0.0

    # Voiced transitions and pause ratio (NEW)
    voiced_mask       = (pitch > 0).astype(int)
    voiced_trans_rate = float(np.sum(np.abs(np.diff(voiced_mask))) / (len(voiced_mask) + 1e-6))
    pause_ratio       = float(1.0 - voiced_ratio)

    rms_all     = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms_all))
    energy_std  = float(np.std(rms_all))
    energy_slope = float(np.polyfit(np.linspace(0,1,len(rms_all)), rms_all, 1)[0]) if len(rms_all) > 1 else 0.0  # NEW

    spec_centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spec_rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr            = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # Spectral entropy and low-freq ratio (NEW)
    stft_mag = np.abs(librosa.stft(y))
    spec_norm = stft_mag / (stft_mag.sum(axis=0, keepdims=True) + 1e-10)
    spectral_entropy = float(np.mean(-np.sum(spec_norm * np.log(spec_norm + 1e-10), axis=0)))
    freqs    = librosa.fft_frequencies(sr=sr)
    lf_mask  = (freqs >= 100) & (freqs <= 300)
    sp2      = stft_mag ** 2
    lowfreq_energy_ratio = float(sp2[lf_mask].sum(axis=0).mean() / (sp2.sum(axis=0).mean() + 1e-10))

    harmonic   = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    hnr = float(10 * np.log10((np.mean(harmonic**2) + 1e-10) / (np.mean(percussive**2) + 1e-10)))

    # CPP (NEW)
    cepstrum  = np.real(np.fft.ifft(np.log(np.abs(librosa.stft(y)) + 1e-10), axis=0))
    quefrency = np.arange(cepstrum.shape[0]) / sr
    f0m = (quefrency > 0.002) & (quefrency < 0.02)
    if f0m.sum() > 0:
        y_vals   = np.abs(cepstrum[f0m]).mean(axis=1) if cepstrum.ndim > 1 else np.abs(cepstrum[f0m])
        x_idx    = np.where(f0m)[0]
        baseline = float(np.polyval(np.polyfit(x_idx, y_vals, 1), x_idx).mean())
        cpp      = float(np.max(y_vals) - baseline)
    else:
        cpp = 0.0

    spec_flatness      = librosa.feature.spectral_flatness(y=y)
    spec_flatness_mean = float(np.mean(spec_flatness))
    spec_flatness_std  = float(np.std(spec_flatness))

    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std  = float(np.std(chroma_mean))
    chroma_max  = float(np.max(chroma_mean))
    chroma_entropy = float(
        -np.sum(chroma_mean / (chroma_mean.sum() + 1e-10) *
                np.log(chroma_mean / (chroma_mean.sum() + 1e-10) + 1e-10)))

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db   = librosa.power_to_db(mel_spec, ref=np.max)
    mel_mean = float(np.mean(mel_db))
    mel_std  = float(np.std(mel_db))
    mel_skew = float(np.mean(((mel_db - mel_mean) / (mel_std + 1e-10)) ** 3))

    # ── Visual ─────────────────────────────────────────────────────────────────
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

    def _safe_std(series):
        v = float(series.std())
        return 0.0 if (v != v) else v

    if frame_feats:
        ff = pd.DataFrame(frame_feats)
        face_detected    = 1.0
        face_mar_mean    = float(ff["mar"].mean());        face_mar_std     = _safe_std(ff["mar"])
        face_ear_mean    = float(ff["ear"].mean());        face_ear_std     = _safe_std(ff["ear"])
        face_brow_mean   = float(ff["brow_raise"].mean()); face_brow_std    = _safe_std(ff["brow_raise"])
        face_smile_mean  = float(ff["smile_ratio"].mean());face_smile_std   = _safe_std(ff["smile_ratio"])
        face_cheek_mean  = float(ff["cheek_raise"].mean());face_cheek_std   = _safe_std(ff["cheek_raise"])
        face_furrow_mean = float(ff["brow_furrow"].mean());face_furrow_std  = _safe_std(ff["brow_furrow"])
    else:
        face_detected = 0.0
        face_mar_mean = face_mar_std = face_ear_mean = face_ear_std = 0.0
        face_brow_mean = face_brow_std = face_smile_mean = face_smile_std = 0.0
        face_cheek_mean = face_cheek_std = face_furrow_mean = face_furrow_std = 0.0

    return {
        **{f"mfcc_mean_{i+1}":      float(mfccs_mean[i])  for i in range(13)},
        **{f"mfcc_std_{i+1}":       float(mfccs_std[i])   for i in range(13)},
        **{f"mfcc_delta_{i+1}":     float(delta_mean[i])  for i in range(13)},
        **{f"mfcc_delta2_{i+1}":    float(delta2_mean[i]) for i in range(13)},
        **{f"mfcc_delta_std_{i+1}": float(delta_std[i])   for i in range(13)},  # NEW
        "pitch_mean":   pitch_mean,   "pitch_std":    pitch_std,
        "pitch_range":  pitch_range,  "voiced_ratio": voiced_ratio,
        "pitch_slope":  pitch_slope,  "jitter":       jitter,
        "shimmer":      shimmer,                                              # NEW
        "pitch_p10": pitch_p10, "pitch_p25": pitch_p25,                      # NEW
        "pitch_p75": pitch_p75, "pitch_p90": pitch_p90,                      # NEW
        "voiced_trans_rate": voiced_trans_rate, "pause_ratio": pause_ratio,  # NEW
        "energy_mean":  energy_mean,  "energy_std":  energy_std,
        "energy_slope": energy_slope,                                         # NEW
        "spec_centroid":  spec_centroid, "spec_bandwidth": spec_bandwidth,
        "spec_rolloff":   spec_rolloff,  "zcr": zcr,
        "spectral_entropy":     spectral_entropy,                             # NEW
        "lowfreq_energy_ratio": lowfreq_energy_ratio,                        # NEW
        "hnr": hnr, "cpp": cpp,                                              # cpp NEW
        "spec_flatness_mean": spec_flatness_mean,
        "spec_flatness_std":  spec_flatness_std,
        "chroma_std":     chroma_std, "chroma_max": chroma_max,
        "chroma_entropy": chroma_entropy,
        **{f"chroma_mean_{i+1}": float(chroma_mean[i]) for i in range(12)},
        "mel_mean": mel_mean, "mel_std": mel_std, "mel_skew": mel_skew,
        "face_detected":   face_detected,
        "face_mar_mean":   face_mar_mean,    "face_mar_std":   face_mar_std,
        "face_ear_mean":   face_ear_mean,    "face_ear_std":   face_ear_std,
        "face_brow_mean":  face_brow_mean,   "face_brow_std":  face_brow_std,
        "face_smile_mean": face_smile_mean,  "face_smile_std": face_smile_std,
        "face_cheek_mean": face_cheek_mean,  "face_cheek_std": face_cheek_std,
        "face_furrow_mean":face_furrow_mean, "face_furrow_std":face_furrow_std,
        # Text features — filled in by route handlers after Whisper
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
    "full": 0.2, "almost": 0.1, "new": 0.3, "like": 0.4, "think": 0.1,
    "stop": -0.2, "cold": -0.3, "forget": -0.3, "wonder": 0.2,
    "bad": -0.7, "sad": -0.8, "angry": -0.8, "hate": -0.9, "terrible": -0.85,
    "awful": -0.85, "horrible": -0.9, "wrong": -0.5, "no": -0.3, "not": -0.3,
    "never": -0.4, "fear": -0.75, "scared": -0.75, "afraid": -0.7,
    "disgust": -0.8, "disgusting": -0.85, "gross": -0.7, "ugly": -0.65,
    "hurt": -0.7, "pain": -0.75, "suffer": -0.8, "cry": -0.65, "dark": -0.4,
    "lonely": -0.75, "lost": -0.5, "fail": -0.65, "failed": -0.65,
    "dead": -0.85, "die": -0.85, "kill": -0.85, "mad": -0.7,
    "furious": -0.9, "rage": -0.9, "upset": -0.65, "miserable": -0.85,
    "slippery": -0.3,
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
    "major":       [0, 2, 4, 5, 7, 9, 11],
    "dorian":      [0, 2, 3, 5, 7, 9, 10],
    "minor":       [0, 2, 3, 5, 7, 8, 10],
    "lydian":      [0, 2, 4, 6, 7, 9, 11],   # dreamy/floating — high valence, low arousal
    "phrygian":    [0, 1, 3, 5, 7, 8, 10],   # dark/tense — low valence, high arousal
}

# (scale_degree, inversion, chord_quality) per bar -- 16 bars for full structure
# chord_quality: "triad", "open5", "sus2", "sus4"
# No 7th chords — they add richness that pushes toward symphonic.
# open5 (root + fifth only) is the most ambient, spacious voicing.
PROGRESSIONS = {
    "major":    [(0,0,"open5"),(3,0,"open5"),(4,0,"sus2"), (0,0,"open5"),
                 (5,0,"open5"),(3,0,"triad"),(1,0,"open5"),(4,0,"sus4"),
                 (0,0,"open5"),(5,0,"open5"),(3,0,"sus2"), (4,0,"open5"),
                 (6,0,"open5"),(2,0,"open5"),(4,0,"triad"),(0,0,"open5")],
    "dorian":   [(0,0,"open5"),(3,0,"sus2"), (0,0,"open5"),(4,0,"open5"),
                 (0,0,"open5"),(5,0,"open5"),(3,0,"sus4"), (2,0,"open5"),
                 (0,0,"open5"),(4,0,"open5"),(5,0,"sus2"), (3,0,"open5"),
                 (0,0,"open5"),(6,0,"open5"),(4,0,"open5"),(0,0,"sus2")],
    "minor":    [(0,0,"open5"),(5,0,"open5"),(3,0,"sus2"), (4,0,"open5"),
                 (0,0,"open5"),(6,0,"open5"),(3,0,"open5"),(4,0,"sus4"),
                 (0,0,"open5"),(2,0,"open5"),(5,0,"open5"),(3,0,"sus2"),
                 (6,0,"open5"),(4,0,"open5"),(5,0,"open5"),(0,0,"open5")],
    "lydian":   [(0,0,"open5"),(1,0,"sus2"), (4,0,"open5"),(0,0,"open5"),
                 (2,0,"open5"),(5,0,"open5"),(1,0,"sus4"), (0,0,"open5"),
                 (0,0,"open5"),(4,0,"open5"),(2,0,"sus2"), (5,0,"open5"),
                 (1,0,"open5"),(3,0,"open5"),(4,0,"open5"),(0,0,"open5")],
    "phrygian": [(0,0,"open5"),(1,0,"open5"),(5,0,"sus2"), (0,0,"open5"),
                 (3,0,"open5"),(1,0,"open5"),(5,0,"open5"),(0,0,"sus4"),
                 (0,0,"open5"),(4,0,"open5"),(1,0,"open5"),(3,0,"sus2"),
                 (5,0,"open5"),(1,0,"open5"),(4,0,"open5"),(0,0,"open5")],
}

# Relative scale-degree offsets for each phrase contour — longer, more varied
CONTOURS = {
    # Happy: stepwise ascending, bright and busy
    "happy":   [ 0, 2, 4, 2,  3, 5, 4, 2,  0, 1, 3, 5,  4, 2, 4, 6],
    # Neutral: minimal movement, hovering near root
    "neutral": [ 0, 1, 0,-1,  0, 1, 2, 0, -1, 0, 1, 0,  0,-1, 0, 1],
    # Sad: slow stepwise descent
    "sad":     [ 0,-1,-2,-1, -3,-2,-4,-2, -1,-2,-3,-2, -4,-3,-5,-3],
    # Angry: jagged violent leaps
    "angry":   [ 0,-3, 1,-4,  2,-2, 3,-5,  0,-3,-1,-4,  1,-2, 0,-4],
    # Disgust: lurching, uneven, dissonant landings
    "disgust": [ 0,-2, 1,-3, -1,-4, 0,-2, -3, 1,-2, 0, -4,-1,-3,-2],
    # Fear: erratic, wide leaps, no resolution
    "fear":    [ 0, 4,-3, 5, -4, 3,-5, 2,  4,-3, 5,-4,  3,-5, 1,-3],
}

PASSING_TONES = {
    "happy":   [ 1,  2,  1],
    "neutral": [ 0,  1,  0],
    "sad":     [-1, -2, -1],
    "angry":   [-1,  1, -2],
    "disgust": [-2,  1, -1],
    "fear":    [ 2, -3,  1],
}

def _va_to_scale(v, a):
    """Each CREMA-D emotion has a natural scale home."""
    if v > 0.10:   return "major"     # Happy
    if abs(v) <= 0.10: return "dorian"  # Neutral — modal, ambiguous
    # Negative valence:
    if a > 0.42:   return "phrygian"  # Angry / Fear — darkest, most unstable
    return "minor"                    # Sad / Disgust — dark but not frantic

def _emotion_region(scale, valence, arousal):
    """Delegate directly to emotion_label for a clean single source of truth."""
    return emotion_label(valence, arousal).lower()

def _va_to_root(v, a=0.5):
    """
    Happy/excited: high bright roots. Angry: low dark roots.
    """
    if v >= 0.15:
        # Happy/excited: bright upper register
        pool = [67, 69, 64, 71]        # G4, A4, E4, B4
    elif v <= -0.15 and a >= 0.5:
        # Angry: low dark roots — Bb2, Ab2, F2, Eb2
        pool = [46, 44, 41, 39]
    elif v <= -0.15:
        # Sad: mid-low dark roots
        pool = [58, 56, 61, 53]        # Bb3, Ab3, Db4, F3
    else:
        pool = [60, 65, 62]            # neutral: C4, F4, D4
    return pool[int(abs(v) * (len(pool) - 1))]

def _va_to_root_name(v) -> str:
    """Return the actual MIDI root note name for display."""
    midi = _va_to_root(v)  # root_name uses default a
    names = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
    return names[midi % 12]

def _va_to_tempo(a, v=0.0):
    """Happy/excited gets faster ceiling; angry gets a faster floor too (aggressive)."""
    if v > 0.15:   return int(np.clip(80  + a * 120, 80,  200))  # jolly: 80–200
    if v < -0.15:  return int(np.clip(50  + a * 120, 50,  170))  # angry: 50–170
    return             int(np.clip(55  + a * 110, 55,  165))      # neutral
def _va_to_velocity(a): return int(np.clip(48 + a * 65,  48, 113))

def _build_scale(root, ivls, octaves=4):
    return [root + i + o * 12 for o in range(octaves) for i in ivls]

def _build_chord(root, ivls, degree, inversion=0, quality="triad"):
    """
    Build a voiced chord — intentionally sparse for ambient feel.
    '7th' is silently demoted to 'triad' to avoid symphonic density.
    'open5' plays only root + fifth (very ambient, no thirds).
    """
    d = degree % 7
    if quality in ("sus2", "open5"):
        # Root + fifth only — open, non-committal, ambient
        degs = [d, (d + 4) % 7]
    elif quality == "sus4":
        degs = [d, (d + 3) % 7, (d + 4) % 7]
    else:
        # triad or 7th (7th demoted) — plain three-note triad
        degs = [d, (d + 2) % 7, (d + 4) % 7]

    pitches = [root + ivls[dg] for dg in degs]

    for _ in range(inversion % len(pitches)):
        pitches[0] += 12
        pitches.sort()

    return [p - 12 for p in pitches]

def _build_bass_note(root, ivls, degree):
    return root + ivls[degree % 7] - 24

def _make_phrase(scale_notes, contour, start_idx, beats_per_phrase,
                 arousal, valence, region, rng, phrase_num=0):

    # ── Rhythm palettes ───────────────────────────────────────────────────────
    if region == "happy":
        base_rhythms = [0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.5,
                        0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.25]
        rest_prob = 0.05;  artic = 0.55
    elif region == "angry":
        base_rhythms = [0.25, 0.5, 0.25, 1.0, 0.25, 0.25, 0.5, 1.5,
                        0.25, 0.25, 1.0, 0.25, 0.5, 0.25, 1.5, 0.25]
        rest_prob = 0.55;  artic = 0.25
    elif region == "fear":
        base_rhythms = [0.25, 1.5, 0.25, 0.5, 2.0, 0.25, 0.25, 1.0,
                        0.5, 0.25, 1.5, 0.25, 0.25, 2.0, 0.5, 0.25]
        rest_prob = 0.50;  artic = 0.35
    elif region == "disgust":
        base_rhythms = [0.5, 1.5, 0.5, 2.0, 0.5, 1.0, 2.0, 0.5,
                        1.0, 0.5, 2.0, 0.5, 1.5, 0.5, 1.0, 2.0]
        rest_prob = 0.48;  artic = 0.38
    elif region == "sad":
        base_rhythms = [2.0, 1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0,
                        2.0, 3.0, 2.0, 1.0, 2.0, 2.0, 3.0, 2.0]
        rest_prob = 0.40;  artic = 0.92
    else:  # neutral
        base_rhythms = [1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0,
                        1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0]
        rest_prob = 0.28;  artic = 0.80

    events   = []
    beat     = 0.0
    base_vel = _va_to_velocity(arousal)

    ph_offset = 0
    if phrase_num == 1: ph_offset = 2  if valence > 0 else -2
    if phrase_num == 2: ph_offset = -1 if valence > 0 else 1
    if phrase_num == 3: ph_offset = 3  if valence > 0 else -3

    # Happy: start high in the scale for brightness
    if valence > 0.1:
        start_idx = min(start_idx + 4, len(scale_notes) - 1)
    # Angry: go low AND add massive velocity spikes
    elif valence < -0.1:
        start_idx = max(start_idx - 5, 0)

    total_contour_steps = len(contour)

    for i, offset in enumerate(contour):
        if beat >= beats_per_phrase:
            break

        dur = min(base_rhythms[i % len(base_rhythms)], beats_per_phrase - beat)
        idx  = int(np.clip(start_idx + ph_offset + offset, 0, len(scale_notes) - 1))
        note = max(0, min(127, scale_notes[idx]))

        beat_pos = beat % 4
        vel_beat = 8 if beat_pos == 0 else (4 if beat_pos == 2 else 0)
        arc_pos  = i / max(total_contour_steps - 1, 1)
        vel_arc  = int(10 * np.sin(arc_pos * np.pi))

        # Angry: random spikes up to +35, sometimes hit max velocity
        angry_spike = rng.randint(10, 35) if region == "angry" else 0

        vel = int(np.clip(base_vel + vel_beat + vel_arc + angry_spike + rng.randint(-4, 4), 24, 127))

        if rng.random() > rest_prob:
            events.append((note, beat, max(0.05, dur * artic), vel))

        beat += dur

    return events


def generate_midi(valence: float, arousal: float,
                  num_bars: int = 16, seed: int = 42) -> bytes:
    """
    Generate a richer MIDI piece. Improvements over v1:
    - 16 bars (was 8) with intro / development / climax / outro arc
    - 5 scales including Lydian and Phrygian, chosen from both V and A
    - Seventh chords, sus2, sus4 alongside triads
    - Melody has 4 distinct phrases (statement, answer, return, climax)
    - Passing-tone ornaments at high arousal
    - Arpeggiated chords scaled to arousal (8th-note arps at high energy)
    - Walking/moving bass line with chromatic approach notes
    - Dynamics arc: intro → development → climax → outro
    - Per-phrase pitch transposition for musical variety
    """
    rng    = random.Random(seed)
    scale  = _va_to_scale(valence, arousal)
    root   = _va_to_root(valence, arousal)
    ivls   = SCALES[scale]
    tempo  = _va_to_tempo(arousal, valence)
    prog   = PROGRESSIONS[scale]
    region = _emotion_region(scale, valence, arousal)
    contour = CONTOURS[region]
    passing = PASSING_TONES[region]

    scale_notes = _build_scale(root, ivls, octaves=4)
    start_idx   = len(scale_notes) // 2   # start in middle register

    # ── Instrument selection ──────────────────────────────────────────────────
    # (melody, chord_pad, bass)  GM program numbers
    INSTR = {
        ("major",    True):  (0,   48, 33),   # piano / strings / finger bass
        ("major",    False): (11,  52, 32),   # vibraphone / slow strings / acoustic bass
        ("lydian",   True):  (10,  49, 33),   # music box / slow strings / finger bass
        ("lydian",   False): (9,   52, 32),   # celesta / slow strings / acoustic bass
        ("dorian",   True):  (73,  48, 33),   # flute / strings / finger bass
        ("dorian",   False): (73,  44, 34),   # flute / tremolo strings / fretless bass
        ("phrygian", True):  (68,  44, 42),   # oboe / tremolo / cello
        ("phrygian", False): (70,  44, 42),   # bassoon / tremolo / cello
        ("minor",    True):  (70,  44, 42),   # bassoon / tremolo / cello
        ("minor",    False): (19,  44, 42),   # church organ / tremolo / cello
    }
    mel_prog, pad_prog, bas_prog = INSTR.get((scale, arousal >= 0.5),
                                              (0, 48, 33))

    # ── Structure: 4 sections of 4 bars each ─────────────────────────────────
    #  intro (0-3): soft, establishes mood
    #  develop (4-7): main theme, fuller
    #  climax (8-11): peak energy/expression
    #  outro (12-15): resolve, fade back

    def section_of(bar):
        if bar < 4:            return "intro"
        elif bar < 8:          return "develop"
        elif bar < 12:         return "climax"
        else:                  return "outro"

    def section_vel_scale(bar):
        s = section_of(bar)
        return {"intro": 0.55, "develop": 0.85, "climax": 1.0, "outro": 0.65}[s]

    midi = MIDIFile(numTracks=4)
    for t in range(4):
        midi.addTempo(t, 0, tempo)
    midi.addProgramChange(0, 0, 0, mel_prog)
    midi.addProgramChange(1, 1, 0, pad_prog)
    midi.addProgramChange(2, 2, 0, bas_prog)

    beats_per_bar    = 4
    beats_per_phrase = beats_per_bar * 4   # one phrase = 4 bars

    # ── Track 0: Melody — 4 distinct 4-bar phrases ───────────────────────────
    for phrase_idx in range(num_bars // 4):
        phrase_start = phrase_idx * beats_per_phrase
        events = _make_phrase(scale_notes, contour, start_idx,
                              beats_per_phrase, arousal, valence, region,
                              rng, phrase_num=phrase_idx)

        first_bar_of_phrase = phrase_idx * 4
        for (note, beat_off, dur, vel) in events:
            bar     = first_bar_of_phrase + int(beat_off // beats_per_bar)
            bar     = min(bar, num_bars - 1)
            out_vel = int(vel * section_vel_scale(bar))
            midi.addNote(0, 0, note, phrase_start + beat_off, dur, out_vel)


    # Convenience flags for all 6 emotions
    em_label   = emotion_label(valence, arousal)
    is_happy   = em_label == "Happy"
    is_neutral = em_label == "Neutral"
    is_sad     = em_label == "Sad"
    is_angry   = em_label == "Angry"
    is_disgust = em_label == "Disgust"
    is_fear    = em_label == "Fear"
    is_negative_high = is_angry or is_fear   # both get aggressive treatment

    # ── Track 1: Chords ───────────────────────────────────────────────────────
    base_chord_vel = max(20, _va_to_velocity(arousal) - 28)

    for bar in range(num_bars):
        # Chord frequency: happy/angry/fear every bar; others every 2 bars
        if not (is_happy or is_negative_high):
            if bar % 2 == 1 or bar == 0:
                continue

        bt  = bar * beats_per_bar
        vs  = section_vel_scale(bar)
        degree, inv, quality = prog[bar % len(prog)]

        if is_happy:    quality = "triad"
        elif is_angry:  quality = "sus4"
        elif is_fear:   quality = "open5"   # open5 + dissonance added below
        elif is_disgust:quality = "sus2"    # sus2 sounds unsettled without being angry

        chord = _build_chord(root, ivls, degree, inv, quality)
        cv    = max(1, int(base_chord_vel * vs))

        if is_happy:
            step = 0.12
            for i, n in enumerate(chord):
                note_dur = beats_per_bar * 0.88 - i * step
                if note_dur > 0.05:
                    midi.addNote(1, 1, max(0,min(127,n)), bt + i*step, note_dur,
                                 max(1, int(cv * (1.0 - i*0.1))))
        elif is_angry:
            for stab_beat in [0.0, 1.5, 3.0]:
                for n in chord:
                    midi.addNote(1, 1, max(0,min(127,n)), bt+stab_beat, 0.22, min(127,int(cv*1.4)))
                cluster_note = max(0, min(127, chord[0]+1))
                midi.addNote(1, 1, cluster_note, bt+stab_beat, 0.22, min(127,int(cv*1.0)))
        elif is_fear:
            # Sudden single-beat stab at random position — unsettling
            stab_pos = rng.choice([0.0, 1.0, 2.0, 3.0])
            for n in chord:
                midi.addNote(1, 1, max(0,min(127,n)), bt+stab_pos, 0.30, min(127,int(cv*1.1)))
            # Tritone above root — maximum dissonance
            tritone = max(0, min(127, chord[0]+6))
            midi.addNote(1, 1, tritone, bt+stab_pos, 0.30, min(127,int(cv*0.85)))
        elif is_disgust:
            # Slow heavy hit on beat 1 only — one lurch per bar
            for n in chord:
                midi.addNote(1, 1, max(0,min(127,n)), bt, 1.20, max(1,int(cv*0.90)))
        else:
            # Sad/Neutral: sustained 2-bar chords
            sus_dur = beats_per_bar * 2 - 0.4
            for n in chord:
                midi.addNote(1, 1, n, bt, sus_dur, cv)

    # ── Track 2: Bass ────────────────────────────────────────────────────────
    base_bass_vel = max(28, _va_to_velocity(arousal) - 18)

    for bar in range(num_bars):
        if not (is_happy or is_negative_high or is_disgust):
            if bar % 2 == 1 or bar == 0:
                continue

        bt     = bar * beats_per_bar
        vs     = section_vel_scale(bar)
        degree, _, _ = prog[bar % len(prog)]
        bass_root  = _build_bass_note(root, ivls, degree)
        bass_fifth = bass_root + 7
        bv         = max(1, int(base_bass_vel * vs))

        if is_happy:
            midi.addNote(2, 2, max(0,min(127,bass_root)),      bt+0.0, 0.45, bv)
            midi.addNote(2, 2, max(0,min(127,bass_fifth)),     bt+1.0, 0.45, max(1,int(bv*0.80)))
            midi.addNote(2, 2, max(0,min(127,bass_root)),      bt+2.0, 0.45, max(1,int(bv*0.90)))
            midi.addNote(2, 2, max(0,min(127,bass_root+12)),   bt+3.0, 0.45, max(1,int(bv*0.70)))
        elif is_angry:
            low_root = max(0, min(127, bass_root-24))
            for pos in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
                hit_vel = min(120, int(bv*(1.5 if pos in (0.0,2.0) else 1.0)))
                midi.addNote(2, 2, low_root, bt+pos, 0.22, max(1, hit_vel))
        elif is_fear:
            # Sudden low hits at unpredictable offsets
            low_root = max(0, min(127, bass_root-12))
            for pos in sorted(rng.sample([0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5], 3)):
                midi.addNote(2, 2, low_root, bt+pos, 0.25, max(1,int(bv*1.1)))
        elif is_disgust:
            # Heavy single stomp on beat 1 per bar
            low_root = max(0, min(127, bass_root-12))
            midi.addNote(2, 2, low_root, bt, 0.80, min(120,int(bv*1.2)))
        else:
            midi.addNote(2, 2, bass_root, bt, beats_per_bar*2-0.5, bv)
            if arousal >= 0.45:
                midi.addNote(2, 2, bass_fifth, bt+2.0, beats_per_bar-0.5, max(1,int(bv*0.65)))

    # ── Track 3: Percussion ───────────────────────────────────────────────────
    KICK=35; SNARE=38; HIHAT=42; OPEN_HH=46; CRASH=49; RIDE=51; MARACAS=70

    def drum_vel_scale(bar):
        s = section_of(bar)
        return {"intro":0.0,"develop":0.70,"climax":1.0,"outro":0.45}[s]

    base_drum_vel = max(48, int(_va_to_velocity(arousal)*0.95))

    def dv(factor, vs):
        return max(1, min(127, int(base_drum_vel*vs*factor)))

    def add_drum(pitch, bt, dur, vel):
        if vel > 0 and dur > 0.0:
            midi.addNote(3, 9, pitch, bt, dur, vel)

    if is_happy:
        def drum_bar(bar, bt, vs):
            add_drum(KICK,  bt+0.0, 0.30, dv(0.70, vs))
            add_drum(KICK,  bt+2.0, 0.30, dv(0.55, vs))
            for eighth in range(8):
                add_drum(HIHAT, bt+eighth*0.5, 0.18, dv(0.35 if eighth%2 else 0.50, vs))
            add_drum(75, bt+1.0, 0.18, dv(0.45, vs))
            add_drum(75, bt+3.0, 0.18, dv(0.40, vs))

    elif is_angry:
        def drum_bar(bar, bt, vs):
            for pos in [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]:
                add_drum(KICK,  bt+pos, 0.28, dv(1.30 if pos in (0.0,2.0) else 1.00, vs))
            add_drum(SNARE, bt+1.0,  0.25, dv(1.20, vs))
            add_drum(SNARE, bt+1.05, 0.25, dv(0.70, vs))
            add_drum(SNARE, bt+3.0,  0.25, dv(1.10, vs))
            add_drum(SNARE, bt+3.05, 0.25, dv(0.65, vs))
            add_drum(CRASH, bt, 0.80, dv(1.10, vs))
            if rng.random() < 0.70:
                add_drum(SNARE, bt+rng.choice([0.5,1.5,2.5,3.5]), 0.20, dv(0.90, vs))

    elif is_fear:
        def drum_bar(bar, bt, vs):
            # Irregular hits — no predictable pattern, very unsettling
            add_drum(KICK,  bt+0.0, 0.30, dv(1.00, vs))
            add_drum(CRASH, bt+0.0, 0.70, dv(0.90, vs))
            # Random snare hits at irregular positions
            for pos in sorted(rng.sample([0.5,1.0,1.5,2.0,2.5,3.0,3.5], rng.randint(2,4))):
                add_drum(SNARE, bt+pos, 0.22, dv(rng.uniform(0.60,1.10), vs))
            # Single open hi-hat at a random position
            add_drum(OPEN_HH, bt+rng.choice([1.0,2.0,3.0]), 0.40, dv(0.55, vs))

    elif is_disgust:
        def drum_bar(bar, bt, vs):
            # Heavy kick every beat — stomping, unpleasant regularity
            for beat_i in range(4):
                add_drum(KICK, bt+beat_i, 0.40, dv(1.05, vs))
            # Slow open hi-hat on beat 3 — grating
            add_drum(OPEN_HH, bt+2.0, 0.70, dv(0.60, vs))

    elif is_sad:
        def drum_bar(bar, bt, vs):
            add_drum(KICK,   bt+0.0, 0.55, dv(0.65, vs))
            add_drum(RIDE,   bt+2.0, 0.45, dv(0.45, vs))
            for beat_i in range(4):
                add_drum(MARACAS, bt+beat_i, 0.16, dv(0.28, vs))

    else:  # neutral
        def drum_bar(bar, bt, vs):
            add_drum(KICK,  bt+0.0, 0.45, dv(1.00, vs))
            add_drum(SNARE, bt+2.0, 0.45, dv(0.85, vs))
            add_drum(HIHAT, bt+1.0, 0.22, dv(0.50, vs))
            add_drum(HIHAT, bt+3.0, 0.22, dv(0.43, vs))

    for bar in range(num_bars):
        bt = bar * beats_per_bar
        vs = drum_vel_scale(bar)
        if vs > 0.0:
            drum_bar(bar, bt, vs)

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
    Map (valence, arousal) → one of 6 CREMA-D categories.

    Calibrated from real pipeline output across 10 actors. The model's VA
    output cannot cleanly separate all 6 emotions — the means overlap heavily:

      ANG: a_mean=0.758  ← clearly highest, reliably separable
      HAP: a_mean=0.712  ← second highest
      DIS: a_mean=0.617  ┐
      FEA: a_mean=0.623  ├ nearly indistinguishable from each other
      NEU: a_mean=0.568  ┘
      SAD: a_mean=0.522  ← clearly lowest, reliably separable

    Strategy: use arousal as primary signal (it's the most reliable), and
    valence as a secondary tiebreaker within overlapping zones.
    Accept that DIS/FEA/NEU will sometimes be confused with each other.
    """
    if arousal > 0.73:
        return "Angry"
    if arousal < 0.55:
        return "Sad"
    if arousal > 0.66:
        # High-mid zone: Happy (0.712) vs Fear (0.623 upper tail)
        # Happy tends to have higher valence in this zone
        return "Happy" if valence > 0.20 else "Fear"
    if arousal > 0.59:
        # Mid zone: Fear (0.623) vs Disgust (0.617) — nearly identical
        # Use valence as tiebreaker: Fear tends higher valence
        return "Fear" if valence > 0.20 else "Disgust"
    # Low-mid zone: Neutral (0.568) territory
    return "Neutral"



def _predict_from_features(feats: dict):
    """
    Run the model on a feature dict and return (valence, arousal, label).
    Uses the emotion head directly when the new model is loaded (post-retrain),
    falls back to VA threshold classifier for the old model.
    """
    row_data   = np.array([[feats.get(k, 0.0) for k in _feature_cols]], dtype=np.float32)
    row_scaled = _scaler.transform(row_data)
    t          = torch.tensor(row_scaled).to(_torch_device)

    with torch.no_grad():
        out = _va_model(t)

    # New model returns (va_tensor, emotion_logits); old returns va_tensor only
    if isinstance(out, tuple):
        va_tensor, em_logits = out
        pred   = va_tensor.cpu().numpy()[0]
        em_idx = int(em_logits.argmax(1).cpu().item())
        label  = _EMOTION_NAMES.get(_EMOTION_CLASSES[em_idx], "Neutral")
    else:
        pred  = out.cpu().numpy()[0]
        label = None

    raw_v = float(np.clip(pred[0], -1.0, 1.0))
    raw_a = float(np.clip(pred[1],  0.0, 1.0))
    valence, arousal = stretch_va(raw_v, raw_a, feats.get("text_lex_valence", 0.0), feats)

    if label is None:
        label = emotion_label(valence, arousal)

    return valence, arousal, label


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
    orig_ext   = os.path.splitext(video_file.filename or "clip.mp4")[1] or ".mp4"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, f"clip{orig_ext}")
            audio_path = os.path.join(tmp, "clip.wav")
            video_file.save(video_path)
            feats    = extract_features(video_path, audio_path)
            duration = feats.pop("_audio_duration", 5.0)
            feats.update(transcribe_and_extract(audio_path, duration))

            # Run model — use emotion head directly for label
            row        = np.array([[feats.get(k, 0.0) for k in _feature_cols]], dtype=np.float32)
            row_scaled = _scaler.transform(row)
            t          = torch.tensor(row_scaled).to(_torch_device)

            with torch.no_grad():
                out = _va_model(t)

            if isinstance(out, tuple):
                va_tensor, em_logits = out
                pred    = va_tensor.cpu().numpy()[0]
                em_idx  = int(em_logits.argmax(1).cpu().item())
                # Softmax probabilities for all 6 classes
                probs   = torch.softmax(em_logits, dim=1).cpu().numpy()[0]
                em_label = _EMOTION_NAMES.get(_EMOTION_CLASSES[em_idx], "Neutral")
            else:
                pred     = out.cpu().numpy()[0]
                em_label = None
                probs    = None

            v_raw   = float(pred[0])
            a_raw   = float(torch.sigmoid(torch.tensor(pred[1])).item())
            a_logit = float(pred[1])
            sv, sa  = stretch_va(
                float(np.clip(v_raw, -1.0, 1.0)),
                float(np.clip(a_raw,  0.0, 1.0)),
                feats.get("text_lex_valence", 0.0), feats)

            if em_label is None:
                em_label = emotion_label(sv, sa)

            rb_v, rb_a, rb_label = _rule_based_va(feats)

    except Exception as e:
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500

    resp = {
        "emotion_label":      em_label,
        "raw_valence":        round(v_raw,   4),
        "raw_arousal":        round(a_raw,   4),
        "arousal_logit":      round(a_logit, 4),
        "stretched_valence":  round(sv, 4),
        "stretched_arousal":  round(sa, 4),
        "rule_valence":       round(rb_v, 4),
        "rule_arousal":       round(rb_a, 4),
        "rule_label":         rb_label,
        "energy_mean":        round(feats.get("energy_mean",      0), 6),
        "energy_std":         round(feats.get("energy_std",       0), 6),
        "pitch_mean":         round(feats.get("pitch_mean",       0), 2),
        "pitch_std":          round(feats.get("pitch_std",        0), 2),
        "zcr":                round(feats.get("zcr",              0), 5),
        "hnr":                round(feats.get("hnr",              0), 3),
        "jitter":             round(feats.get("jitter",           0), 5),
        "shimmer":            round(feats.get("shimmer",          0), 5),
        "cpp":                round(feats.get("cpp",              0), 4),
        "energy_slope":       round(feats.get("energy_slope",     0), 6),
        "pitch_slope":        round(feats.get("pitch_slope",      0), 3),
        "pause_ratio":        round(feats.get("pause_ratio",      0), 3),
        "voiced_trans_rate":  round(feats.get("voiced_trans_rate",0), 5),
        "spectral_entropy":   round(feats.get("spectral_entropy", 0), 4),
        "lowfreq_energy_ratio": round(feats.get("lowfreq_energy_ratio", 0), 4),
        "face_detected":      feats.get("face_detected", 0),
        "face_smile_mean":    round(feats.get("face_smile_mean",  0), 4),
        "face_furrow_mean":   round(feats.get("face_furrow_mean", 0), 4),
        "text_lex_valence":   round(feats.get("text_lex_valence", 0), 3),
        "feature_count":      len(_feature_cols),
        "nonzero_features":   int(np.count_nonzero(row[0])),
        "audio_duration_s":   round(duration, 2),
    }
    if probs is not None:
        for i, cls in enumerate(_EMOTION_CLASSES):
            resp[f"prob_{cls}"] = round(float(probs[i]), 4)
    return jsonify(resp)


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
        orig_ext = os.path.splitext(video_file.filename or "clip.mp4")[1] or ".mp4"
        video_path = os.path.join(tmp, f"clip{orig_ext}")
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

        valence, arousal, label = _predict_from_features(feats)
        scale = _va_to_scale(valence, arousal)
        tempo = _va_to_tempo(arousal, valence)

    return jsonify({
        "valence":       round(valence, 4),
        "arousal":       round(arousal, 4),
        "scale":         scale,
        "root_name":     _va_to_root_name(valence),
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
        orig_ext = os.path.splitext(video_file.filename or "clip.mp4")[1] or ".mp4"
        video_path = os.path.join(tmp, f"clip{orig_ext}")
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

        valence, arousal, label = _predict_from_features(feats)
    scale = _va_to_scale(valence, arousal)
    tempo = _va_to_tempo(arousal, valence)

    import time as _time
    midi_seed  = int(_time.time() * 1000) % (2**31)
    midi_bytes = generate_midi(valence, arousal, num_bars=16, seed=midi_seed)
    import base64
    midi_b64 = base64.b64encode(midi_bytes).decode()

    return jsonify({
        "valence":       round(valence, 4),
        "arousal":       round(arousal, 4),
        "scale":         scale,
        "root_name":     _va_to_root_name(valence),
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
