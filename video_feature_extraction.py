import os
import urllib.request
import cv2
import librosa
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
mp4_folder   = "/Users/janemaguire/Desktop/Final/MP4"
audio_folder = "/Users/janemaguire/Desktop/Final/CREMA-D/AudioWAV"
output_csv   = "/Users/janemaguire/Desktop/Final/CREMA-D/video_features.csv"

MODEL_PATH = "/Users/janemaguire/Desktop/Final/CREMA-D/face_landmarker.task"

os.makedirs(audio_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Download MediaPipe face landmarker model if needed
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print("Downloading face_landmarker.task model (~5 MB)...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete.")

# ---------------------------------------------------------------------------
# Build FaceLandmarker
# ---------------------------------------------------------------------------
options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
face_landmarker = FaceLandmarker.create_from_options(options)

# Load ground truth mapping
va_mapping = pd.read_csv(
    "/Users/janemaguire/Desktop/Final/CREMA-D/video_va_mapping_dynamic.csv"
)

# ---------------------------------------------------------------------------
# Landmark index constants  (MediaPipe 478-point model)
# ---------------------------------------------------------------------------
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291

LEFT_EYE_TOP     = 159
LEFT_EYE_BOTTOM  = 145
LEFT_EYE_LEFT    = 33
LEFT_EYE_RIGHT   = 133

RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT   = 362
RIGHT_EYE_RIGHT  = 263

LEFT_BROW_INNER  = 107
RIGHT_BROW_INNER = 336

# ---------------------------------------------------------------------------
# Additional landmark indices for valence-sensitive facial features
# Smile / lip corner geometry correlates strongly with positive valence
# ---------------------------------------------------------------------------
MOUTH_CORNER_LEFT  = 61    # left lip corner
MOUTH_CORNER_RIGHT = 291   # right lip corner
UPPER_LIP_CENTER   = 13    # top of upper lip
LOWER_LIP_CENTER   = 14    # bottom of lower lip
LEFT_CHEEK         = 234   # left cheek landmark
RIGHT_CHEEK        = 454   # right cheek landmark
NOSE_TIP           = 4     # nose tip (reference for smile height)


def euclidean(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def compute_landmark_features(lm):
    """
    Compute expression-relevant scalar features from a list of landmarks.
    Returns a dict of scalar values.

    New valence-sensitive features added:
      smile_ratio    — lip corner height relative to mouth width; rises for smiles
      cheek_raise    — cheek elevation relative to nose; rises with genuine smiles
                       (Duchenne marker — hard to fake)
      brow_furrow    — inner brow convergence; rises with anger/fear/sadness
    """
    # ── Original features ────────────────────────────────────────────────────
    mouth_height = euclidean(lm[MOUTH_TOP],   lm[MOUTH_BOTTOM])
    mouth_width  = euclidean(lm[MOUTH_LEFT],  lm[MOUTH_RIGHT])
    mar = mouth_height / (mouth_width + 1e-6)

    left_ear  = euclidean(lm[LEFT_EYE_TOP],  lm[LEFT_EYE_BOTTOM])  / (euclidean(lm[LEFT_EYE_LEFT],  lm[LEFT_EYE_RIGHT])  + 1e-6)
    right_ear = euclidean(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOTTOM]) / (euclidean(lm[RIGHT_EYE_LEFT], lm[RIGHT_EYE_RIGHT]) + 1e-6)
    ear = (left_ear + right_ear) / 2.0

    left_brow_raise  = euclidean(lm[LEFT_BROW_INNER],  lm[LEFT_EYE_LEFT])
    right_brow_raise = euclidean(lm[RIGHT_BROW_INNER], lm[RIGHT_EYE_RIGHT])
    brow_raise = (left_brow_raise + right_brow_raise) / 2.0

    # ── NEW: smile ratio ─────────────────────────────────────────────────────
    # Lip corners rise vertically relative to the center of the mouth in smiles.
    # Positive y-direction in MediaPipe is downward, so a smile pulls corners UP
    # (lower y value). We measure how high corners are relative to lip center.
    lip_center_y = (lm[UPPER_LIP_CENTER].y + lm[LOWER_LIP_CENTER].y) / 2.0
    corner_y_avg = (lm[MOUTH_CORNER_LEFT].y + lm[MOUTH_CORNER_RIGHT].y) / 2.0
    # Positive value → corners above center → smile
    smile_ratio = (lip_center_y - corner_y_avg) / (mouth_width + 1e-6)

    # ── NEW: cheek raise (Duchenne smile marker) ─────────────────────────────
    # In genuine happy expressions, cheeks lift. We measure cheek y relative
    # to nose tip — higher cheeks = smaller y value.
    nose_y       = lm[NOSE_TIP].y
    cheek_y_avg  = (lm[LEFT_CHEEK].y + lm[RIGHT_CHEEK].y) / 2.0
    cheek_raise  = nose_y - cheek_y_avg   # positive → cheeks above nose level

    # ── NEW: brow furrow ─────────────────────────────────────────────────────
    # Inner brows converge and drop in anger, fear, sadness.
    # We measure horizontal distance between inner brow points.
    # Smaller distance → more furrowed → negative valence.
    brow_furrow = euclidean(lm[LEFT_BROW_INNER], lm[RIGHT_BROW_INNER])

    return {
        "mar":        mar,
        "ear":        ear,
        "brow_raise": brow_raise,
        "smile_ratio":smile_ratio,
        "cheek_raise":cheek_raise,
        "brow_furrow":brow_furrow,
    }


# ---------------------------------------------------------------------------
# Main feature extraction loop
# ---------------------------------------------------------------------------
features_list = []

for idx, row in va_mapping.iterrows():
    mp4_file   = os.path.join(mp4_folder, row["fileName"])
    fname_base = row["fileName"].replace(".mp4", "")

    print(f"[{idx + 1}/{len(va_mapping)}] Processing {row['fileName']} ...")

    # ------------------------------------------------------------------
    # AUDIO FEATURES
    # ------------------------------------------------------------------
    wav_file = os.path.join(audio_folder, f"{fname_base}.wav")
    if not os.path.exists(wav_file):
        os.system(f'ffmpeg -i "{mp4_file}" -q:a 0 -map a "{wav_file}" -y -loglevel quiet')

    audio_y, sr = librosa.load(wav_file, sr=None)

    # ── Existing features ─────────────────────────────────────────────────────

    # MFCCs (52 values: mean + std + delta + delta2, each 13-dim)
    mfccs       = librosa.feature.mfcc(y=audio_y, sr=sr, n_mfcc=13)
    mfccs_mean  = np.mean(mfccs,  axis=1)
    mfccs_std   = np.std(mfccs,   axis=1)
    delta       = librosa.feature.delta(mfccs)
    delta2      = librosa.feature.delta(mfccs, order=2)
    delta_mean  = np.mean(delta,  axis=1)
    delta2_mean = np.mean(delta2, axis=1)

    # Pitch
    pitch        = librosa.yin(audio_y, fmin=50, fmax=500)
    pitch_voiced = pitch[pitch > 0]
    pitch_mean   = float(np.mean(pitch_voiced))  if len(pitch_voiced) > 0 else 0.0
    pitch_std    = float(np.std(pitch_voiced))   if len(pitch_voiced) > 0 else 0.0
    pitch_range  = float(np.max(pitch_voiced) - np.min(pitch_voiced)) if len(pitch_voiced) > 0 else 0.0
    voiced_ratio = len(pitch_voiced) / (len(pitch) + 1e-6)

    # Energy
    rms         = librosa.feature.rms(y=audio_y)
    energy_mean = float(np.mean(rms))
    energy_std  = float(np.std(rms))

    # Spectral shape
    spec_centroid  = float(np.mean(librosa.feature.spectral_centroid(y=audio_y,  sr=sr)))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio_y, sr=sr)))
    spec_rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=audio_y,   sr=sr)))
    zcr            = float(np.mean(librosa.feature.zero_crossing_rate(audio_y)))

    # ── NEW valence-specific audio features ───────────────────────────────────

    # Harmonics-to-Noise Ratio (HNR)
    # Happy speech is more periodic/tonal; sad/angry is breathier or harsher.
    # One of the strongest known acoustic predictors of positive valence.
    # librosa.effects.harmonic separates the tonal component; we compare
    # its energy to the noise (percussive) component.
    harmonic    = librosa.effects.harmonic(audio_y)
    percussive  = librosa.effects.percussive(audio_y)
    harmonic_energy   = float(np.mean(harmonic   ** 2) + 1e-10)
    percussive_energy = float(np.mean(percussive ** 2) + 1e-10)
    hnr = float(10 * np.log10(harmonic_energy / percussive_energy))  # dB

    # Spectral flatness
    # Measures how "tonal" vs "noise-like" the spectrum is.
    # 0 = perfectly tonal (sine wave), 1 = white noise.
    # Disgust and anger push toward flatness; happiness toward tonality (low flatness).
    spec_flatness      = librosa.feature.spectral_flatness(y=audio_y)
    spec_flatness_mean = float(np.mean(spec_flatness))
    spec_flatness_std  = float(np.std(spec_flatness))

    # Chroma features (12-dim: one per pitch class C, C#, D, ... B)
    # Capture the melodic/harmonic quality of the voice.
    # Happy speech tends toward higher chroma variation and brighter pitch classes.
    chroma      = librosa.feature.chroma_stft(y=audio_y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)   # 12 values
    chroma_std  = float(np.std(chroma_mean))
    chroma_max  = float(np.max(chroma_mean))
    chroma_entropy = float(
        -np.sum(chroma_mean / (chroma_mean.sum() + 1e-10) *
                np.log(chroma_mean / (chroma_mean.sum() + 1e-10) + 1e-10))
    )  # low entropy → concentrated on few pitch classes → more tonal

    # Mel-spectrogram summary statistics
    # Richer frequency representation than MFCCs; captures formant structure.
    mel_spec     = librosa.feature.melspectrogram(y=audio_y, sr=sr, n_mels=40)
    mel_db       = librosa.power_to_db(mel_spec, ref=np.max)
    mel_mean     = float(np.mean(mel_db))
    mel_std      = float(np.std(mel_db))
    mel_skew     = float(
        np.mean(((mel_db - mel_mean) / (mel_std + 1e-10)) ** 3)
    )  # skewness: asymmetry in energy distribution across mel bands

    # Pitch slope (linear trend of F0 over time)
    # Happy speech tends to end on rising pitch; sad on falling pitch.
    if len(pitch_voiced) > 1:
        t        = np.linspace(0, 1, len(pitch_voiced))
        slope    = float(np.polyfit(t, pitch_voiced, 1)[0])
    else:
        slope = 0.0
    pitch_slope = slope

    # Jitter (cycle-to-cycle pitch variation)
    # Emotional distress → higher jitter; calm/happy → lower jitter.
    if len(pitch_voiced) > 2:
        jitter = float(np.mean(np.abs(np.diff(pitch_voiced))) / (pitch_mean + 1e-6))
    else:
        jitter = 0.0

    # ------------------------------------------------------------------
    # VISUAL FEATURES  (MediaPipe Tasks FaceLandmarker)
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(mp4_file)
    frame_features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = face_landmarker.detect(mp_image)

        if result.face_landmarks:
            lm_list = result.face_landmarks[0]
            frame_features.append(compute_landmark_features(lm_list))

    cap.release()

    if frame_features:
        ff_df          = pd.DataFrame(frame_features)
        face_detected  = 1.0
        face_mar_mean  = float(ff_df["mar"].mean())
        face_mar_std   = float(ff_df["mar"].std())
        face_ear_mean  = float(ff_df["ear"].mean())
        face_ear_std   = float(ff_df["ear"].std())
        face_brow_mean = float(ff_df["brow_raise"].mean())
        face_brow_std  = float(ff_df["brow_raise"].std())
        # New visual features
        face_smile_mean  = float(ff_df["smile_ratio"].mean())
        face_smile_std   = float(ff_df["smile_ratio"].std())
        face_cheek_mean  = float(ff_df["cheek_raise"].mean())
        face_cheek_std   = float(ff_df["cheek_raise"].std())
        face_furrow_mean = float(ff_df["brow_furrow"].mean())
        face_furrow_std  = float(ff_df["brow_furrow"].std())
    else:
        face_detected  = 0.0
        face_mar_mean  = face_mar_std  = 0.0
        face_ear_mean  = face_ear_std  = 0.0
        face_brow_mean = face_brow_std = 0.0
        face_smile_mean= face_smile_std  = 0.0
        face_cheek_mean= face_cheek_std  = 0.0
        face_furrow_mean=face_furrow_std = 0.0

    # ------------------------------------------------------------------
    # Assemble feature dict
    # ------------------------------------------------------------------
    feat = {
        "fileName": row["fileName"],
        "valence":  row["valence"],
        "arousal":  row["arousal"],

        # MFCC means (13)
        **{f"mfcc_mean_{i + 1}":   float(mfccs_mean[i])  for i in range(13)},
        # MFCC stds (13)
        **{f"mfcc_std_{i + 1}":    float(mfccs_std[i])   for i in range(13)},
        # Delta MFCC means (13)
        **{f"mfcc_delta_{i + 1}":  float(delta_mean[i])  for i in range(13)},
        # Delta-delta MFCC means (13)
        **{f"mfcc_delta2_{i + 1}": float(delta2_mean[i]) for i in range(13)},

        # Prosodic features
        "pitch_mean":   pitch_mean,
        "pitch_std":    pitch_std,
        "pitch_range":  pitch_range,
        "voiced_ratio": voiced_ratio,
        "energy_mean":  energy_mean,
        "energy_std":   energy_std,

        # Spectral shape (original)
        "spec_centroid":  spec_centroid,
        "spec_bandwidth": spec_bandwidth,
        "spec_rolloff":   spec_rolloff,
        "zcr":            zcr,

        # NEW: valence-specific audio features
        "hnr":               hnr,
        "spec_flatness_mean":spec_flatness_mean,
        "spec_flatness_std": spec_flatness_std,
        "chroma_std":        chroma_std,
        "chroma_max":        chroma_max,
        "chroma_entropy":    chroma_entropy,
        **{f"chroma_mean_{i + 1}": float(chroma_mean[i]) for i in range(12)},
        "mel_mean":          mel_mean,
        "mel_std":           mel_std,
        "mel_skew":          mel_skew,
        "pitch_slope":       pitch_slope,
        "jitter":            jitter,

        # Facial landmark features (original)
        "face_detected":  face_detected,
        "face_mar_mean":  face_mar_mean,
        "face_mar_std":   face_mar_std,
        "face_ear_mean":  face_ear_mean,
        "face_ear_std":   face_ear_std,
        "face_brow_mean": face_brow_mean,
        "face_brow_std":  face_brow_std,

        # NEW: valence-specific facial features
        "face_smile_mean":  face_smile_mean,
        "face_smile_std":   face_smile_std,
        "face_cheek_mean":  face_cheek_mean,
        "face_cheek_std":   face_cheek_std,
        "face_furrow_mean": face_furrow_mean,
        "face_furrow_std":  face_furrow_std,
    }

    features_list.append(feat)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
features_df = pd.DataFrame(features_list)
features_df.to_csv(output_csv, index=False)
print(f"\nAll features saved to {output_csv}")
print(f"Feature matrix shape: {features_df.shape}")

# Quick correlation report — shows which new features are most predictive
print("\n── New feature correlations with valence / arousal ─────────────")
new_cols = ["hnr", "spec_flatness_mean", "chroma_std", "chroma_entropy",
            "mel_skew", "pitch_slope", "jitter",
            "face_smile_mean", "face_cheek_mean", "face_furrow_mean"]
for col in new_cols:
    rv = features_df[col].corr(features_df["valence"])
    ra = features_df[col].corr(features_df["arousal"])
    print(f"  {col:<22s}  valence r={rv:+.3f}   arousal r={ra:+.3f}")
