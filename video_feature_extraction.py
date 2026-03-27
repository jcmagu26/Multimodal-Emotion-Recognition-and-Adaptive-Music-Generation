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

# Path where the face landmarker model file will be stored
MODEL_PATH = "/Users/janemaguire/Desktop/Final/CREMA-D/face_landmarker.task"

os.makedirs(audio_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Download the MediaPipe face landmarker model if not already present.
# This is a one-time ~5 MB download from Google's servers.
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
# Build FaceLandmarker using the new MediaPipe Tasks API
# (replaces the old mp.solutions.face_mesh interface)
# ---------------------------------------------------------------------------
options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,         # process one frame at a time
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


def euclidean(p1, p2):
    """2D Euclidean distance between two landmark objects (each has .x and .y)."""
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def compute_landmark_features(lm):
    """
    Compute expression-relevant scalar features from a list of landmarks.
    Each landmark has normalised .x / .y / .z coordinates in [0, 1].
    Returns a dict with three values: mar, ear, brow_raise.
    """
    # Mouth Aspect Ratio: vertical opening / horizontal width
    mouth_height = euclidean(lm[MOUTH_TOP],   lm[MOUTH_BOTTOM])
    mouth_width  = euclidean(lm[MOUTH_LEFT],  lm[MOUTH_RIGHT])
    mar = mouth_height / (mouth_width + 1e-6)

    # Eye Aspect Ratio (average of left and right eyes)
    left_ear  = euclidean(lm[LEFT_EYE_TOP],  lm[LEFT_EYE_BOTTOM])  / (euclidean(lm[LEFT_EYE_LEFT],  lm[LEFT_EYE_RIGHT])  + 1e-6)
    right_ear = euclidean(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOTTOM]) / (euclidean(lm[RIGHT_EYE_LEFT], lm[RIGHT_EYE_RIGHT]) + 1e-6)
    ear = (left_ear + right_ear) / 2.0

    # Eyebrow raise: distance from inner brow point to nearest eye corner
    left_brow_raise  = euclidean(lm[LEFT_BROW_INNER],  lm[LEFT_EYE_LEFT])
    right_brow_raise = euclidean(lm[RIGHT_BROW_INNER], lm[RIGHT_EYE_RIGHT])
    brow_raise = (left_brow_raise + right_brow_raise) / 2.0

    return {"mar": mar, "ear": ear, "brow_raise": brow_raise}


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

    # MFCCs: mean + std + delta mean + delta-delta mean  (13 x 4 = 52 values)
    mfccs       = librosa.feature.mfcc(y=audio_y, sr=sr, n_mfcc=13)
    mfccs_mean  = np.mean(mfccs,  axis=1)
    mfccs_std   = np.std(mfccs,   axis=1)
    delta       = librosa.feature.delta(mfccs)
    delta2      = librosa.feature.delta(mfccs, order=2)
    delta_mean  = np.mean(delta,  axis=1)
    delta2_mean = np.mean(delta2, axis=1)

    # Pitch statistics (only voiced frames)
    pitch        = librosa.yin(audio_y, fmin=50, fmax=500)
    pitch_voiced = pitch[pitch > 0]
    pitch_mean   = float(np.mean(pitch_voiced))  if len(pitch_voiced) > 0 else 0.0
    pitch_std    = float(np.std(pitch_voiced))   if len(pitch_voiced) > 0 else 0.0
    pitch_range  = float(np.max(pitch_voiced) - np.min(pitch_voiced)) if len(pitch_voiced) > 0 else 0.0
    voiced_ratio = len(pitch_voiced) / (len(pitch) + 1e-6)

    # Energy / loudness
    rms         = librosa.feature.rms(y=audio_y)
    energy_mean = float(np.mean(rms))
    energy_std  = float(np.std(rms))

    # Spectral shape features
    spec_centroid  = float(np.mean(librosa.feature.spectral_centroid(y=audio_y,  sr=sr)))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio_y, sr=sr)))
    spec_rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=audio_y,   sr=sr)))
    zcr            = float(np.mean(librosa.feature.zero_crossing_rate(audio_y)))

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

        # result.face_landmarks is a list-of-lists (one entry per detected face)
        if result.face_landmarks:
            lm_list = result.face_landmarks[0]      # landmarks for the first face
            frame_features.append(compute_landmark_features(lm_list))

    cap.release()

    # Aggregate per-frame landmark features across the whole clip
    if frame_features:
        ff_df          = pd.DataFrame(frame_features)
        face_detected  = 1.0
        face_mar_mean  = float(ff_df["mar"].mean())
        face_mar_std   = float(ff_df["mar"].std())
        face_ear_mean  = float(ff_df["ear"].mean())
        face_ear_std   = float(ff_df["ear"].std())
        face_brow_mean = float(ff_df["brow_raise"].mean())
        face_brow_std  = float(ff_df["brow_raise"].std())
    else:
        face_detected  = 0.0
        face_mar_mean  = face_mar_std  = 0.0
        face_ear_mean  = face_ear_std  = 0.0
        face_brow_mean = face_brow_std = 0.0

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

        # Spectral features
        "spec_centroid":  spec_centroid,
        "spec_bandwidth": spec_bandwidth,
        "spec_rolloff":   spec_rolloff,
        "zcr":            zcr,

        # Facial landmark features
        "face_detected":  face_detected,
        "face_mar_mean":  face_mar_mean,
        "face_mar_std":   face_mar_std,
        "face_ear_mean":  face_ear_mean,
        "face_ear_std":   face_ear_std,
        "face_brow_mean": face_brow_mean,
        "face_brow_std":  face_brow_std,
    }

    features_list.append(feat)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
features_df = pd.DataFrame(features_list)
features_df.to_csv(output_csv, index=False)
print(f"\nAll features saved to {output_csv}")
print(f"Feature matrix shape: {features_df.shape}")
