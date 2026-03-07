import os
import cv2
import librosa
import numpy as np
import pandas as pd

# Paths
mp4_folder = "/Users/janemaguire/Desktop/Final/CREMA-D/MP4"
audio_folder = "/Users/janemaguire/Desktop/Final/CREMA-D/AudioWAV"
output_csv = "/Users/janemaguire/Desktop/Final/CREMA-D/video_features.csv"

# Make sure audio folder exists
os.makedirs(audio_folder, exist_ok=True)

# Load ground truth mapping
va_mapping = pd.read_csv("/Users/janemaguire/Desktop/Final/CREMA-D/video_va_mapping_dynamic.csv")

# Initialize face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Feature extraction
features_list = []

for idx, row in va_mapping.iterrows():
    mp4_file = os.path.join(mp4_folder, row['fileName'])
    fname_base = row['fileName'].replace(".mp4","")
    
    print(f"Processing {mp4_file}...")
    
    # Extract audio features 
    wav_file = os.path.join(audio_folder, f"{fname_base}.wav")
    # Extract audio if not already extracted
    if not os.path.exists(wav_file):
        os.system(f"ffmpeg -i \"{mp4_file}\" -q:a 0 -map a \"{wav_file}\" -y")
    
    # Load audio
    y, sr = librosa.load(wav_file, sr=None)
    
    # Extract MFCCs (mean over time)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Extract pitch and energy
    rms = librosa.feature.rms(y=y)
    pitch = librosa.yin(y, fmin=50, fmax=500)  # estimate fundamental frequency
    energy_mean = np.mean(rms)
    pitch_mean = np.mean(pitch)
    
    # Extract face features 
    cap = cv2.VideoCapture(mp4_file)
    face_count = 0
    face_areas = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x,y,w,h) in faces:
            face_areas.append(w*h)
            face_count += 1
    cap.release()
    
    # Face summary features
    face_avg_area = np.mean(face_areas) if face_areas else 0
    face_count_total = face_count
    
    # Combine features 
    features_list.append({
        "fileName": row['fileName'],
        "valence": row['valence'],
        "arousal": row['arousal'],
        "mfcc_mean_1": mfccs_mean[0],
        "mfcc_mean_2": mfccs_mean[1],
        "mfcc_mean_3": mfccs_mean[2],
        "mfcc_mean_4": mfccs_mean[3],
        "mfcc_mean_5": mfccs_mean[4],
        "mfcc_mean_6": mfccs_mean[5],
        "mfcc_mean_7": mfccs_mean[6],
        "mfcc_mean_8": mfccs_mean[7],
        "mfcc_mean_9": mfccs_mean[8],
        "mfcc_mean_10": mfccs_mean[9],
        "mfcc_mean_11": mfccs_mean[10],
        "mfcc_mean_12": mfccs_mean[11],
        "mfcc_mean_13": mfccs_mean[12],
        "pitch_mean": pitch_mean,
        "energy_mean": energy_mean,
        "face_count": face_count_total,
        "face_avg_area": face_avg_area
    })

# Save features to CSV
features_df = pd.DataFrame(features_list)
features_df.to_csv(output_csv, index=False)
print(f"All features saved to {output_csv}")