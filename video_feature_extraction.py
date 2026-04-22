"""
video_feature_extraction.py  (v2)
───────────────────────────────────
Changes from v1:
  - Points at video_va_mapping_clean.csv (filtered ground truth)
  - 21 new discriminative features:
      shimmer, CPP, mfcc_delta_std x13, energy_slope,
      pitch_p10/p25/p75/p90, voiced_trans_rate, pause_ratio,
      lowfreq_energy_ratio, spectral_entropy
  - subprocess ffmpeg (surfaces errors vs silent os.system)
  - NaN-safe std for single-frame clips
  - ETA progress indicator

Run:  python video_feature_extraction.py
"""

import os, subprocess, time, urllib.request
import cv2, librosa, numpy as np, pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# ── Paths ─────────────────────────────────────────────────────────────────────
mp4_folder   = "/Users/janemaguire/Desktop/Final/MP4"
audio_folder = "/Users/janemaguire/Desktop/Final/CREMA-D/AudioWAV"
output_csv   = "/Users/janemaguire/Desktop/Final/CREMA-D/video_features.csv"
MODEL_PATH   = "/Users/janemaguire/Desktop/Final/CREMA-D/face_landmarker.task"
GT_CSV       = "/Users/janemaguire/Desktop/Final/CREMA-D/video_va_mapping_clean.csv"

os.makedirs(audio_folder, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading face_landmarker.task...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH)

options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE, num_faces=1,
    min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False, output_facial_transformation_matrixes=False)
face_landmarker = FaceLandmarker.create_from_options(options)

va_mapping = pd.read_csv(GT_CSV)
print(f"Loaded {len(va_mapping)} clips from clean ground truth")

# ── Landmark indices ──────────────────────────────────────────────────────────
MOUTH_TOP=13; MOUTH_BOTTOM=14; MOUTH_LEFT=61; MOUTH_RIGHT=291
LEFT_EYE_TOP=159; LEFT_EYE_BOTTOM=145; LEFT_EYE_LEFT=33; LEFT_EYE_RIGHT=133
RIGHT_EYE_TOP=386; RIGHT_EYE_BOTTOM=374; RIGHT_EYE_LEFT=362; RIGHT_EYE_RIGHT=263
LEFT_BROW_INNER=107; RIGHT_BROW_INNER=336
UPPER_LIP_CENTER=13; LOWER_LIP_CENTER=14
LEFT_CHEEK=234; RIGHT_CHEEK=454; NOSE_TIP=4

def safe_std(s):
    v = float(s.std())
    return 0.0 if (v != v) else v  # NaN check

def euclidean(p1,p2):
    return np.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2)

def compute_landmark_features(lm):
    mw = euclidean(lm[MOUTH_LEFT], lm[MOUTH_RIGHT])
    mar = euclidean(lm[MOUTH_TOP], lm[MOUTH_BOTTOM]) / (mw+1e-6)
    le = euclidean(lm[LEFT_EYE_TOP],lm[LEFT_EYE_BOTTOM])/(euclidean(lm[LEFT_EYE_LEFT],lm[LEFT_EYE_RIGHT])+1e-6)
    re = euclidean(lm[RIGHT_EYE_TOP],lm[RIGHT_EYE_BOTTOM])/(euclidean(lm[RIGHT_EYE_LEFT],lm[RIGHT_EYE_RIGHT])+1e-6)
    ear = (le+re)/2.0
    brow_raise = (euclidean(lm[LEFT_BROW_INNER],lm[LEFT_EYE_LEFT])+euclidean(lm[RIGHT_BROW_INNER],lm[RIGHT_EYE_RIGHT]))/2.0
    lip_cy = (lm[UPPER_LIP_CENTER].y+lm[LOWER_LIP_CENTER].y)/2.0
    cor_cy = (lm[MOUTH_LEFT].y+lm[MOUTH_RIGHT].y)/2.0
    smile_ratio = (lip_cy-cor_cy)/(mw+1e-6)
    cheek_raise = lm[NOSE_TIP].y-(lm[LEFT_CHEEK].y+lm[RIGHT_CHEEK].y)/2.0
    brow_furrow = euclidean(lm[LEFT_BROW_INNER],lm[RIGHT_BROW_INNER])
    return {"mar":mar,"ear":ear,"brow_raise":brow_raise,
            "smile_ratio":smile_ratio,"cheek_raise":cheek_raise,"brow_furrow":brow_furrow}

def extract_wav(mp4_path, wav_path):
    if os.path.exists(wav_path): return
    r = subprocess.run(["ffmpeg","-i",mp4_path,"-q:a","0","-map","a",wav_path,"-y","-loglevel","error"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {r.stderr[:200]}")

# ── Main loop ─────────────────────────────────────────────────────────────────
features_list = []
n = len(va_mapping)
t0 = time.time()

for idx, row in va_mapping.iterrows():
    mp4_file  = os.path.join(mp4_folder, row["fileName"])
    fname_base = row["fileName"].replace(".mp4","")
    wav_file  = os.path.join(audio_folder, f"{fname_base}.wav")
    elapsed = time.time()-t0
    eta = (elapsed/max(idx,1))*(n-idx) if idx>0 else 0
    print(f"[{idx+1:4d}/{n}] {row['fileName']}  eta={eta/60:.1f}m")

    try:
        extract_wav(mp4_file, wav_file)
        y, sr = librosa.load(wav_file, sr=None)
    except Exception as e:
        print(f"  SKIP: {e}"); continue

    # ── MFCCs ──────────────────────────────────────────────────────────────
    mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std  = np.std(mfccs,  axis=1)
    delta      = librosa.feature.delta(mfccs)
    delta2     = librosa.feature.delta(mfccs, order=2)
    delta_mean = np.mean(delta,  axis=1)
    delta2_mean= np.mean(delta2, axis=1)
    delta_std  = np.std(delta,   axis=1)   # NEW: variability of spectral change

    # ── Pitch ──────────────────────────────────────────────────────────────
    pitch        = librosa.yin(y, fmin=50, fmax=500)
    pitch_voiced = pitch[pitch>0]
    pitch_mean   = float(np.mean(pitch_voiced))  if len(pitch_voiced)>0 else 0.0
    pitch_std    = float(np.std(pitch_voiced))   if len(pitch_voiced)>0 else 0.0
    pitch_range  = float(np.ptp(pitch_voiced))   if len(pitch_voiced)>0 else 0.0
    voiced_ratio = len(pitch_voiced)/(len(pitch)+1e-6)
    if len(pitch_voiced)>=4:
        pitch_p10=float(np.percentile(pitch_voiced,10)); pitch_p25=float(np.percentile(pitch_voiced,25))
        pitch_p75=float(np.percentile(pitch_voiced,75)); pitch_p90=float(np.percentile(pitch_voiced,90))
    else:
        pitch_p10=pitch_p25=pitch_p75=pitch_p90=pitch_mean
    pitch_slope = float(np.polyfit(np.linspace(0,1,len(pitch_voiced)),pitch_voiced,1)[0]) if len(pitch_voiced)>1 else 0.0
    jitter = float(np.mean(np.abs(np.diff(pitch_voiced)))/(pitch_mean+1e-6)) if len(pitch_voiced)>2 else 0.0

    # Shimmer — amplitude cycle-to-cycle variation
    rms_frames = librosa.feature.rms(y=y, frame_length=int(sr*0.025), hop_length=int(sr*0.010))[0]
    rms_v = rms_frames[rms_frames>rms_frames.mean()*0.1]
    shimmer = float(np.mean(np.abs(np.diff(rms_v)))/(np.mean(rms_v)+1e-6)) if len(rms_v)>2 else 0.0

    voiced_mask = (pitch>0).astype(int)
    voiced_trans_rate = float(np.sum(np.abs(np.diff(voiced_mask)))/(len(voiced_mask)+1e-6))
    pause_ratio = float(1.0-voiced_ratio)

    # ── Energy ─────────────────────────────────────────────────────────────
    rms_all     = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms_all))
    energy_std  = float(np.std(rms_all))
    energy_slope= float(np.polyfit(np.linspace(0,1,len(rms_all)),rms_all,1)[0]) if len(rms_all)>1 else 0.0

    # ── Spectral ────────────────────────────────────────────────────────────
    spec_centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y,sr=sr)))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y,sr=sr)))
    spec_rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=y,sr=sr)))
    zcr            = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    stft_mag = np.abs(librosa.stft(y))
    spec_norm = stft_mag/(stft_mag.sum(axis=0,keepdims=True)+1e-10)
    spectral_entropy = float(np.mean(-np.sum(spec_norm*np.log(spec_norm+1e-10),axis=0)))

    freqs    = librosa.fft_frequencies(sr=sr)
    lf_mask  = (freqs>=100)&(freqs<=300)
    sp2      = stft_mag**2
    lowfreq_energy_ratio = float(sp2[lf_mask].sum(axis=0).mean()/(sp2.sum(axis=0).mean()+1e-10))

    spec_flatness     = librosa.feature.spectral_flatness(y=y)
    spec_flatness_mean= float(np.mean(spec_flatness))
    spec_flatness_std = float(np.std(spec_flatness))

    # ── HNR & CPP ──────────────────────────────────────────────────────────
    harmonic   = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    hnr = float(10*np.log10((np.mean(harmonic**2)+1e-10)/(np.mean(percussive**2)+1e-10)))

    cepstrum  = np.real(np.fft.ifft(np.log(np.abs(librosa.stft(y))+1e-10),axis=0))
    quefrency = np.arange(cepstrum.shape[0])/sr
    f0m       = (quefrency>0.002)&(quefrency<0.02)
    if f0m.sum()>0:
        y_vals   = np.abs(cepstrum[f0m]).mean(axis=1) if cepstrum.ndim>1 else np.abs(cepstrum[f0m])
        x_idx    = np.where(f0m)[0]
        baseline = float(np.polyval(np.polyfit(x_idx,y_vals,1),x_idx).mean())
        cpp      = float(np.max(y_vals)-baseline)
    else:
        cpp = 0.0

    # ── Chroma & Mel ────────────────────────────────────────────────────────
    chroma      = librosa.feature.chroma_stft(y=y,sr=sr)
    chroma_mean = np.mean(chroma,axis=1)
    chroma_std  = float(np.std(chroma_mean))
    chroma_max  = float(np.max(chroma_mean))
    chroma_entropy = float(-np.sum(chroma_mean/(chroma_mean.sum()+1e-10)*np.log(chroma_mean/(chroma_mean.sum()+1e-10)+1e-10)))

    mel_spec = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=40)
    mel_db   = librosa.power_to_db(mel_spec,ref=np.max)
    mel_mean = float(np.mean(mel_db)); mel_std=float(np.std(mel_db))
    mel_skew = float(np.mean(((mel_db-mel_mean)/(mel_std+1e-10))**3))

    # ── Visual ──────────────────────────────────────────────────────────────
    cap=cv2.VideoCapture(mp4_file); frame_feats=[]
    while True:
        ret,frame=cap.read()
        if not ret: break
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=face_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
        if res.face_landmarks: frame_feats.append(compute_landmark_features(res.face_landmarks[0]))
    cap.release()

    if frame_feats:
        ff=pd.DataFrame(frame_feats)
        fd=1.0
        fmarm=float(ff["mar"].mean());    fmars=safe_std(ff["mar"])
        fearm=float(ff["ear"].mean());    fears=safe_std(ff["ear"])
        fbrowm=float(ff["brow_raise"].mean()); fbrows=safe_std(ff["brow_raise"])
        fsmm=float(ff["smile_ratio"].mean());  fsms=safe_std(ff["smile_ratio"])
        fchkm=float(ff["cheek_raise"].mean()); fchks=safe_std(ff["cheek_raise"])
        ffurm=float(ff["brow_furrow"].mean()); ffurs=safe_std(ff["brow_furrow"])
    else:
        fd=fmarm=fmars=fearm=fears=fbrowm=fbrows=fsmm=fsms=fchkm=fchks=ffurm=ffurs=0.0

    feat = {
        "fileName":row["fileName"], "valence":row["valence"], "arousal":row["arousal"],
        **{f"mfcc_mean_{i+1}":   float(mfccs_mean[i])  for i in range(13)},
        **{f"mfcc_std_{i+1}":    float(mfccs_std[i])   for i in range(13)},
        **{f"mfcc_delta_{i+1}":  float(delta_mean[i])  for i in range(13)},
        **{f"mfcc_delta2_{i+1}": float(delta2_mean[i]) for i in range(13)},
        **{f"mfcc_delta_std_{i+1}": float(delta_std[i]) for i in range(13)},
        "pitch_mean":pitch_mean,"pitch_std":pitch_std,"pitch_range":pitch_range,
        "voiced_ratio":voiced_ratio,"pitch_slope":pitch_slope,
        "jitter":jitter,"shimmer":shimmer,
        "pitch_p10":pitch_p10,"pitch_p25":pitch_p25,"pitch_p75":pitch_p75,"pitch_p90":pitch_p90,
        "voiced_trans_rate":voiced_trans_rate,"pause_ratio":pause_ratio,
        "energy_mean":energy_mean,"energy_std":energy_std,"energy_slope":energy_slope,
        "spec_centroid":spec_centroid,"spec_bandwidth":spec_bandwidth,
        "spec_rolloff":spec_rolloff,"zcr":zcr,
        "spectral_entropy":spectral_entropy,"lowfreq_energy_ratio":lowfreq_energy_ratio,
        "hnr":hnr,"cpp":cpp,
        "spec_flatness_mean":spec_flatness_mean,"spec_flatness_std":spec_flatness_std,
        "chroma_std":chroma_std,"chroma_max":chroma_max,"chroma_entropy":chroma_entropy,
        **{f"chroma_mean_{i+1}":float(chroma_mean[i]) for i in range(12)},
        "mel_mean":mel_mean,"mel_std":mel_std,"mel_skew":mel_skew,
        "face_detected":fd,
        "face_mar_mean":fmarm,"face_mar_std":fmars,
        "face_ear_mean":fearm,"face_ear_std":fears,
        "face_brow_mean":fbrowm,"face_brow_std":fbrows,
        "face_smile_mean":fsmm,"face_smile_std":fsms,
        "face_cheek_mean":fchkm,"face_cheek_std":fchks,
        "face_furrow_mean":ffurm,"face_furrow_std":ffurs,
    }
    features_list.append(feat)

features_df = pd.DataFrame(features_list)
features_df.to_csv(output_csv, index=False)
print(f"\n✓ Saved {len(features_df)} rows → {output_csv}")
print(f"Feature matrix: {features_df.shape}  ({features_df.shape[1]-3} features + 3 label cols)")

print("\n── Correlations with valence / arousal ──────────────────────────")
check = ["hnr","cpp","shimmer","jitter","spectral_entropy","lowfreq_energy_ratio",
         "energy_slope","voiced_trans_rate","pause_ratio","pitch_p90","face_smile_mean"]
for c in check:
    if c in features_df:
        rv=features_df[c].corr(features_df["valence"])
        ra=features_df[c].corr(features_df["arousal"])
        print(f"  {c:<28s} valence r={rv:+.3f}  arousal r={ra:+.3f}")
