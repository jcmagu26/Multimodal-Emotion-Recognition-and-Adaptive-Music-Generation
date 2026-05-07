# Resonance — Multimodal Emotion Recognition and Adaptive Music Generation

A multimodal machine learning system that detects emotional state from video clips and generates music that reflects the predicted emotion. Built using the CREMA-D dataset.

**Authors:** Olivia Doherty and Jane Maguire

---

## Overview

Resonance predicts a speaker's emotional state from a short video clip by analysing three complementary signals:

- **Audio** — pitch, energy, MFCCs, and spectral features extracted from speech
- **Visual** — facial expression geometry tracked via MediaPipe
- **Text** — lexical sentiment from speech transcription via Whisper

The predicted emotion is represented as a continuous valence (positive/negative) and arousal (intensity) score, which drives a generative music system that adapts tempo, key, instrumentation, and rhythm to match the detected emotional state.

---

## Project Structure

```
resonance/
├── clean_ground_truth.py        # Step 1 — clean and balance the CREMA-D labels
├── video_feature_extraction.py  # Step 2 — extract audio and visual features
├── transcribe_features.py       # Step 3 — add Whisper text features
├── train_and_save.py            # Step 4 — train the model and save artifacts
├── app.py                       # Step 5 — run the Flask backend
├── index.html                   # Frontend UI
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses the **CREMA-D** dataset (Crowd-sourced Emotional Multimodal Actors Dataset), which contains 7,442 video clips of 91 actors across six emotion categories: Anger, Disgust, Fear, Happy, Neutral, and Sad.

Download it from the official repository: https://github.com/CheyneyComputerScience/CREMA-D

Once downloaded, update the file paths at the top of each script to point to your local copy. The key paths to set are:

| Script | Path variable | What it points to |
|--------|--------------|-------------------|
| `clean_ground_truth.py` | `INPUT_PATH` | `video_va_mapping_dynamic.csv` |
| `video_feature_extraction.py` | `mp4_folder`, `audio_folder`, `GT_CSV` | MP4 directory, WAV output directory, cleaned CSV |
| `transcribe_features.py` | `DEFAULT_FEATURES_CSV` | Output of step 2 |
| `train_and_save.py` | `WITH_TEXT_CSV` | Output of step 3 |
| `app.py` | `MODEL_DIR` | Where trained model files are saved |
 
---

## Installation

```bash
pip install -r requirements.txt
```

You will also need:
- **ffmpeg** — for extracting audio from video files (`brew install ffmpeg` on macOS)

---

## Running the Pipeline

Run the scripts in order. Each step produces output that the next step depends on.

**Step 1 — Clean the ground truth labels**
```bash
python clean_ground_truth.py
```
Removes ambiguous clips, drops VA=(0,0) entries, and downsamples the Neutral class to balance training. Outputs `video_va_mapping_clean.csv`.

**Step 2 — Extract audio and visual features**
```bash
python video_feature_extraction.py
```
Extracts ~170 features per clip from audio (MFCCs, pitch, energy, spectral) and video (MediaPipe facial landmarks). Outputs `video_features.csv`. This step is slow — expect several hours for the full dataset.

**Step 3 — Add text features**
```bash
python transcribe_features.py
```
Maps each clip to its scripted sentence and extracts lexical sentiment features. Outputs `video_features_with_text.csv`.

**Step 4 — Train the model**
```bash
python train_and_save.py
```
Trains a multi-task neural network on valence/arousal regression and six-class emotion classification. Saves model artifacts to the `models/` directory. Reports 5-fold cross-validation results and final test set performance.

**Step 5 — Run the app**
```bash
python app.py
```
Starts the Flask server at `http://localhost:5000`. Open `index.html` in a browser to use the interface.

> **Note:** Open `index.html` via a local server rather than directly as a `file://` URL, as the browser will block camera access otherwise. A simple option: `python -m http.server 8080` then visit `http://localhost:8080/index.html`.

---

## Model Performance

Results on the CREMA-D test set (80/20 stratified split):

| Model | Arousal RMSE | Valence RMSE | Class Accuracy |
|-------|-------------|-------------|----------------|
| Audio only | 0.1639 | 0.3406 | 61.4% |
| Visual only | 0.1901 | 0.3421 | 62.4% |
| Text only | 0.2171 | 0.3922 | 28.5% |
| **Multimodal (combined)** | **0.1594** | **0.2958** | **76.2%** |

---

## Model Artifacts

The `models/` directory is not included in this repository. It is generated automatically when you run `train_and_save.py` and will contain:

```
models/
├── va_mlp.pt              # Trained model weights
├── scaler.joblib          # Feature normalisation scaler
├── feature_cols.joblib    # Ordered list of feature column names
├── mlp_config.joblib      # Model architecture config
└── label_encoder.joblib   # Emotion label encoder
├── face_landmarker.task   # Downloaded automatically on first run
```

---

## Dependencies

See `requirements.txt`. Key libraries:

- `torch` — model training and inference
- `librosa` — audio feature extraction
- `opencv-python` + `mediapipe` — facial landmark detection
- `scikit-learn` — preprocessing and evaluation
- `flask` + `flask-cors` — web server
- `midiutil` — MIDI file generation
- `openai-whisper` — speech transcription (optional, improves valence prediction)

---
