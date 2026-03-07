# Multimodal Emotion Recognition and Adaptive Music Generation

## Overview

This project builds a multimodal emotion recognition system using the CREMA-D dataset. The goal is to predict a speaker’s emotional state from video clips by analyzing both **audio signals** and **facial expressions**. Emotional state is represented using the **valence–arousal model**, where valence measures how positive or negative an emotion is, and arousal measures emotional intensity.

The predicted emotional values are then used to guide a simple **music generation system**, where musical parameters such as tempo and key are adjusted to reflect the detected emotion. This project explores how multimodal emotion recognition can serve as the foundation for adaptive media systems.

---

## Dataset

The system is trained using the **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**, which contains over 7,000 video clips of actors portraying different emotional expressions. Each clip has been annotated by multiple human raters who labeled the perceived emotion and intensity.

The dataset includes six primary emotion categories:

* Anger
* Disgust
* Fear
* Happy
* Neutral
* Sad

For this project, these categorical labels are converted into **continuous valence and arousal values** using the aggregated ratings provided in the dataset.

---

## Project Pipeline

The overall pipeline consists of several stages that transform raw video data into emotion-aware music output.

### 1. Data Preprocessing

The CREMA-D video files are first standardized and aligned with the dataset’s metadata. Human annotation data from the dataset is processed to determine the dominant emotional label for each clip. These labels are then converted into continuous **valence and arousal values** that will serve as the ground truth for training the model.

### 2. Feature Extraction

Features are extracted separately from the audio and visual components of each video.

**Audio features** capture acoustic properties of speech, such as pitch, energy, and spectral characteristics.

**Visual features** are derived from facial expressions detected in the video frames, capturing information about facial movement and expression.

These features provide complementary signals for understanding emotional expression.

### 3. Emotion Prediction Models

Machine learning models are trained to predict the valence and arousal values associated with each video. Three model configurations are evaluated:

* **Audio-only model**, which uses speech features
* **Visual-only model**, which uses facial expression features
* **Multimodal model**, which combines both feature types

Comparing these models helps determine how much each modality contributes to accurate emotion recognition.

### 4. Model Evaluation

The dataset is divided into training and testing sets to evaluate model performance. Regression metrics such as **Root Mean Squared Error (RMSE)** are used to measure how accurately the models predict valence and arousal.

The comparison between audio-only, visual-only, and multimodal models allows us to assess whether combining modalities improves emotion prediction.

### 5. Emotion-Driven Music Generation

The predicted valence and arousal values are used to control a simple music generation system. Emotional dimensions are mapped to musical properties such as:

* **Valence → musical key (major or minor)**
* **Arousal → tempo and rhythmic intensity**

Using these mappings, the system generates music that reflects the emotional state detected in the video input.

---

## Goal

The goal of this project is to demonstrate how multimodal emotion recognition can be integrated with generative systems to create adaptive media experiences. By combining facial and vocal emotion cues with music generation techniques, this system serves as a prototype for emotion-aware interactive applications.
