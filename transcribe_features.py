"""
transcribe_features.py
───────────────────────
Run this ONCE after video_feature_extraction.py has produced video_features.csv.

CREMA-D ships with SentenceFilenames.csv which maps every clip filename to one
of 12 fixed sentences the actors spoke. This script uses that lookup — no
Whisper needed for training data — which is both faster and more accurate than
transcribing audio.

Produces:
  video_features_with_text.csv  (video_features.csv + 10 text feature columns)

Whisper is only used at inference time (live webcam clips in app.py).

Run
───
  python transcribe_features.py
  python transcribe_features.py /path/to/video_features.csv

CREMA-D sentence codes (3rd field in filename, e.g. 1001_IEO_ANG_XX.mp4)
─────────────────────────────────────────────────────────────────────────
  IEO  → "It's eleven o'clock"
  TIE  → "That is exactly what happened"
  IOM  → "I'm on my way to the meeting"
  IWW  → "I wonder what this is about"
  TAI  → "The airplane is almost full"
  MTI  → "Maybe tomorrow it will be cold"
  IWL  → "I would like a new alarm clock"
  ITH  → "I think I have a doctor's appointment"
  DFA  → "Don't forget a jacket"
  ITS  → "I think I've seen this before"
  TSI  → "The surface is slippery"
  WSI  → "We'll stop in a couple of minutes"
"""

import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR             = "/Users/janemaguire/Desktop/Final/CREMA-D"
DEFAULT_FEATURES_CSV = os.path.join(BASE_DIR, "video_features.csv")
SENTENCE_CSV         = os.path.join(BASE_DIR, "SentenceFilenames.csv")
OUTPUT_CSV           = os.path.join(BASE_DIR, "video_features_with_text.csv")

# ─────────────────────────────────────────────────────────────────────────────
# CREMA-D sentence lookup
# ─────────────────────────────────────────────────────────────────────────────

SENTENCE_MAP = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slippery",
    "WSI": "We'll stop in a couple of minutes",
}


def get_sentence_code(filename: str) -> str:
    """Extract 3-letter sentence code from CREMA-D filename.
    Format: {actorID}_{sentenceCode}_{emotion}_{level}.mp4
    """
    parts = filename.replace(".mp4", "").split("_")
    return parts[1].upper() if len(parts) >= 2 else ""


def load_sentence_csv(path: str) -> dict:
    """Load SentenceFilenames.csv → {key: sentence} dict."""
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, header=None)
        lookup = {str(r.iloc[0]).strip(): str(r.iloc[1]).strip()
                  for _, r in df.iterrows() if df.shape[1] >= 2}
        print(f"Loaded {len(lookup)} entries from SentenceFilenames.csv")
        return lookup
    except Exception as e:
        print(f"Could not parse SentenceFilenames.csv: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment lexicon (keep identical to app.py)
# ─────────────────────────────────────────────────────────────────────────────

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
    "dead": -0.85, "die": -0.85, "mad": -0.7, "furious": -0.9,
    "rage": -0.9, "upset": -0.65, "miserable": -0.85, "slippery": -0.3,
    "very": 1.3, "really": 1.2, "so": 1.15, "extremely": 1.5, "quite": 1.1,
    "absolutely": 1.4, "totally": 1.3, "completely": 1.3,
}


def extract_text_features(transcript: str, duration_seconds: float) -> dict:
    """Extract text features from transcript. Identical to app.py version."""
    text  = transcript.strip().lower()
    words = re.findall(r"[a-z']+", text)
    word_count     = len(words)
    speaking_rate  = word_count / max(duration_seconds, 0.1)
    has_transcript = 1.0 if word_count > 0 else 0.0
    exclamation_ratio = transcript.count("!") / max(word_count, 1)
    question_ratio    = transcript.count("?") / max(word_count, 1)

    sentiment_scores, intensifier = [], 1.0
    for word in words:
        score = SENTIMENT_LEXICON.get(word)
        if score is None:
            intensifier = 1.0; continue
        if score > 1.0:
            intensifier = score; continue
        sentiment_scores.append(score * intensifier)
        intensifier = 1.0

    lex_valence     = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
    lex_valence_std = float(np.std(sentiment_scores))  if sentiment_scores else 0.0
    sentences       = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
    avg_sent_length = word_count / max(len(sentences), 1)
    raw_words       = re.findall(r"[A-Za-z']+", transcript)
    caps_ratio      = sum(1 for w in raw_words if w.isupper() and len(w) > 1) / max(len(raw_words), 1)

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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(features_csv: str = DEFAULT_FEATURES_CSV) -> None:
    print(f"Loading features from: {features_csv}")
    df = pd.read_csv(features_csv)
    print(f"  {len(df)} clips loaded.")

    csv_lookup = load_sentence_csv(SENTENCE_CSV)
    text_rows  = []
    unmatched  = 0
    n          = len(df)

    for i, row in df.iterrows():
        fname = str(row["fileName"])

        # Try SentenceFilenames.csv by full filename first
        transcript = csv_lookup.get(fname, "")

        # Fall back to 3-letter code in filename
        if not transcript:
            code       = get_sentence_code(fname)
            transcript = SENTENCE_MAP.get(code, "")

        # Fall back to csv_lookup keyed by code
        if not transcript and code:
            transcript = csv_lookup.get(code, "")

        if not transcript:
            unmatched += 1

        # Estimate duration: CREMA-D clips average ~3s; refine by word count
        word_count = len(transcript.split()) if transcript else 0
        duration   = max(word_count / 2.5, 3.0)   # 2.5 words/sec typical

        feats = extract_text_features(transcript, duration)

        if i % 500 == 0 or i < 5:
            print(f"  [{i+1:5d}/{n}] {fname:<45s} → '{transcript[:50]}'")

        text_rows.append({"fileName": fname, "transcript": transcript, **feats})

    # Merge and save
    text_df = pd.DataFrame(text_rows)
    merged  = df.merge(text_df, on="fileName", how="left")
    text_cols = [c for c in text_df.columns if c.startswith("text_")]
    merged[text_cols] = merged[text_cols].fillna(0.0)
    merged.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✓ Saved {len(merged)} rows → {OUTPUT_CSV}")
    print(f"  Unmatched clips: {unmatched}/{n}")
    print(f"  Text features:   {text_cols}")

    print("\n── Correlations with ground-truth valence ──────────────────────")
    for col in text_cols:
        r = merged[col].corr(merged["valence"])
        print(f"  {col:<32s} r = {r:+.3f}")

    print("\n── Sample rows ─────────────────────────────────────────────────")
    print(merged[["fileName", "transcript", "text_lex_valence",
                  "valence"]].head(6).to_string(index=False))


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FEATURES_CSV
    run(csv_path)
