"""
transcribe_features.py  (v2)
─────────────────────────────
Same as v1 but points at the new video_features.csv produced by
video_feature_extraction.py v2 (which uses video_va_mapping_clean.csv).

No logic changes — the CREMA-D sentence lookup and text feature extraction
are identical. Output: video_features_with_text.csv

Run after video_feature_extraction.py:
  python transcribe_features.py
"""

import os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

BASE_DIR             = "/Users/janemaguire/Desktop/Final/CREMA-D"
DEFAULT_FEATURES_CSV = os.path.join(BASE_DIR, "video_features.csv")
SENTENCE_CSV         = os.path.join(BASE_DIR, "SentenceFilenames.csv")
OUTPUT_CSV           = os.path.join(BASE_DIR, "video_features_with_text.csv")

SENTENCE_MAP = {
    "IEO":"It's eleven o'clock","TIE":"That is exactly what happened",
    "IOM":"I'm on my way to the meeting","IWW":"I wonder what this is about",
    "TAI":"The airplane is almost full","MTI":"Maybe tomorrow it will be cold",
    "IWL":"I would like a new alarm clock","ITH":"I think I have a doctor's appointment",
    "DFA":"Don't forget a jacket","ITS":"I think I've seen this before",
    "TSI":"The surface is slippery","WSI":"We'll stop in a couple of minutes",
}

# Unified lexicon — matches app.py exactly
SENTIMENT_LEXICON = {
    "good":0.7,"great":0.8,"happy":0.9,"love":0.85,"wonderful":0.9,
    "excellent":0.85,"fantastic":0.9,"nice":0.6,"beautiful":0.8,
    "joy":0.85,"joyful":0.85,"pleased":0.7,"glad":0.65,"fine":0.4,
    "okay":0.2,"ok":0.2,"yes":0.3,"sure":0.3,"right":0.2,
    "bright":0.5,"warm":0.5,"thanks":0.6,"thank":0.6,"welcome":0.5,
    "excited":0.75,"amazing":0.85,"brilliant":0.8,"perfect":0.85,
    "full":0.2,"almost":0.1,"new":0.3,"like":0.4,"think":0.1,
    "stop":-0.2,"cold":-0.3,"forget":-0.3,"wonder":0.2,
    "bad":-0.7,"sad":-0.8,"angry":-0.8,"hate":-0.9,"terrible":-0.85,
    "awful":-0.85,"horrible":-0.9,"wrong":-0.5,"no":-0.3,"not":-0.3,
    "never":-0.4,"fear":-0.75,"scared":-0.75,"afraid":-0.7,
    "disgust":-0.8,"disgusting":-0.85,"gross":-0.7,"ugly":-0.65,
    "hurt":-0.7,"pain":-0.75,"suffer":-0.8,"cry":-0.65,"dark":-0.4,
    "lonely":-0.75,"lost":-0.5,"fail":-0.65,"failed":-0.65,
    "dead":-0.85,"die":-0.85,"kill":-0.85,"mad":-0.7,
    "furious":-0.9,"rage":-0.9,"upset":-0.65,"miserable":-0.85,"slippery":-0.3,
    "very":1.3,"really":1.2,"so":1.15,"extremely":1.5,"quite":1.1,
    "absolutely":1.4,"totally":1.3,"completely":1.3,
}

def get_sentence_code(filename):
    parts = filename.replace(".mp4","").split("_")
    return parts[1].upper() if len(parts)>=2 else ""

def load_sentence_csv(path):
    if not os.path.exists(path): return {}
    try:
        df = pd.read_csv(path,header=None)
        return {str(r.iloc[0]).strip():str(r.iloc[1]).strip() for _,r in df.iterrows() if df.shape[1]>=2}
    except: return {}

def extract_text_features(transcript, duration_seconds):
    text  = transcript.strip().lower()
    words = re.findall(r"[a-z']+", text)
    wc    = len(words)
    speaking_rate     = wc/max(duration_seconds,0.1)
    has_transcript    = 1.0 if wc>0 else 0.0
    exclamation_ratio = transcript.count("!")/max(wc,1)
    question_ratio    = transcript.count("?")/max(wc,1)
    scores=[]; intens=1.0
    for w in words:
        s=SENTIMENT_LEXICON.get(w)
        if s is None: intens=1.0; continue
        if s>1.0: intens=s; continue
        scores.append(s*intens); intens=1.0
    lex_valence     = float(np.mean(scores))     if scores else 0.0
    lex_valence_std = float(np.std(scores))      if scores else 0.0
    sentences       = [s.strip() for s in re.split(r'[.!?]+',transcript) if s.strip()]
    avg_sent_length = wc/max(len(sentences),1)
    raw_words       = re.findall(r"[A-Za-z']+",transcript)
    caps_ratio      = sum(1 for w in raw_words if w.isupper() and len(w)>1)/max(len(raw_words),1)
    return {
        "text_has_transcript":    has_transcript,
        "text_word_count":        float(wc),
        "text_speaking_rate":     float(speaking_rate),
        "text_exclamation_ratio": float(exclamation_ratio),
        "text_question_ratio":    float(question_ratio),
        "text_lex_valence":       float(np.clip(lex_valence,-1.0,1.0)),
        "text_lex_valence_std":   float(lex_valence_std),
        "text_sentiment_words":   float(len(scores)),
        "text_avg_sent_length":   float(avg_sent_length),
        "text_caps_ratio":        float(caps_ratio),
    }

def run(features_csv=DEFAULT_FEATURES_CSV):
    print(f"Loading: {features_csv}")
    df = pd.read_csv(features_csv)
    print(f"  {len(df)} clips")
    csv_lookup = load_sentence_csv(SENTENCE_CSV)
    text_rows=[]; unmatched=0; n=len(df)
    for i,row in df.iterrows():
        fname      = str(row["fileName"])
        transcript = csv_lookup.get(fname,"")
        if not transcript:
            code       = get_sentence_code(fname)
            transcript = SENTENCE_MAP.get(code,"")
        if not transcript and code:
            transcript = csv_lookup.get(code,"")
        if not transcript: unmatched+=1
        wc       = len(transcript.split()) if transcript else 0
        duration = max(wc/2.5, 3.0)
        feats    = extract_text_features(transcript, duration)
        if i%500==0 or i<3:
            print(f"  [{i+1:5d}/{n}] {fname:<45s} → '{transcript[:50]}'")
        text_rows.append({"fileName":fname,"transcript":transcript,**feats})
    text_df = pd.DataFrame(text_rows)
    merged  = df.merge(text_df,on="fileName",how="left")
    tcols   = [c for c in text_df.columns if c.startswith("text_")]
    merged[tcols] = merged[tcols].fillna(0.0)
    merged.to_csv(OUTPUT_CSV,index=False)
    print(f"\n✓ Saved {len(merged)} rows → {OUTPUT_CSV}")
    print(f"  Unmatched: {unmatched}/{n}")
    print(f"  Text features: {tcols}")
    print("\n── Correlations with valence ──────────────────────────────────")
    for col in tcols:
        r = merged[col].corr(merged["valence"])
        print(f"  {col:<32s} r={r:+.3f}")

if __name__=="__main__":
    csv_path = sys.argv[1] if len(sys.argv)>1 else DEFAULT_FEATURES_CSV
    run(csv_path)
