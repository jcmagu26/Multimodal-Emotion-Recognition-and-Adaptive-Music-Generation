"""
clean_ground_truth.py
─────────────────────
Cleans the CREMA-D ground truth VA mapping file by:

  1. Removing ambiguous clips (multi-label codes like N:S, F:N, A:D etc.)
     These are clips where raters disagreed — their VA values average toward
     zero and poison the training signal.

  2. Removing clips with VA=(0,0)
     These are contested/neutral clips with no useful label information.

  3. Downsampling NEU from 3,897 → 500
     NEU has 11x more clips than HAP (353). The extreme imbalance causes the
     model to learn "predict near-zero" as the optimal strategy. NEU also has
     the weakest VA signal (mean v=0.079, a=0.079) so losing most of them
     costs very little information while fixing the collapse.

Before filtering:
  Total: 7,442 clips
  NEU: 3,897  ANG: 986  FEA: 645  DIS: 547  SAD: 370  HAP: 353
  Valence range: -0.443 to +0.160  (0.603)
  Arousal range:  0.090 to  0.508  (0.418)

After filtering:
  Total: ~2,900 clips
  All classes within ~350–986 clips of each other
  Valence range: -0.571 to +0.635  (1.206)  ← doubled
  Arousal range:  0.079 to  0.653  (0.574)  ← 37% wider

Run:
  python clean_ground_truth.py

Output:
  /Users/janemaguire/Desktop/Final/CREMA-D/video_va_mapping_clean.csv

Then update video_feature_extraction.py to point at this file and re-run it,
then retrain with train_and_save.py.
"""

import pandas as pd
import os

INPUT_PATH  = "/Users/janemaguire/Desktop/Final/CREMA-D/video_va_mapping_dynamic.csv"
OUTPUT_PATH = "/Users/janemaguire/Desktop/Final/CREMA-D/video_va_mapping_clean.csv"
RANDOM_SEED = 42
NEU_TARGET  = 500   # downsample Neutral to this — slightly above other minority classes

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_PATH)
print(f"Loaded: {len(df)} clips")
print(f"Columns: {df.columns.tolist()}")
print(f"\nRaw emotion code counts:")
print(df['emotion'].value_counts().head(15).to_string())

# ── Step 1: Remove ambiguous multi-label clips ────────────────────────────────
# Codes like N:S, F:N, A:D represent rater disagreement.
# Their VA values are averaged toward zero and carry no clean signal.

before = len(df)
clean = df[~df['emotion'].str.contains(':', na=False)].copy()
print(f"\nStep 1 — Removed ambiguous clips: {before - len(clean)} removed, {len(clean)} remain")

# ── Step 2: Remove VA=(0,0) clips ─────────────────────────────────────────────
# These are contested clips assigned no emotional content.

before = len(clean)
clean = clean[(clean['valence'] != 0.0) | (clean['arousal'] != 0.0)]
print(f"Step 2 — Removed VA=(0,0) clips: {before - len(clean)} removed, {len(clean)} remain")

# ── Step 3: Map single-letter codes to 3-letter CREMA-D codes ────────────────

code_map = {'A': 'ANG', 'D': 'DIS', 'F': 'FEA',
            'H': 'HAP', 'N': 'NEU', 'S': 'SAD'}
clean['emo_code'] = clean['emotion'].map(code_map)

before = len(clean)
clean = clean[clean['emo_code'].notna()].reset_index(drop=True)
print(f"Step 3 — Unmapped codes removed: {before - len(clean)} removed")

print(f"\nClass counts before downsampling:")
print(clean['emo_code'].value_counts().to_string())

# ── Step 4: Downsample NEU ────────────────────────────────────────────────────

neu  = clean[clean['emo_code'] == 'NEU'].sample(n=NEU_TARGET, random_state=RANDOM_SEED)
rest = clean[clean['emo_code'] != 'NEU']
clean = pd.concat([rest, neu], ignore_index=True).sample(
    frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"\nStep 4 — NEU downsampled to {NEU_TARGET}")
print(f"\nFinal class counts:")
print(clean['emo_code'].value_counts().to_string())

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\nFinal dataset: {len(clean)} clips")
print(f"\nVA means per emotion:")
print(clean.groupby('emo_code')[['valence', 'arousal']].mean().round(3).to_string())

stats = clean.groupby('emo_code')[['valence', 'arousal']].mean()
v_range = stats['valence'].max() - stats['valence'].min()
a_range = stats['arousal'].max() - stats['arousal'].min()
print(f"\nVA separation:")
print(f"  Valence: {stats['valence'].min():.3f} to {stats['valence'].max():.3f}  (range={v_range:.3f})")
print(f"  Arousal: {stats['arousal'].min():.3f} to {stats['arousal'].max():.3f}  (range={a_range:.3f})")

# ── Save ──────────────────────────────────────────────────────────────────────
# Keep same columns as the original so video_feature_extraction.py works unchanged
# (just update the path in that file to point here)

out = clean[['fileName', 'emotion', 'level', 'valence', 'arousal']].reset_index(drop=True)
out.to_csv(OUTPUT_PATH, index=False)

print(f"\n✓ Saved clean ground truth → {OUTPUT_PATH}")
print(f"\nNext steps:")
print(f"  1. In video_feature_extraction.py, change:")
print(f'       va_mapping = pd.read_csv(".../video_va_mapping_dynamic.csv")')
print(f'     to:')
print(f'       va_mapping = pd.read_csv(".../video_va_mapping_clean.csv")')
print(f"  2. Run: python video_feature_extraction.py")
print(f"     (re-extracts features for the ~2,900 clean clips)")
print(f"  3. Run: python transcribe_features.py  (if using text features)")
print(f"  4. Run: python train_and_save.py")
print(f"     (retrains with clean ground truth + new classification head)")
2
