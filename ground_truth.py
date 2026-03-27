import pandas as pd
import os

# Paths
crema_csv = "/Users/janemaguire/Desktop/Final/CREMA-D/processedResults/tabulatedVotes.csv"
mp4_folder = "/Users/janemaguire/Desktop/Final/MP4"
output_csv = "/Users/janemaguire/Desktop/Final/CREMA-D/video_va_mapping_dynamic.csv"

# Load tabulated votes
votes = pd.read_csv(crema_csv)

# 2D valence-arousal base values per emotion category
# Based on Russell's circumplex model of affect
# Each tuple is (valence, arousal) at full intensity
emotion_to_va = {
    "H": ( 0.8,  0.6),   # Happy:   positive valence, moderate-high arousal
    "S": (-0.6,  0.2),   # Sad:     negative valence, low arousal
    "A": (-0.7,  0.8),   # Anger:   negative valence, high arousal
    "F": (-0.6,  0.7),   # Fear:    negative valence, high arousal
    "D": (-0.7,  0.4),   # Disgust: negative valence, moderate arousal
    "N": ( 0.1,  0.1),   # Neutral: near-zero on both axes
}

# Process MP4 files
mp4_files = [f for f in os.listdir(mp4_folder) if f.endswith(".mp4")]
mapping_list = []

for f in mp4_files:
    fname_no_ext = f.replace(".mp4", "")

    # Find matching row in tabulatedVotes
    row = votes[votes['fileName'] == fname_no_ext]

    if not row.empty:
        emo_code = row['emoVote'].values[0]     # Majority emotion label
        level    = row['meanEmoResp'].values[0]  # Intensity rating (0–100)

        # Normalize intensity to [0, 1]
        intensity = float(level) / 100.0
        intensity = max(0.0, min(1.0, intensity))

        # Look up base (valence, arousal) for this emotion
        base_valence, base_arousal = emotion_to_va.get(emo_code, (0.0, 0.0))

        # Scale both dimensions by intensity so low-intensity clips
        # are pulled toward neutral rather than pinned at their full value.
        # Formula: base * (0.5 + 0.5 * intensity)
        #   - at intensity=0.0  → 50 % of base value
        #   - at intensity=1.0  → 100 % of base value
        valence = base_valence * (0.5 + 0.5 * intensity)
        arousal = base_arousal * (0.5 + 0.5 * intensity)

        # Clamp to [-1, 1] and [0, 1] respectively
        valence = max(-1.0, min(1.0, valence))
        arousal = max( 0.0, min(1.0, arousal))

        mapping_list.append({
            "fileName": f,
            "emotion":  emo_code,
            "level":    level,
            "valence":  round(valence, 4),
            "arousal":  round(arousal, 4),
        })
    else:
        print(f"Warning: {fname_no_ext} not found in tabulatedVotes.csv")

# Save to CSV
mapping_df = pd.DataFrame(mapping_list)
mapping_df.to_csv(output_csv, index=False)
print(f"Dynamic ground-truth mapping saved to {output_csv}")
print(f"Total clips mapped: {len(mapping_df)}")
print(mapping_df[['emotion', 'valence', 'arousal']].groupby('emotion').describe().round(3))
