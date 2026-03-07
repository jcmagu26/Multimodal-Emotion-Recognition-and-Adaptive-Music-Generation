import pandas as pd
import os

# Paths
crema_csv = "/Users/janemaguire/Desktop/Final/CREMA-D/processedResults/tabulatedVotes.csv"
mp4_folder = "/Users/janemaguire/Desktop/Final/CREMA-D/MP4"
output_csv = "/Users/janemaguire/Desktop/Final/CREMA-D/video_va_mapping_dynamic.csv"

# Load tabulated votes
votes = pd.read_csv(crema_csv)

# Define valence mapping for emotions
# Arousal will be dynamic based on meanEmoResp
emotion_to_valence = {
    "H": 1.0,   # Happy
    "S": -1.0,  # Sad
    "A": -1.0,  # Anger
    "F": -0.8,  # Fear
    "D": -0.9,  # Disgust
    "N": 0.0,   # Neutral
}

# Process MP4 files
mp4_files = [f for f in os.listdir(mp4_folder) if f.endswith(".mp4")]
mapping_list = []

for f in mp4_files:
    # Remove .mp4 extension to match tabulatedVotes
    fname_no_ext = f.replace(".mp4", "")
    
    # Find row in tabulatedVotes
    row = votes[votes['fileName'] == fname_no_ext]
    
    if not row.empty:
        emo_code = row['emoVote'].values[0] # Majority emotion
        level = row['meanEmoResp'].values[0] # Intensity 0-100
        
        # Lookup valence
        valence = emotion_to_valence.get(emo_code, 0.0)
        
        # Scale arousal dynamically 0-1
        # If level > 100 in some rows, normalize by dividing by 100
        arousal = float(level) / 100.0
        arousal = max(0.0, min(1.0, arousal))  # ensure within [0,1]
        
        mapping_list.append({
            "fileName": f,
            "emotion": emo_code,
            "level": level,
            "valence": valence,
            "arousal": arousal
        })
    else:
        print(f"Warning: {fname_no_ext} not found in tabulatedVotes.csv")

# Save to CSV
mapping_df = pd.DataFrame(mapping_list)
mapping_df.to_csv(output_csv, index=False)
print(f"Dynamic ground-truth mapping saved to {output_csv}")