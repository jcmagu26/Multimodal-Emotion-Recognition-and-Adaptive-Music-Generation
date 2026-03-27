import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Load features
# ---------------------------------------------------------------------------
features_csv = "/Users/janemaguire/Desktop/Final/CREMA-D/video_features.csv"
df = pd.read_csv(features_csv)

print(f"Dataset shape: {df.shape}")
print(f"Valence range: [{df['valence'].min():.3f}, {df['valence'].max():.3f}]")
print(f"Arousal range: [{df['arousal'].min():.3f}, {df['arousal'].max():.3f}]")

# ---------------------------------------------------------------------------
# Define feature column groups for modality comparison
# ---------------------------------------------------------------------------
audio_cols = [c for c in df.columns if any(c.startswith(p) for p in
              ('mfcc_', 'pitch_', 'voiced_', 'energy_', 'spec_', 'zcr'))]

visual_cols = [c for c in df.columns if c.startswith('face_')]

all_cols = audio_cols + visual_cols   # multimodal

print(f"\nAudio features:    {len(audio_cols)}")
print(f"Visual features:   {len(visual_cols)}")
print(f"Multimodal total:  {len(all_cols)}")

# Labels
y_valence = df['valence']
y_arousal = df['arousal']

# ---------------------------------------------------------------------------
# Single, aligned train/test split shared by all experiments
# ---------------------------------------------------------------------------
X_full = df[all_cols]

X_train, X_test, y_val_train, y_val_test, y_aro_train, y_aro_test = train_test_split(
    X_full, y_valence, y_arousal,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------------------------------
# Model factory
# Returns a sklearn Pipeline with scaling + GradientBoosting
# ---------------------------------------------------------------------------
def make_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ))
    ])

# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------
def cv_rmse(pipeline, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=kf,
                             scoring='neg_root_mean_squared_error')
    return -scores.mean(), scores.std()

# ---------------------------------------------------------------------------
# Evaluate all three modality configurations
# ---------------------------------------------------------------------------
modalities = {
    "Audio-only":   df[audio_cols],
    "Visual-only":  df[visual_cols],
    "Multimodal":   df[all_cols],
}

print("\n" + "="*60)
print(f"{'Modality':<16} {'Target':<10} {'CV RMSE':>10} {'±':>8}")
print("="*60)

results = {}
for name, X in modalities.items():
    for target_name, y in [("Valence", y_valence), ("Arousal", y_arousal)]:
        mean_rmse, std_rmse = cv_rmse(make_pipeline(), X, y, cv=5)
        print(f"{name:<16} {target_name:<10} {mean_rmse:>10.4f} {std_rmse:>8.4f}")
        results[(name, target_name)] = mean_rmse

print("="*60)

# ---------------------------------------------------------------------------
# Train final multimodal models on the full training split
# and report held-out test RMSE
# ---------------------------------------------------------------------------
print("\n--- Held-out test set evaluation (multimodal) ---")

pipe_valence = make_pipeline()
pipe_valence.fit(X_train, y_val_train)
val_pred  = pipe_valence.predict(X_test)
val_rmse  = np.sqrt(mean_squared_error(y_val_test, val_pred))
print(f"Valence RMSE (test): {val_rmse:.4f}")

pipe_arousal = make_pipeline()
pipe_arousal.fit(X_train, y_aro_train)
aro_pred  = pipe_arousal.predict(X_test)
aro_rmse  = np.sqrt(mean_squared_error(y_aro_test, aro_pred))
print(f"Arousal RMSE (test): {aro_rmse:.4f}")

# ---------------------------------------------------------------------------
# Feature importance (multimodal valence model as example)
# ---------------------------------------------------------------------------
print("\n--- Top 15 features by importance (Valence model) ---")
feature_names = all_cols
importances   = pipe_valence.named_steps['model'].feature_importances_
importance_df = (
    pd.DataFrame({'feature': feature_names, 'importance': importances})
    .sort_values('importance', ascending=False)
    .head(15)
    .reset_index(drop=True)
)
print(importance_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Predict on all clips and save output
# ---------------------------------------------------------------------------
df['valence_pred'] = pipe_valence.predict(df[all_cols])
df['arousal_pred'] = pipe_arousal.predict(df[all_cols])

output_csv = "/Users/janemaguire/Desktop/Final/CREMA-D/video_va_predicted.csv"
df[['fileName', 'valence', 'arousal', 'valence_pred', 'arousal_pred']].to_csv(
    output_csv, index=False
)
print(f"\nPredicted valence/arousal saved to {output_csv}")

# ---------------------------------------------------------------------------
# Simple emotion-to-music parameter mapping
# ---------------------------------------------------------------------------
print("\n--- Emotion-Driven Music Parameters (sample predictions) ---")
print(f"{'fileName':<35} {'val_pred':>9} {'aro_pred':>9} {'key':>8} {'tempo':>7}")
print("-"*72)

for _, r in df.head(10).iterrows():
    v, a = r['valence_pred'], r['arousal_pred']
    key   = "Major" if v >= 0 else "Minor"
    tempo = int(60 + a * 120)          # maps [0,1] → [60, 180] BPM
    print(f"{r['fileName']:<35} {v:>9.3f} {a:>9.3f} {key:>8} {tempo:>7} BPM")
