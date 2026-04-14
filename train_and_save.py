"""
train_and_save.py
──────────────────
Trains a multi-output MLP to jointly predict valence and arousal.

Why joint MLP instead of two GradientBoosting models?
  Valence and arousal are correlated. A shared hidden layer lets the model
  learn that "high energy + tonal voice = happy" vs "high energy + harsh
  voice = angry" — a distinction requiring both dimensions simultaneously.

Architecture:
  Input → BatchNorm → FC(256,ReLU,Drop) → FC(128,ReLU,Drop) → FC(64,ReLU)
       → valence head (linear)
       → arousal head (sigmoid → [0,1])

Run:
  python train_and_save.py
  python train_and_save.py /path/to/features.csv
"""

import os, sys, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR         = "/Users/janemaguire/Desktop/Final/CREMA-D"
WITH_TEXT_CSV    = os.path.join(BASE_DIR, "video_features_with_text.csv")
WITHOUT_TEXT_CSV = os.path.join(BASE_DIR, "video_features.csv")
MODEL_DIR        = os.path.join(os.path.dirname(__file__), "models")

EPOCHS = 150; BATCH_SIZE = 64; LR = 3e-4; HIDDEN = [256, 128, 64]
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

device = torch.device("mps"  if torch.backends.mps.is_available()  else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data ─────────────────────────────────────────────────────────────────

if len(sys.argv) > 1:
    features_csv = sys.argv[1]
elif os.path.exists(WITH_TEXT_CSV):
    features_csv = WITH_TEXT_CSV
    print("Using video_features_with_text.csv")
else:
    features_csv = WITHOUT_TEXT_CSV
    print("Using audio + visual features only (run transcribe_features.py for text features)")

df = pd.read_csv(features_csv)
print(f"Shape: {df.shape}")
print(f"Valence [{df['valence'].min():.3f}, {df['valence'].max():.3f}]  "
      f"Arousal [{df['arousal'].min():.3f}, {df['arousal'].max():.3f}]")

# ── Feature columns ───────────────────────────────────────────────────────────

audio_cols  = [c for c in df.columns if any(c.startswith(p) for p in
               ('mfcc_','pitch_','voiced_','energy_','spec_','zcr',
                'hnr','chroma_','mel_','jitter'))]
visual_cols = [c for c in df.columns if c.startswith('face_')]
text_cols   = [c for c in df.columns if c.startswith('text_')]
all_cols    = audio_cols + visual_cols + text_cols

print(f"Audio: {len(audio_cols)}  Visual: {len(visual_cols)}  "
      f"Text: {len(text_cols)}  Total: {len(all_cols)}")

# ── Oversample positive/neutral valence ───────────────────────────────────────

positive = df[df['valence'] > 0.2]
neutral  = df[(df['valence'] >= -0.1) & (df['valence'] <= 0.2)]
df = pd.concat([df, positive, positive, positive, neutral, neutral],
               ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"After resampling: {len(df)} clips")

X = df[all_cols].values.astype(np.float32)
Y = df[['valence', 'arousal']].values.astype(np.float32)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_SEED)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Model ─────────────────────────────────────────────────────────────────────

class VAPredictor(nn.Module):
    def __init__(self, input_dim, hidden, dropout=0.3):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(),
                       nn.Dropout(max(dropout, 0.1))]
            in_dim = h; dropout -= 0.1
        self.shared       = nn.Sequential(*layers)
        self.valence_head = nn.Linear(in_dim, 1)
        self.arousal_head = nn.Linear(in_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        return torch.cat([self.valence_head(h),
                          torch.sigmoid(self.arousal_head(h))], dim=1)

# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(X_tr, Y_tr, X_va, Y_va):
    model     = VAPredictor(X_tr.shape[1], HIDDEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 1e-5)
    criterion = nn.MSELoss()

    dl      = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)),
                         BATCH_SIZE, shuffle=True)
    X_va_t  = torch.tensor(X_va).to(device)
    Y_va_t  = torch.tensor(Y_va).to(device)

    best_loss, best_state, patience, no_imp = 1e9, None, 20, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_va_t), Y_va_t).item()

        if vl < best_loss:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1

        if epoch % 25 == 0:
            print(f"  epoch {epoch:3d}  val_loss={vl:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}")
        if no_imp >= patience:
            print(f"  Early stop at epoch {epoch}"); break

    model.load_state_dict(best_state)
    return model

# ── Cross-validation ──────────────────────────────────────────────────────────

print("\n── 5-fold cross-validation ──────────────────────────────────────")
kf = KFold(5, shuffle=True, random_state=RANDOM_SEED)
val_r, aro_r = [], []

for fold, (ti, vi) in enumerate(kf.split(X_train), 1):
    m = train_model(X_train[ti], Y_train[ti], X_train[vi], Y_train[vi])
    m.eval()
    with torch.no_grad():
        p = m(torch.tensor(X_train[vi]).to(device)).cpu().numpy()
    vr = np.sqrt(mean_squared_error(Y_train[vi, 0], p[:, 0]))
    ar = np.sqrt(mean_squared_error(Y_train[vi, 1], p[:, 1]))
    val_r.append(vr); aro_r.append(ar)
    print(f"  Fold {fold}  valence={vr:.4f}  arousal={ar:.4f}")

print(f"\n  Valence  {np.mean(val_r):.4f} ± {np.std(val_r):.4f}")
print(f"  Arousal  {np.mean(aro_r):.4f} ± {np.std(aro_r):.4f}")

# ── Final model ───────────────────────────────────────────────────────────────

print("\n── Training final model ─────────────────────────────────────────")
final = train_model(X_train, Y_train, X_test, Y_test)
final.eval()
with torch.no_grad():
    tp = final(torch.tensor(X_test).to(device)).cpu().numpy()

print(f"\n  Test valence RMSE: {np.sqrt(mean_squared_error(Y_test[:,0], tp[:,0])):.4f}")
print(f"  Test arousal RMSE: {np.sqrt(mean_squared_error(Y_test[:,1], tp[:,1])):.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(final.state_dict(), os.path.join(MODEL_DIR, "va_mlp.pt"))
joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(all_cols, os.path.join(MODEL_DIR, "feature_cols.joblib"))
joblib.dump({"input_dim": X_train.shape[1], "hidden": HIDDEN, "dropout": 0.0},
            os.path.join(MODEL_DIR, "mlp_config.joblib"))

print(f"\n✓ Saved to {MODEL_DIR}/")
print("  va_mlp.pt  scaler.joblib  feature_cols.joblib  mlp_config.joblib")
print("\nRun: python app.py")
