"""
train_and_save.py  (v2 — anti-collapse edition)
─────────────────────────────────────────────────
The original model mean-collapsed: raw_valence ≈ +0.04 and raw_arousal ≈ 0.08–0.35
for every clip regardless of emotion. This version fixes that with:

  1. Joint VA regression + 6-class classification loss.
     Cross-entropy on the emotion label gives a direct high-signal gradient
     that cannot be minimised by predicting the mean.

  2. Stratified split by emotion label — all 6 classes in train and test.

  3. Label-aware oversampling using the actual 6 CREMA-D categories.

  4. Higher valence loss weight (was under-trained vs arousal).

  5. Emotion label read directly from filename — ground truth, not derived.

Architecture:
  Input → BN → FC(256,ReLU,Drop0.3) → FC(128,ReLU,Drop0.2) → FC(64,ReLU,Drop0.1)
       → valence head   (linear)
       → arousal head   (sigmoid)
       → emotion head   (6-class cross-entropy)

Loss = MSE(valence)*2.0 + MSE(arousal)*1.5 + CrossEntropy(emotion)*1.0

app.py also needs updating to use the emotion head for labelling.
Run:
  python train_and_save.py
  python train_and_save.py /path/to/features.csv
"""

import os, sys, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR         = "/Users/janemaguire/Desktop/Final/CREMA-D"
WITH_TEXT_CSV    = os.path.join(BASE_DIR, "video_features_with_text.csv")
WITHOUT_TEXT_CSV = os.path.join(BASE_DIR, "video_features.csv")
MODEL_DIR        = os.path.join(os.path.dirname(__file__), "models")

EPOCHS      = 400
BATCH_SIZE  = 64
LR          = 2e-4
HIDDEN      = [256, 128, 64]
RANDOM_SEED = 42
VALENCE_W   = 2.0
AROUSAL_W   = 1.5
CLASS_W     = 1.0

EMOTION_CLASSES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

device = torch.device("mps"  if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data ─────────────────────────────────────────────────────────────────

features_csv = sys.argv[1] if len(sys.argv) > 1 else (
    WITH_TEXT_CSV if os.path.exists(WITH_TEXT_CSV) else WITHOUT_TEXT_CSV)
print(f"Loading: {features_csv}")

df = pd.read_csv(features_csv)
print(f"Shape: {df.shape}")

# ── Extract emotion label from filename ───────────────────────────────────────
# CREMA-D: {actorID}_{sentence}_{emotion}_{level}.mp4  e.g. 1001_IEO_ANG_HI.mp4

def extract_emotion(fname):
    parts = str(fname).replace(".mp4", "").split("_")
    return parts[2].upper() if len(parts) >= 3 else "NEU"

df["emotion_code"] = df["fileName"].apply(extract_emotion)
df = df[df["emotion_code"].isin(EMOTION_CLASSES)].reset_index(drop=True)

le = LabelEncoder()
le.fit(EMOTION_CLASSES)
df["emotion_idx"] = le.transform(df["emotion_code"])

print(f"Emotion distribution:\n{df['emotion_code'].value_counts().to_string()}")
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

# ── Stratified split ──────────────────────────────────────────────────────────

X_raw = df[all_cols].values.astype(np.float32)
Y_va  = df[['valence', 'arousal']].values.astype(np.float32)
Y_cls = df['emotion_idx'].values.astype(np.int64)

X_train_raw, X_test, Y_va_tr_raw, Y_va_test, Y_cls_tr_raw, Y_cls_test = \
    train_test_split(X_raw, Y_va, Y_cls, test_size=0.2,
                     stratify=Y_cls, random_state=RANDOM_SEED)

print(f"Train: {len(X_train_raw)}  Test: {len(X_test)}")

# ── Label-aware oversampling on training set only ────────────────────────────

train_df = pd.DataFrame(X_train_raw, columns=all_cols)
train_df['valence']     = Y_va_tr_raw[:, 0]
train_df['arousal']     = Y_va_tr_raw[:, 1]
train_df['emotion_idx'] = Y_cls_tr_raw

counts   = train_df['emotion_idx'].value_counts()
target_n = int(counts.max() * 1.5)

parts = []
for cls_idx in range(len(EMOTION_CLASSES)):
    subset = train_df[train_df['emotion_idx'] == cls_idx]
    if len(subset) == 0:
        continue
    n_needed = max(0, target_n - len(subset))
    if n_needed > 0:
        extra = subset.sample(n=n_needed, replace=True, random_state=RANDOM_SEED)
        parts.append(pd.concat([subset, extra], ignore_index=True))
    else:
        parts.append(subset)

train_df = pd.concat(parts, ignore_index=True).sample(
    frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"After oversampling: {len(train_df)} clips")
print(f"Class distribution: {dict(train_df['emotion_idx'].value_counts().sort_index())}")

X_train     = train_df[all_cols].values.astype(np.float32)
Y_va_train  = train_df[['valence', 'arousal']].values.astype(np.float32)
Y_cls_train = train_df['emotion_idx'].values.astype(np.int64)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Model ─────────────────────────────────────────────────────────────────────

class VAPredictor(nn.Module):
    def __init__(self, input_dim, hidden, n_classes=6, dropout=0.3):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        drop   = dropout
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(max(drop, 0.1))]
            in_dim = h
            drop  -= 0.1
        self.shared       = nn.Sequential(*layers)
        self.valence_head = nn.Linear(in_dim, 1)
        self.arousal_head = nn.Linear(in_dim, 1)
        self.emotion_head = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        h = self.shared(x)
        v = self.valence_head(h)
        a = torch.sigmoid(self.arousal_head(h))
        e = self.emotion_head(h)
        return torch.cat([v, a], dim=1), e

# ── Training loop ─────────────────────────────────────────────────────────────

ce_loss      = nn.CrossEntropyLoss()
loss_weights = torch.tensor([VALENCE_W, AROUSAL_W], device=device)

def combined_loss(va_pred, em_logits, va_target, cls_target):
    mse = ((va_pred - va_target) ** 2 * loss_weights).mean()
    ce  = ce_loss(em_logits, cls_target) * CLASS_W
    return mse + ce

def train_model(X_tr, Y_va_tr, Y_cls_tr, X_va, Y_va_va, Y_cls_va):
    model     = VAPredictor(X_tr.shape[1], HIDDEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 1e-5)

    ds = TensorDataset(torch.tensor(X_tr),
                       torch.tensor(Y_va_tr),
                       torch.tensor(Y_cls_tr))
    dl = DataLoader(ds, BATCH_SIZE, shuffle=True)

    X_va_t  = torch.tensor(X_va).to(device)
    Y_va_t  = torch.tensor(Y_va_va).to(device)
    Y_cls_t = torch.tensor(Y_cls_va).to(device)

    # Larger dataset needs more patience — final model trains on 6k+ clips
    patience = 60 if len(X_tr) > 3000 else 40
    best_loss, best_state, no_imp = 1e9, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb_va, yb_cls in dl:
            xb     = xb.to(device)
            yb_va  = yb_va.to(device)
            yb_cls = yb_cls.to(device)
            optimizer.zero_grad()
            va_pred, em_logits = model(xb)
            loss = combined_loss(va_pred, em_logits, yb_va, yb_cls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_p, em_p = model(X_va_t)
            vl  = combined_loss(va_p, em_p, Y_va_t, Y_cls_t).item()
            acc = (em_p.argmax(1) == Y_cls_t).float().mean().item()

        if vl < best_loss:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1

        if epoch % 50 == 0:
            with torch.no_grad():
                v_std = va_p[:, 0].std().item()
                a_std = va_p[:, 1].std().item()
            print(f"  epoch {epoch:3d}  loss={vl:.4f}  cls_acc={acc:.3f}  "
                  f"v_std={v_std:.3f}  a_std={a_std:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}")
        if no_imp >= patience:
            print(f"  Early stop at epoch {epoch}"); break

    model.load_state_dict(best_state)
    return model

# ── Cross-validation ──────────────────────────────────────────────────────────

print("\n── 5-fold stratified cross-validation ──────────────────────────")
skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
val_r, aro_r, acc_r = [], [], []

for fold, (ti, vi) in enumerate(skf.split(X_train, Y_cls_train), 1):
    m = train_model(X_train[ti], Y_va_train[ti], Y_cls_train[ti],
                    X_train[vi], Y_va_train[vi], Y_cls_train[vi])
    m.eval()
    with torch.no_grad():
        va_p, em_p = m(torch.tensor(X_train[vi]).to(device))
        p   = va_p.cpu().numpy()
        acc = (em_p.argmax(1).cpu().numpy() == Y_cls_train[vi]).mean()
    vr = np.sqrt(mean_squared_error(Y_va_train[vi, 0], p[:, 0]))
    ar = np.sqrt(mean_squared_error(Y_va_train[vi, 1], p[:, 1]))
    val_r.append(vr); aro_r.append(ar); acc_r.append(acc)
    print(f"  Fold {fold}  valence={vr:.4f}  arousal={ar:.4f}  cls_acc={acc:.3f}")

print(f"\n  Valence  {np.mean(val_r):.4f} ± {np.std(val_r):.4f}")
print(f"  Arousal  {np.mean(aro_r):.4f} ± {np.std(aro_r):.4f}")
print(f"  Cls Acc  {np.mean(acc_r):.4f} ± {np.std(acc_r):.4f}")

# ── Final model ───────────────────────────────────────────────────────────────

print("\n── Training final model ─────────────────────────────────────────")
final = train_model(X_train, Y_va_train, Y_cls_train, X_test, Y_va_test, Y_cls_test)
final.eval()

with torch.no_grad():
    va_tp, em_tp = final(torch.tensor(X_test).to(device))
    tp    = va_tp.cpu().numpy()
    em_np = em_tp.argmax(1).cpu().numpy()

print(f"\n  Test valence RMSE: {np.sqrt(mean_squared_error(Y_va_test[:,0], tp[:,0])):.4f}")
print(f"  Test arousal RMSE: {np.sqrt(mean_squared_error(Y_va_test[:,1], tp[:,1])):.4f}")
print(f"  Test cls accuracy: {(em_np == Y_cls_test).mean():.4f}")
print(f"  Valence pred range: [{tp[:,0].min():.3f}, {tp[:,0].max():.3f}]  std={tp[:,0].std():.3f}")
print(f"  Arousal pred range: [{tp[:,1].min():.3f}, {tp[:,1].max():.3f}]  std={tp[:,1].std():.3f}")
print(f"\n{classification_report(Y_cls_test, em_np, target_names=EMOTION_CLASSES)}")

# ── Save ──────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(final.state_dict(), os.path.join(MODEL_DIR, "va_mlp.pt"))
joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(all_cols, os.path.join(MODEL_DIR, "feature_cols.joblib"))
joblib.dump({"input_dim": X_train.shape[1], "hidden": HIDDEN,
             "dropout": 0.0, "n_classes": len(EMOTION_CLASSES)},
            os.path.join(MODEL_DIR, "mlp_config.joblib"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

print(f"\n✓ Saved to {MODEL_DIR}/")
print("  va_mlp.pt  scaler.joblib  feature_cols.joblib  mlp_config.joblib  label_encoder.joblib")
print("\nRun: python app.py")
