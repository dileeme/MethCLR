"""
MethCLR — Training Script (Google Colab)

Run this on Colab after mounting Google Drive with the following layout:
    /MyDrive/MethCLR/
        beta_matrices/          ← 12 *_intersected.csv.gz files
        GSE85210_series_matrix.txt.gz

Steps:
    1. Mount Google Drive (Colab cell)
    2. Install dependencies (Colab cell)
    3. Run this script

Output saved to /MyDrive/MethCLR/checkpoints/
"""

# ── COLAB SETUP ──────────────────────────────────────────────────────────────
# Run these in separate Colab cells before executing this script:
#
# Cell 1 — Mount Drive:
#   from google.colab import drive
#   drive.mount('/content/drive')
#
# Cell 2 — Install:
#   !pip install torch scikit-learn umap-learn matplotlib
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Add model/ directory to path if running from Colab root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import load_intersected_matrices, load_smoking_labels, MethCLRDataset, get_labeled_embeddings
from encoder import MLPEncoder
from loss import InfoNCELoss

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _linear_probe_auc(embeddings: np.ndarray, labels: np.ndarray,
                      seed: int = 42) -> float:
    """
    Fits a logistic regression on embeddings and returns AUC.
    Uses all labeled samples — 5-fold cross-validation.
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    clf = LogisticRegression(max_iter=500, random_state=seed, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, labels, cv=cv, scoring="roc_auc")
    return float(scores.mean())

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DRIVE_ROOT   = "/content/colab_dump"
MATRIX_DIR   = os.path.join(DRIVE_ROOT, "beta_matrices")
SERIES_MAT   = os.path.join(DRIVE_ROOT, "GSE85210_series_matrix.txt.gz")
CKPT_DIR     = os.path.join(DRIVE_ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# Training hyperparameters
EPOCHS         = 150
BATCH_SIZE     = 64
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
TEMPERATURE    = 0.07
DROPOUT        = 0.1
EVAL_EVERY     = 10       # linear probe AUC evaluation interval (epochs)
PATIENCE       = 20       # early stopping patience (in eval intervals)
TRAIN_SPLIT    = 0.85     # fraction of samples used for training

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Loading data")
print("=" * 60)

matrices  = load_intersected_matrices(MATRIX_DIR)
label_map = load_smoking_labels(SERIES_MAT)
dataset   = MethCLRDataset(matrices, label_map)

input_dim = dataset.input_dim
print(f"\nEncoder input dimension: {input_dim:,}")

# Train / val split (stratified on labeled samples where possible)
n_total = len(dataset)
n_train = int(n_total * TRAIN_SPLIT)
n_val   = n_total - n_train
train_set, val_set = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=(device.type == "cuda"))
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=(device.type == "cuda"))

print(f"Train samples: {n_train} | Val samples: {n_val}")

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

encoder   = MLPEncoder(input_dim=input_dim, dropout=DROPOUT).to(device)
criterion = InfoNCELoss(temperature=TEMPERATURE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print(f"Encoder parameters: {total_params:,}")

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"Training — {EPOCHS} epochs | batch={BATCH_SIZE} | τ={TEMPERATURE} | lr={LR}")
print("=" * 60)

history = {"train_loss": [], "val_loss": [], "linear_probe_auc": [], "eval_epochs": []}
best_auc   = 0.0
best_epoch = 0
patience_counter = 0
global_start = time.time()

for epoch in range(1, EPOCHS + 1):
    # ── Train ──────────────────────────────────────────────────────
    encoder.train()
    train_losses = []
    for anchor, positive, _ in train_loader:
        anchor   = anchor.to(device)
        positive = positive.to(device)

        _, z_i = encoder(anchor)
        _, z_j = encoder(positive)
        loss = criterion(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    # ── Val ────────────────────────────────────────────────────────
    encoder.eval()
    val_losses = []
    with torch.no_grad():
        for anchor, positive, _ in val_loader:
            anchor   = anchor.to(device)
            positive = positive.to(device)
            _, z_i = encoder(anchor)
            _, z_j = encoder(positive)
            val_losses.append(criterion(z_i, z_j).item())

    scheduler.step()

    train_loss = np.mean(train_losses)
    val_loss   = np.mean(val_losses)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}", end="")

    # ── Linear probe evaluation ────────────────────────────────────
    if epoch % EVAL_EVERY == 0:
        embeddings, labels = get_labeled_embeddings(dataset, encoder, device)
        auc = _linear_probe_auc(embeddings, labels, seed=SEED)
        history["linear_probe_auc"].append(auc)
        history["eval_epochs"].append(epoch)
        print(f" | linear_probe_AUC={auc:.4f}", end="")

        if auc > best_auc:
            best_auc   = auc
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = os.path.join(CKPT_DIR, "methclr_best.pt")
            torch.save({
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "auc": best_auc,
                "input_dim": input_dim,
            }, ckpt_path)
            print(f" ← best", end="")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[STOP] Early stopping at epoch {epoch} "
                      f"(best AUC {best_auc:.4f} at epoch {best_epoch})")
                break

    print()

elapsed = time.time() - global_start
print(f"\nTraining complete in {elapsed/60:.1f} min")
print(f"Best linear probe AUC: {best_auc:.4f} at epoch {best_epoch}")

# ─────────────────────────────────────────────
# SAVE FINAL CHECKPOINT
# ─────────────────────────────────────────────

torch.save({
    "epoch": epoch,
    "encoder_state": encoder.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "auc": best_auc,
    "input_dim": input_dim,
    "history": history,
}, os.path.join(CKPT_DIR, "methclr_final.pt"))

# ─────────────────────────────────────────────
# TRAINING PLOTS
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="Train loss")
axes[0].plot(history["val_loss"],   label="Val loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("InfoNCE Loss")
axes[0].set_title("Contrastive Loss")
axes[0].legend()

if history["linear_probe_auc"]:
    axes[1].plot(history["eval_epochs"], history["linear_probe_auc"], marker="o")
    axes[1].axhline(best_auc, linestyle="--", color="red",
                    label=f"Best AUC = {best_auc:.4f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC (linear probe)")
    axes[1].set_title("Smoking Classification (Linear Probe)")
    axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(CKPT_DIR, "training_curves.png"), dpi=150)
plt.show()
print(f"Plots saved → {CKPT_DIR}/training_curves.png")
