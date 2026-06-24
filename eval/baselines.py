"""
MethCLR — Baseline Comparisons (Cross-Cohort, Train A → Test B)

All baselines trained on Cohort A (GSE85210, N=253), tested on Cohort B
(GSE224807, N=789) — identical setup to MethCLR. Fair comparison.

Baselines:
  1. Raw Beta + LR       — fit LR on Cohort A beta values, predict on Cohort B
  2. PCA + LR            — fit PCA on Cohort A, transform B, fit LR on A, predict B
  3. Autoencoder + LR    — train AE on Cohort A (unsupervised), encode both cohorts,
                           fit LR on A embeddings, predict B embeddings

Feature selection: top 20,000 probes by inter-sample variance on Cohort A mean,
applied consistently to both cohorts.
"""

import os, sys, gzip
import numpy as np
import pandas as pd
import json

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

_HERE         = os.path.dirname(os.path.abspath(__file__))
_ROOT         = os.path.join(_HERE, "..")
MATRIX_DIR_A  = os.path.join(_ROOT, "beta_matrices")
MATRIX_DIR_B  = os.path.join(_ROOT, "cohort_b_beta_matrices")
SERIES_MAT_A  = os.path.join(os.path.expanduser("~"), "Downloads",
                              "GSE85210_series_matrix.txt.gz")
SERIES_MAT_B  = os.path.join(os.path.expanduser("~"), "Downloads",
                              "GSE224807-GPL13534_series_matrix.txt.gz")
OUT_DIR       = _HERE
TOP_PROBES    = 20000
EMBED_DIM     = 128
SEED          = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

METHCLR_AUC   = 0.7764
METHCLR_STD   = 0.0128

# ─────────────────────────────────────────────
# LABEL LOADERS
# ─────────────────────────────────────────────

def load_labels_a(path):
    """GSE85210 — links via Sentrix IDs from supplementary URLs."""
    samples, labels, supp_files = [], [], []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("!Sample_geo_accession"):
                samples = [p.strip('"') for p in line.split("\t")[1:]]
            if line.startswith("!Sample_characteristics_ch1") and "subject status" in line:
                labels = [p.strip('"') for p in line.split("\t")[1:]]
            if line.startswith("!Sample_supplementary_file"):
                supp_files.append([p.strip('"') for p in line.split("\t")[1:]])
    smoking = [0 if "non" in lb.lower() else 1 for lb in labels]
    grn_files = []
    for row in supp_files:
        matches = [x for x in row if "Grn.idat" in x]
        if matches:
            grn_files = matches
            break
    if len(grn_files) != len(samples):
        return {}
    label_map = {}
    for url, sm in zip(grn_files, smoking):
        fname = url.split("/")[-1]
        parts = fname.split("_")
        if len(parts) >= 3:
            label_map[f"{parts[1]}_{parts[2]}"] = sm
    return label_map

def load_labels_b(path):
    """GSE224807 — smoking_status field."""
    samples, smoking, supp_files = [], [], []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("!Sample_geo_accession"):
                samples = [p.strip('"') for p in line.split("\t")[1:]]
            if line.startswith("!Sample_characteristics_ch1") and "smoking_status" in line:
                smoking = [p.strip('"') for p in line.split("\t")[1:]]
            if line.startswith("!Sample_supplementary_file"):
                supp_files.append([p.strip('"') for p in line.split("\t")[1:]])
    grn_files = []
    for row in supp_files:
        matches = [x for x in row if "Grn.idat" in x]
        if matches:
            grn_files = matches
            break
    label_map = {}
    for url, status in zip(grn_files, smoking):
        fname = url.split("/")[-1]
        parts = fname.split("_")
        if len(parts) < 3:
            continue
        key = f"{parts[1]}_{parts[2]}"
        s = status.lower()
        if "sm" in s and "ns" not in s:
            label_map[key] = 1
        elif "ns" in s:
            label_map[key] = 0
    return label_map

# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────

def load_cohort(matrix_dir, label_map):
    files = sorted([f for f in os.listdir(matrix_dir) if f.endswith("_intersected.csv.gz")])
    matrices = {}
    for fname in files:
        pid = fname.replace("_intersected.csv.gz", "")
        df = pd.read_csv(os.path.join(matrix_dir, fname), index_col=0, compression="gzip")
        matrices[pid] = df
    pipeline_ids   = list(matrices.keys())
    all_cols       = [set(df.columns) for df in matrices.values()]
    shared_samples = sorted(set.intersection(*all_cols))
    data = np.stack([
        matrices[pid][shared_samples].T.values.astype(np.float32)
        for pid in pipeline_ids
    ], axis=0)
    data = np.nan_to_num(data, nan=0.5)
    labels_arr = np.array([label_map.get(sid, -1) for sid in shared_samples], dtype=np.int64)
    return data, labels_arr, shared_samples, pipeline_ids

# ─────────────────────────────────────────────
# LOAD COHORT A + B
# ─────────────────────────────────────────────

print("Loading Cohort A (GSE85210) ...")
label_map_a = load_labels_a(SERIES_MAT_A)
data_a, labels_a, samples_a, pids_a = load_cohort(MATRIX_DIR_A, label_map_a)
print(f"  Shape: {data_a.shape} | Labeled: {(labels_a>=0).sum()}")

print("\nLoading Cohort B (GSE224807) ...")
label_map_b = load_labels_b(SERIES_MAT_B)
data_b, labels_b, samples_b, pids_b = load_cohort(MATRIX_DIR_B, label_map_b)
print(f"  Shape: {data_b.shape} | Labeled: {(labels_b>=0).sum()}")

# ─────────────────────────────────────────────
# FEATURE SELECTION — fit on Cohort A, apply to both
# ─────────────────────────────────────────────

print(f"\nFeature selection: top {TOP_PROBES:,} probes by Cohort A inter-sample variance ...")
mean_a     = data_a.mean(axis=0)          # (n_samples_a, n_probes)
probe_var  = mean_a.var(axis=0)           # (n_probes,)
top_idx    = np.sort(np.argsort(probe_var)[-TOP_PROBES:])

X_a_full   = data_a[:, :, top_idx].mean(axis=0)   # mean across pipelines (n_a, 20k)
X_b_full   = data_b[:, :, top_idx].mean(axis=0)   # (n_b, 20k)

mask_a     = labels_a >= 0
mask_b     = labels_b >= 0
X_a        = X_a_full[mask_a]
y_a        = labels_a[mask_a]
X_b        = X_b_full[mask_b]
y_b        = labels_b[mask_b]

print(f"Cohort A: {X_a.shape} | Smokers: {y_a.sum()}, Non-smokers: {(y_a==0).sum()}")
print(f"Cohort B: {X_b.shape} | Smokers: {y_b.sum()}, Non-smokers: {(y_b==0).sum()}")

scaler = StandardScaler().fit(X_a)
X_a_s  = scaler.transform(X_a)
X_b_s  = scaler.transform(X_b)

clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
results = {}

# ─────────────────────────────────────────────
# BASELINE 1 — Raw Beta + LR (Train A → Test B)
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("Baseline 1: Raw Beta + LR  (Train A → Test B)")
print("=" * 55)

clf.fit(X_a_s, y_a)
prob_b    = clf.predict_proba(X_b_s)[:, 1]
auc_raw   = float(roc_auc_score(y_b, prob_b))
print(f"Cohort B AUC: {auc_raw:.4f}")
results["raw_beta_lr"] = {"auc": auc_raw, "train": "CohortA", "test": "CohortB"}

# ─────────────────────────────────────────────
# BASELINE 2 — PCA + LR (Train A → Test B)
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print(f"Baseline 2: PCA ({EMBED_DIM} PCs) + LR  (Train A → Test B)")
print("=" * 55)

pca      = PCA(n_components=EMBED_DIM, random_state=SEED).fit(X_a_s)
X_a_pca  = pca.transform(X_a_s)
X_b_pca  = pca.transform(X_b_s)
var_exp  = pca.explained_variance_ratio_.sum()
print(f"Variance explained (Cohort A): {var_exp:.3f}")

clf.fit(X_a_pca, y_a)
prob_b_pca = clf.predict_proba(X_b_pca)[:, 1]
auc_pca    = float(roc_auc_score(y_b, prob_b_pca))
print(f"Cohort B AUC: {auc_pca:.4f}")
results["pca_lr"] = {"auc": auc_pca, "variance_explained": float(var_exp),
                     "train": "CohortA", "test": "CohortB"}

# ─────────────────────────────────────────────
# BASELINE 3 — Autoencoder + LR (Train A → Test B)
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print(f"Baseline 3: Autoencoder ({EMBED_DIM}-dim) + LR  (Train A → Test B)")
print("=" * 55)

class MethAE(nn.Module):
    def __init__(self, input_dim, bottleneck=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 256), nn.ReLU(),
            nn.Linear(256, 512),        nn.ReLU(),
            nn.Linear(512, input_dim),  nn.Sigmoid(),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_a_t     = torch.from_numpy(X_a_s.astype(np.float32))
loader_ae = DataLoader(TensorDataset(X_a_t), batch_size=32, shuffle=True)

ae  = MethAE(input_dim=TOP_PROBES, bottleneck=EMBED_DIM).to(device)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-4)
mse = nn.MSELoss()

AE_EPOCHS = 50
print(f"Training AE on Cohort A ({AE_EPOCHS} epochs, {device}) ...")
for epoch in range(1, AE_EPOCHS + 1):
    ae.train()
    losses = []
    for (xb,) in loader_ae:
        xb = xb.to(device)
        out, _ = ae(xb)
        loss = mse(out, xb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}/{AE_EPOCHS} | recon_loss={np.mean(losses):.4f}")

ae.eval()
with torch.no_grad():
    _, X_a_ae = ae(X_a_t.to(device))
    _, X_b_ae = ae(torch.from_numpy(X_b_s.astype(np.float32)).to(device))
X_a_ae = X_a_ae.cpu().numpy()
X_b_ae = X_b_ae.cpu().numpy()

clf.fit(X_a_ae, y_a)
prob_b_ae = clf.predict_proba(X_b_ae)[:, 1]
auc_ae    = float(roc_auc_score(y_b, prob_b_ae))
print(f"Cohort B AUC: {auc_ae:.4f}")
results["autoencoder_lr"] = {"auc": auc_ae, "train": "CohortA", "test": "CohortB"}

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("COMPARISON TABLE — Cross-Cohort (Train: GSE85210 → Test: GSE224807)")
print("=" * 60)
print(f"{'Method':<35} {'Cohort B AUC':>12}  {'Supervision':>15}")
print("-" * 65)
print(f"{'Raw Beta + LR':<35} {auc_raw:>12.4f}  {'supervised':>15}")
print(f"{'PCA (128) + LR':<35} {auc_pca:>12.4f}  {'supervised':>15}")
print(f"{'Autoencoder (128) + LR':<35} {auc_ae:>12.4f}  {'unsupervised':>15}")
print(f"{'MethCLR (ours)':<35} {METHCLR_AUC:>12.4f}  {'unsupervised':>15}")
print("=" * 60)
print("All methods: trained on Cohort A only, evaluated on Cohort B.")

results["methclr"] = {"auc": METHCLR_AUC, "std": METHCLR_STD,
                      "train": "CohortA", "test": "CohortB"}

out_path = os.path.join(OUT_DIR, "baseline_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out_path}")
