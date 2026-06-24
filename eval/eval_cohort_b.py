"""
MethCLR — Cross-Cohort Evaluation (Cohort B: GSE224807)

Upload to Colab flat alongside encoder.py, dataset.py, methclr_best.pt,
and all 12 cohort_b *_intersected.csv.gz files.

Expected /content/ layout:
    methclr_best.pt
    encoder.py
    dataset.py
    GSE224807_series_matrix.txt.gz          ← download separately if needed
    norm=*_intersected.csv.gz               ← 12 Cohort B intersected matrices

Run:
    !python /content/eval_cohort_b.py
"""

import os, sys, gzip
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
sys.path.append("/content")
from encoder import MLPEncoder

# ─────────────────────────────────────────────
# CONFIGURATION — edit these paths if needed
# ─────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.join(_HERE, "..")
CONTENT_DIR  = "/content"   # Colab path (unused locally)
CKPT_PATH    = os.path.join(_ROOT, "results", "methclr_best.pt")
MATRIX_DIR   = os.path.join(_ROOT, "cohort_b_beta_matrices")
SERIES_MAT_B = os.path.join(os.path.expanduser("~"), "Downloads", "GSE224807-GPL13534_series_matrix.txt.gz")
OUT_DIR      = os.path.join(_ROOT, "eval")
os.makedirs(OUT_DIR, exist_ok=True)

TOP_PROBES   = 20000   # must match training run 3
SEED         = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─────────────────────────────────────────────
# STEP 1 — Load Cohort B intersected matrices
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Loading Cohort B matrices")
print("=" * 60)

files = sorted([
    f for f in os.listdir(MATRIX_DIR)
    if f.endswith("_intersected.csv.gz")
])
if not files:
    raise FileNotFoundError(f"No *_intersected.csv.gz found in {MATRIX_DIR}")

matrices = {}
for fname in files:
    pid = fname.replace("_intersected.csv.gz", "")
    fpath = os.path.join(MATRIX_DIR, fname)
    print(f"  Loading {fname} ...", end=" ", flush=True)
    df = pd.read_csv(fpath, index_col=0, compression="gzip")
    matrices[pid] = df
    print(df.shape)

pipeline_ids = list(matrices.keys())
print(f"\n[DATA] {len(matrices)} pipelines | "
      f"Probes: {next(iter(matrices.values())).shape[0]:,} | "
      f"Samples: {next(iter(matrices.values())).shape[1]}")

# ─────────────────────────────────────────────
# STEP 2 — Parse Cohort B smoking labels
# ─────────────────────────────────────────────

def load_cohort_b_labels(series_matrix_path: str) -> dict:
    """
    Parses GSE224807 series matrix.
    Smoking label field: 'smoking_status: SM' / 'smoking_status: NS'
    Keys matched to beta matrix columns via Sentrix IDs parsed from supplementary URLs.
    Returns dict: '{Sentrix_ID}_{Sentrix_Position}' → int (0=NS, 1=SM)
    """
    if not os.path.exists(series_matrix_path):
        print(f"[WARN] Series matrix not found: {series_matrix_path}")
        print("[WARN] Proceeding without labels — AUC will not be computed")
        return {}

    samples, smoking, supp_files = [], [], []
    opener = gzip.open if series_matrix_path.endswith(".gz") else open
    try:
        with opener(series_matrix_path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("!Sample_geo_accession"):
                    samples = [p.strip('"') for p in line.split("\t")[1:]]
                if line.startswith("!Sample_characteristics_ch1") and "smoking_status" in line:
                    smoking = [p.strip('"') for p in line.split("\t")[1:]]
                if line.startswith("!Sample_supplementary_file"):
                    supp_files.append([p.strip('"') for p in line.split("\t")[1:]])
    except Exception as e:
        print(f"[WARN] Could not read series matrix: {e}")
        return {}

    if not samples or not smoking:
        print("[WARN] Could not parse samples/smoking from GSE224807 series matrix")
        return {}

    # Parse Sentrix keys from supplementary Grn IDAT URLs
    grn_files = []
    for row in supp_files:
        matches = [x for x in row if "Grn.idat" in x]
        if matches:
            grn_files = matches
            break

    if len(grn_files) != len(samples):
        print(f"[WARN] Sentrix URL count ({len(grn_files)}) != sample count ({len(samples)}) — falling back to GSM keys")
        sentrix_keys = samples
    else:
        sentrix_keys = []
        for url in grn_files:
            fname = url.split("/")[-1]   # e.g. GSM7032611_100946230055_R01C01_Grn.idat.gz
            parts = fname.split("_")
            if len(parts) >= 3:
                sentrix_keys.append(f"{parts[1]}_{parts[2]}")
            else:
                sentrix_keys.append(None)

    label_map = {}
    for key, status in zip(sentrix_keys, smoking):
        if key is None:
            continue
        s = status.lower()
        if "sm" in s and "ns" not in s:
            label_map[key] = 1
        elif "ns" in s:
            label_map[key] = 0

    print(f"[LABELS] Cohort B — {len(label_map)} labeled | "
          f"Smokers: {sum(label_map.values())} | "
          f"Non-smokers: {sum(1 for v in label_map.values() if v == 0)}")
    return label_map

label_map = load_cohort_b_labels(SERIES_MAT_B)

# ─────────────────────────────────────────────
# STEP 3 — Align samples, stack data, feature select
# ─────────────────────────────────────────────

print("\n[DATA] Aligning sample columns across pipelines ...")
all_cols = [set(df.columns) for df in matrices.values()]
shared_samples = sorted(set.intersection(*all_cols))
print(f"[DATA] Shared samples: {len(shared_samples)}")

print(f"[DATA] Stacking {len(pipeline_ids)} × {len(shared_samples)} samples ...")
data = np.stack([
    matrices[pid][shared_samples].T.values.astype(np.float32)
    for pid in pipeline_ids
], axis=0)  # (n_pipelines, n_samples, n_probes)

nan_count = np.isnan(data).sum()
if nan_count > 0:
    print(f"[DATA] Imputing {nan_count:,} NaN → 0.5")
    data = np.nan_to_num(data, nan=0.5)

# Feature selection — same logic as training: top-N by inter-sample variance
# on mean-across-pipelines to isolate biological signal
print(f"[FEAT] Selecting top {TOP_PROBES:,} probes by inter-sample variance ...")
mean_across_pipelines = data.mean(axis=0)           # (n_samples, n_probes)
probe_variance        = mean_across_pipelines.var(axis=0)  # (n_probes,)
top_idx = np.argsort(probe_variance)[-TOP_PROBES:]
top_idx = np.sort(top_idx)
data    = data[:, :, top_idx]
print(f"[FEAT] Done — data shape: {data.shape}")

# Attach labels
labels_arr = np.array([label_map.get(sid, -1) for sid in shared_samples], dtype=np.int64)
labeled_mask = labels_arr >= 0
print(f"[DATA] Samples with labels: {labeled_mask.sum()}/{len(shared_samples)}")

# ─────────────────────────────────────────────
# STEP 4 — Load trained encoder (frozen)
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Loading checkpoint")
print("=" * 60)

ckpt = torch.load(CKPT_PATH, map_location=device)
input_dim   = ckpt["input_dim"]
train_auc   = ckpt.get("auc", None)
train_epoch = ckpt.get("epoch", None)

print(f"Checkpoint: epoch={train_epoch} | train AUC={train_auc:.4f} | input_dim={input_dim:,}")

if input_dim != TOP_PROBES:
    raise ValueError(
        f"Checkpoint input_dim={input_dim} but TOP_PROBES={TOP_PROBES}. "
        f"Set TOP_PROBES to match the checkpoint."
    )

encoder = MLPEncoder(input_dim=input_dim, dropout=0.0).to(device)
encoder.load_state_dict(ckpt["encoder_state"])
encoder.eval()

total_params = sum(p.numel() for p in encoder.parameters())
print(f"Encoder parameters: {total_params:,}")

# ─────────────────────────────────────────────
# STEP 5 — Embed Cohort B samples
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Embedding Cohort B")
print("=" * 60)

embeddings = []
with torch.no_grad():
    for i in range(data.shape[1]):
        vecs = torch.from_numpy(data[:, i, :]).to(device)  # (n_pipelines, input_dim)
        hs = []
        for v in vecs:
            h, _ = encoder(v.unsqueeze(0))
            hs.append(h.squeeze(0).cpu().numpy())
        embeddings.append(np.mean(hs, axis=0))  # average across pipelines

embeddings = np.array(embeddings)   # (n_samples, 128)
print(f"Embeddings shape: {embeddings.shape}")

# Save embeddings
np.save(os.path.join(OUT_DIR, "cohort_b_embeddings.npy"), embeddings)
np.save(os.path.join(OUT_DIR, "cohort_b_labels.npy"), labels_arr)
np.save(os.path.join(OUT_DIR, "cohort_b_sample_ids.npy"), np.array(shared_samples))
print(f"Embeddings saved → {OUT_DIR}/cohort_b_embeddings.npy")

# ─────────────────────────────────────────────
# STEP 6 — Linear probe AUC (cross-cohort evaluation)
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Cross-cohort linear probe evaluation")
print("=" * 60)

if labeled_mask.sum() < 10:
    print("[WARN] Fewer than 10 labeled samples — skipping AUC evaluation")
    print("       Upload GSE224807_series_matrix.txt.gz to /content/ and re-run")
else:
    X = embeddings[labeled_mask]
    y = labels_arr[labeled_mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_folds = min(5, int(y.sum()), int((y == 0).sum()))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)

    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")

    cohort_b_auc = float(scores.mean())
    cohort_b_std = float(scores.std())

    print(f"\n{'─'*40}")
    print(f"  Cohort B — Cross-cohort AUC: {cohort_b_auc:.4f} ± {cohort_b_std:.4f}")
    print(f"  Cohort A train AUC (Run 3):  {train_auc:.4f}")
    print(f"  Transfer gap:                {train_auc - cohort_b_auc:+.4f}")
    print(f"  CV folds: {n_folds} | n_labeled: {labeled_mask.sum()} "
          f"(SM={y.sum()}, NS={(y==0).sum()})")
    print(f"{'─'*40}\n")

    # Per-fold breakdown
    print("Per-fold AUC:", " | ".join(f"{s:.4f}" for s in scores))

    # ─────────────────────────────────────────────
    # STEP 7 — Negative control: label permutation
    # ─────────────────────────────────────────────
    print("\n[NEG CTRL] Label permutation test (10 shuffles) ...")
    rng = np.random.default_rng(SEED)
    perm_aucs = []
    for _ in range(10):
        y_perm = rng.permutation(y)
        s = cross_val_score(clf, X_scaled, y_perm, cv=cv, scoring="roc_auc")
        perm_aucs.append(s.mean())
    print(f"Permutation AUC: {np.mean(perm_aucs):.4f} ± {np.std(perm_aucs):.4f}  "
          f"(expected ~0.50)")

    # ─────────────────────────────────────────────
    # STEP 8 — Save results summary
    # ─────────────────────────────────────────────
    results = {
        "cohort_b_auc_mean":  cohort_b_auc,
        "cohort_b_auc_std":   cohort_b_std,
        "cohort_b_auc_folds": scores.tolist(),
        "cohort_a_train_auc": train_auc,
        "transfer_gap":       float(train_auc - cohort_b_auc),
        "n_labeled":          int(labeled_mask.sum()),
        "n_smokers":          int(y.sum()),
        "n_nonsmokers":       int((y == 0).sum()),
        "permutation_auc_mean": float(np.mean(perm_aucs)),
        "permutation_auc_std":  float(np.std(perm_aucs)),
        "top_probes":         TOP_PROBES,
        "checkpoint_epoch":   train_epoch,
    }

    import json
    results_path = os.path.join(OUT_DIR, "cross_cohort_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # ─────────────────────────────────────────────
    # STEP 9 — UMAP visualization
    # ─────────────────────────────────────────────
    try:
        import umap
        print("\n[UMAP] Computing 2D projection ...")
        reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#2166ac" if lb == 0 else "#d6604d" for lb in y]
        ax.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, s=25, alpha=0.8, edgecolors="none")

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#2166ac", label="Non-smoker"),
            Patch(color="#d6604d", label="Smoker"),
        ])
        ax.set_title(
            f"MethCLR Embeddings — Cohort B (GSE224807)\n"
            f"AUC={cohort_b_auc:.4f} | TOP_PROBES={TOP_PROBES:,}",
            fontsize=11
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        plt.tight_layout()
        umap_path = os.path.join(OUT_DIR, "umap_cohort_b.png")
        plt.savefig(umap_path, dpi=150)
        print(f"UMAP saved → {umap_path}")
    except ImportError:
        print("[UMAP] umap-learn not installed — skipping. Run: !pip install umap-learn")

print("\n[DONE] Cross-cohort evaluation complete.")
print(f"       Results in {OUT_DIR}/")
