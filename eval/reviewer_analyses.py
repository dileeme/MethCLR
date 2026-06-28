#!/usr/bin/env python
"""
MethCLR — Reviewer Analyses
Tasks 1, 2, 3 in response to BIBM review comments.

Task 1 — Matched-N CV on Cohort B (fix sample-size confound)
Task 2 — Paired bootstrap significance tests (10,000 resamples)
Task 3 — Semi-supervised fine-tuning curve (0%→100% Cohort B supervision)

All analyses use the frozen Run 3 checkpoint (methclr_best.pt).
Probe selection is fit strictly on Cohort A and applied to Cohort B.
Embeddings are cached after first computation; re-run is fast.

Run from project root:
    python eval/reviewer_analyses.py

Outputs in eval/:
    rev_table1_matched_n.json
    rev_table2_bootstrap.json
    rev_table3_semisup.json
    rev_summary.txt              <- formatted tables for paper
"""

import os, sys, json, gzip, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from encoder import MLPEncoder

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.join(_HERE, "..")
CKPT_PATH    = os.path.join(_ROOT, "results", "methclr_best.pt")
MATRIX_DIR_A = os.path.join(_ROOT, "beta_matrices")
MATRIX_DIR_B = os.path.join(_ROOT, "cohort_b_beta_matrices")
SERIES_A     = os.path.join(os.path.expanduser("~"), "Downloads",
                             "GSE85210_series_matrix.txt.gz")
SERIES_B     = os.path.join(os.path.expanduser("~"), "Downloads",
                             "GSE224807-GPL13534_series_matrix.txt.gz")
CACHE_DIR    = _HERE

TOP_PROBES   = 20_000
SEED         = 42
N_FOLDS      = 5
N_REPS_T1    = 20       # Task 1 repetitions
N_REPS_T3    = 10       # Task 3 repetitions
N_BOOT       = 10_000   # Task 2 bootstrap resamples
AE_EPOCHS    = 100
AE_BATCH     = 64
AE_DIM       = 128

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")   # inference-only; CPU is sufficient

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_labels(path, smoking_field):
    if not os.path.exists(path):
        print(f"[WARN] Not found: {path}")
        return {}
    samples, phenotype, supp_files = [], [], []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("!Sample_geo_accession"):
                samples = [p.strip('"') for p in line.split("\t")[1:]]
            if line.startswith("!Sample_characteristics_ch1") and smoking_field in line:
                phenotype = [p.strip('"') for p in line.split("\t")[1:]]
            if line.startswith("!Sample_supplementary_file"):
                supp_files.append([p.strip('"') for p in line.split("\t")[1:]])
    if not samples or not phenotype:
        return {}
    grn = []
    for row in supp_files:
        hits = [x for x in row if "Grn.idat" in x]
        if hits:
            grn = hits; break
    if len(grn) != len(samples):
        return {}
    out = {}
    for url, status in zip(grn, phenotype):
        parts = url.split("/")[-1].split("_")
        if len(parts) < 3:
            continue
        key = f"{parts[1]}_{parts[2]}"
        s = status.lower()
        out[key] = 0 if ("non" in s or "ns" in s) else 1
    return out


def load_cohort(matrix_dir, label_map):
    files = sorted(f for f in os.listdir(matrix_dir) if f.endswith("_intersected.csv.gz"))
    matrices = {}
    for fname in files:
        pid = fname.replace("_intersected.csv.gz", "")
        df  = pd.read_csv(os.path.join(matrix_dir, fname), index_col=0, compression="gzip")
        matrices[pid] = df
        print(f"  {fname}: {df.shape}", flush=True)
    pid_list = list(matrices.keys())
    shared   = sorted(set.intersection(*[set(df.columns) for df in matrices.values()]))
    data = np.stack(
        [matrices[pid][shared].T.values.astype(np.float32) for pid in pid_list], axis=0
    )  # (n_pipes, n_samples, n_probes)
    data = np.nan_to_num(data, nan=0.5)
    labels = np.array([label_map.get(s, -1) for s in shared], dtype=np.int64)
    print(f"  -> {len(pid_list)} pipelines | {len(shared)} samples | labeled: {(labels>=0).sum()}")
    return data, labels


def select_probes_on_a(data_a, k):
    mean_a    = data_a.mean(axis=0)   # (n_samples, n_probes)
    probe_var = mean_a.var(axis=0)    # (n_probes,)
    top_idx   = np.sort(np.argsort(probe_var)[-k:])
    print(f"[FEAT] Selected top {k:,} probes from {data_a.shape[2]:,} (fit on Cohort A)")
    return top_idx


def embed_methclr(data_sel, enc):
    """data_sel: (n_pipes, n_samples, top_k). Returns (n_samples, 128)."""
    enc.eval()
    out = []
    with torch.no_grad():
        for i in range(data_sel.shape[1]):
            vecs = torch.from_numpy(data_sel[:, i, :])
            hs   = [enc(v.unsqueeze(0))[0].squeeze(0).numpy() for v in vecs]
            out.append(np.mean(hs, axis=0))
    return np.array(out, dtype=np.float32)


class _MethAE(nn.Module):
    def __init__(self, input_dim, bottleneck=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, bottleneck),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512),        nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )
    def forward(self, x):
        z = self.enc(x); return self.dec(z), z
    def encode(self, x):
        with torch.no_grad(): return self.enc(x)


def train_ae(X_train_s):
    ae  = _MethAE(X_train_s.shape[1], AE_DIM).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    Xt  = torch.from_numpy(X_train_s.astype(np.float32))
    dl  = DataLoader(TensorDataset(Xt), batch_size=AE_BATCH, shuffle=True)
    for ep in range(1, AE_EPOCHS + 1):
        ae.train()
        for (xb,) in dl:
            out, _ = ae(xb); loss = mse(out, xb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if ep % 25 == 0:
            print(f"  [AE] epoch {ep}/{AE_EPOCHS}", flush=True)
    ae.eval()
    with torch.no_grad():
        Z_train = ae.encode(Xt).numpy()
    return ae, Z_train.astype(np.float32)


def stratified_subsample(y, n_target, rng):
    """Sample n_target indices from y preserving class proportions."""
    classes, counts = np.unique(y, return_counts=True)
    selected = []
    remaining = n_target
    for i, (c, cnt) in enumerate(zip(classes, counts)):
        if i == len(classes) - 1:
            n_c = remaining
        else:
            n_c = round(n_target * cnt / len(y))
            remaining -= n_c
        idx_c = np.where(y == c)[0]
        n_c = min(n_c, len(idx_c))
        chosen = rng.choice(idx_c, size=n_c, replace=False)
        selected.extend(chosen.tolist())
    return np.array(selected)


def cv_collect_preds(X, y, folds, n_match=None):
    """
    Run CV on (X, y) with pre-specified folds.
    If n_match is given, subsample each training fold to n_match (stratified).
    Returns per-sample predicted probabilities for all samples.
    """
    preds = np.zeros(len(y), dtype=np.float64)
    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te       = X[te_idx]
        if n_match is not None and len(y_tr) > n_match:
            rng_sub = np.random.RandomState(fold_i * 999 + SEED)
            sub     = stratified_subsample(y_tr, n_match, rng_sub)
            X_tr    = X_tr[sub]
            y_tr    = y_tr[sub]
        scaler = StandardScaler().fit(X_tr)
        clf    = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        clf.fit(scaler.transform(X_tr), y_tr)
        preds[te_idx] = clf.predict_proba(scaler.transform(X_te))[:, 1]
    return preds


def bootstrap_delta_auc(y, p1, p2, n_boot=N_BOOT, seed=SEED):
    """
    Paired bootstrap on AUC difference (method1 - method2).
    Returns (delta_obs, ci_lo, ci_hi, p_value).
    p_value = fraction of bootstrap deltas that cross zero (one-sided).
    """
    rng     = np.random.RandomState(seed)
    n       = len(y)
    delta_obs = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    deltas  = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        deltas.append(roc_auc_score(y_b, p1[idx]) - roc_auc_score(y_b, p2[idx]))
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    if delta_obs >= 0:
        p_val = float(np.mean(deltas <= 0))
    else:
        p_val = float(np.mean(deltas >= 0))
    return float(delta_obs), float(ci_lo), float(ci_hi), p_val


# ─────────────────────────────────────────────────────────────────────────────
# CACHE LAYER
# ─────────────────────────────────────────────────────────────────────────────

CACHE_FILES = {
    "emb_a_methclr": os.path.join(CACHE_DIR, "rev_cache_emb_a_methclr.npy"),
    "emb_b_methclr": os.path.join(CACHE_DIR, "rev_cache_emb_b_methclr.npy"),
    "emb_a_pca":     os.path.join(CACHE_DIR, "rev_cache_emb_a_pca.npy"),
    "emb_b_pca":     os.path.join(CACHE_DIR, "rev_cache_emb_b_pca.npy"),
    "emb_a_ae":      os.path.join(CACHE_DIR, "rev_cache_emb_a_ae.npy"),
    "emb_b_ae":      os.path.join(CACHE_DIR, "rev_cache_emb_b_ae.npy"),
    "emb_a_rand":    os.path.join(CACHE_DIR, "rev_cache_emb_a_rand.npy"),
    "emb_b_rand":    os.path.join(CACHE_DIR, "rev_cache_emb_b_rand.npy"),
    "labels_a":      os.path.join(CACHE_DIR, "rev_cache_labels_a.npy"),
    "labels_b":      os.path.join(CACHE_DIR, "rev_cache_labels_b.npy"),
}

cache_complete = all(os.path.exists(p) for p in CACHE_FILES.values())

# ─────────────────────────────────────────────────────────────────────────────
# LOAD / COMPUTE EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

if cache_complete:
    print("[CACHE] All embeddings found — loading from disk (fast path).")
    emb_a_methclr = np.load(CACHE_FILES["emb_a_methclr"])
    emb_b_methclr = np.load(CACHE_FILES["emb_b_methclr"])
    emb_a_pca     = np.load(CACHE_FILES["emb_a_pca"])
    emb_b_pca     = np.load(CACHE_FILES["emb_b_pca"])
    emb_a_ae      = np.load(CACHE_FILES["emb_a_ae"])
    emb_b_ae      = np.load(CACHE_FILES["emb_b_ae"])
    emb_a_rand    = np.load(CACHE_FILES["emb_a_rand"])
    emb_b_rand    = np.load(CACHE_FILES["emb_b_rand"])
    y_a           = np.load(CACHE_FILES["labels_a"])
    y_b           = np.load(CACHE_FILES["labels_b"])
    print(f"  Cohort A: {emb_a_methclr.shape}  y_a: {y_a.shape} "
          f"(SM={y_a.sum()} NS={(y_a==0).sum()})")
    print(f"  Cohort B: {emb_b_methclr.shape}  y_b: {y_b.shape} "
          f"(SM={y_b.sum()} NS={(y_b==0).sum()})")

else:
    print("[CACHE] Embeddings not cached — loading beta matrices (this takes ~15 min).")

    # ── Labels ─────────────────────────────────────────────────────────────────
    print("\nLoading labels ...")
    label_map_a = _parse_labels(SERIES_A, "subject status")
    label_map_b = _parse_labels(SERIES_B, "smoking_status")
    print(f"  A labels: {len(label_map_a)}  B labels: {len(label_map_b)}")

    # ── Beta matrices ──────────────────────────────────────────────────────────
    print("\nLoading Cohort A beta matrices ...")
    data_a, labs_a_full = load_cohort(MATRIX_DIR_A, label_map_a)

    print("\nLoading Cohort B beta matrices ...")
    data_b, labs_b_full = load_cohort(MATRIX_DIR_B, label_map_b)

    # ── Feature selection — strictly on Cohort A ───────────────────────────────
    print()
    top_idx = select_probes_on_a(data_a, TOP_PROBES)
    data_a_sel = data_a[:, :, top_idx]   # (n_pipes, n_a, 20k)
    data_b_sel = data_b[:, :, top_idx]   # (n_pipes, n_b, 20k)

    # Labeled subsets
    mask_a = labs_a_full >= 0
    mask_b = labs_b_full >= 0
    y_a    = labs_a_full[mask_a].astype(np.int64)
    y_b    = labs_b_full[mask_b].astype(np.int64)
    print(f"Labeled A: {y_a.shape} SM={y_a.sum()} NS={(y_a==0).sum()}")
    print(f"Labeled B: {y_b.shape} SM={y_b.sum()} NS={(y_b==0).sum()}")

    # ── Mean-across-pipelines feature matrices (for PCA/AE) ───────────────────
    X_a_raw = data_a_sel[:, mask_a, :].mean(axis=0)   # (n_a, 20k)
    X_b_raw = data_b_sel[:, mask_b, :].mean(axis=0)   # (n_b, 20k)

    scaler_feat = StandardScaler().fit(X_a_raw)
    X_a_s = scaler_feat.transform(X_a_raw)
    X_b_s = scaler_feat.transform(X_b_raw)

    # ── Load MethCLR checkpoint ────────────────────────────────────────────────
    print("\nLoading MethCLR checkpoint ...")
    ckpt       = torch.load(CKPT_PATH, map_location="cpu")
    input_dim  = ckpt["input_dim"]
    ckpt_epoch = ckpt.get("epoch")
    ckpt_auc   = ckpt.get("auc")
    print(f"  epoch={ckpt_epoch}  stored_AUC={ckpt_auc:.4f}  input_dim={input_dim:,}")

    encoder = MLPEncoder(input_dim=input_dim, dropout=0.0)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    # ── MethCLR embeddings ─────────────────────────────────────────────────────
    print("\nEmbedding Cohort A with MethCLR ...")
    emb_a_methclr = embed_methclr(data_a_sel[:, mask_a, :], encoder)
    print(f"  A MethCLR: {emb_a_methclr.shape}")

    print("Embedding Cohort B with MethCLR ...")
    emb_b_methclr = embed_methclr(data_b_sel[:, mask_b, :], encoder)
    print(f"  B MethCLR: {emb_b_methclr.shape}")

    # ── Random-init encoder ────────────────────────────────────────────────────
    print("\nEmbedding with random-init encoder ...")
    torch.manual_seed(SEED)
    rand_enc = MLPEncoder(input_dim=input_dim, dropout=0.0)
    rand_enc.eval()
    emb_a_rand = embed_methclr(data_a_sel[:, mask_a, :], rand_enc)
    emb_b_rand = embed_methclr(data_b_sel[:, mask_b, :], rand_enc)
    print(f"  A rand: {emb_a_rand.shape}  B rand: {emb_b_rand.shape}")

    # ── PCA (fit on A, transform A+B) ─────────────────────────────────────────
    print("\nFitting PCA on Cohort A ...")
    pca        = PCA(n_components=AE_DIM, random_state=SEED).fit(X_a_s)
    emb_a_pca  = pca.transform(X_a_s).astype(np.float32)
    emb_b_pca  = pca.transform(X_b_s).astype(np.float32)
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"  A PCA: {emb_a_pca.shape}  B PCA: {emb_b_pca.shape}")

    # ── Autoencoder (train on A, encode A+B) ──────────────────────────────────
    print("\nTraining Autoencoder on Cohort A ...")
    ae, emb_a_ae = train_ae(X_a_s)
    with torch.no_grad():
        emb_b_ae = ae.encode(
            torch.from_numpy(X_b_s.astype(np.float32))
        ).numpy().astype(np.float32)
    print(f"  A AE: {emb_a_ae.shape}  B AE: {emb_b_ae.shape}")

    # ── Save cache ─────────────────────────────────────────────────────────────
    print("\nSaving embedding cache ...")
    np.save(CACHE_FILES["emb_a_methclr"], emb_a_methclr)
    np.save(CACHE_FILES["emb_b_methclr"], emb_b_methclr)
    np.save(CACHE_FILES["emb_a_pca"],     emb_a_pca)
    np.save(CACHE_FILES["emb_b_pca"],     emb_b_pca)
    np.save(CACHE_FILES["emb_a_ae"],      emb_a_ae)
    np.save(CACHE_FILES["emb_b_ae"],      emb_b_ae)
    np.save(CACHE_FILES["emb_a_rand"],    emb_a_rand)
    np.save(CACHE_FILES["emb_b_rand"],    emb_b_rand)
    np.save(CACHE_FILES["labels_a"],      y_a)
    np.save(CACHE_FILES["labels_b"],      y_b)
    print("  Cached.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE N_MATCH
# ─────────────────────────────────────────────────────────────────────────────

N_A = len(y_a)
N_B = len(y_b)
N_MATCH = int(N_A * (N_FOLDS - 1) / N_FOLDS)  # ≈ 202 for N_A=253, 5-fold
print(f"\n[CONFIG] N_A={N_A}  N_B={N_B}  N_MATCH={N_MATCH} "
      f"(= N_A × {N_FOLDS-1}/{N_FOLDS})")

# Method dictionaries
methods_a = {
    "MethCLR": emb_a_methclr,
    "PCA":     emb_a_pca,
    "AE":      emb_a_ae,
    "RandInit":emb_a_rand,
}
methods_b = {
    "MethCLR": emb_b_methclr,
    "PCA":     emb_b_pca,
    "AE":      emb_b_ae,
    "RandInit":emb_b_rand,
}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — MATCHED-N CV ON COHORT B
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("TASK 1 — Matched-N CV on Cohort B")
print(f"  {N_REPS_T1} repetitions × {N_FOLDS}-fold CV | "
      f"training pool subsampled to N={N_MATCH}")
print("="*65)

t1_results = {}

for method_name in ["MethCLR", "PCA", "AE"]:
    X_b = methods_b[method_name]
    rep_means = []
    for rep in range(N_REPS_T1):
        rep_seed = rep * 1000 + 7
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=rep_seed)
        fold_aucs = []
        for fold_i, (tr_idx, te_idx) in enumerate(cv.split(np.zeros(N_B), y_b)):
            X_tr, y_tr = X_b[tr_idx], y_b[tr_idx]
            X_te, y_te = X_b[te_idx], y_b[te_idx]
            # Subsample training pool to N_MATCH (stratified by B's class proportions)
            rng_sub = np.random.RandomState(rep_seed * 13 + fold_i)
            sub     = stratified_subsample(y_tr, N_MATCH, rng_sub)
            X_tr_s, y_tr_s = X_tr[sub], y_tr[sub]
            scaler = StandardScaler().fit(X_tr_s)
            clf    = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
            clf.fit(scaler.transform(X_tr_s), y_tr_s)
            prob   = clf.predict_proba(scaler.transform(X_te))[:, 1]
            if len(np.unique(y_te)) == 2:
                fold_aucs.append(roc_auc_score(y_te, prob))
        rep_means.append(np.mean(fold_aucs))

    mean_auc = float(np.mean(rep_means))
    std_auc  = float(np.std(rep_means))
    t1_results[method_name] = {
        "auc_mean":    mean_auc,
        "auc_std":     std_auc,
        "rep_means":   [float(x) for x in rep_means],
        "n_reps":      N_REPS_T1,
        "n_folds":     N_FOLDS,
        "n_match":     N_MATCH,
        "n_b":         N_B,
    }
    print(f"  {method_name:<10}  {mean_auc:.4f} ± {std_auc:.4f}")

# Also compute standard (unmatched) B CV for reference — just MethCLR, 1 run
print("\n  [Reference] Standard Cohort B 5-fold CV (no subsample):")
cv_ref = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for method_name in ["MethCLR", "PCA", "AE"]:
    X_b = methods_b[method_name]
    fold_aucs = []
    for tr_idx, te_idx in cv_ref.split(np.zeros(N_B), y_b):
        scaler = StandardScaler().fit(X_b[tr_idx])
        clf    = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        clf.fit(scaler.transform(X_b[tr_idx]), y_b[tr_idx])
        prob   = clf.predict_proba(scaler.transform(X_b[te_idx]))[:, 1]
        if len(np.unique(y_b[te_idx])) == 2:
            fold_aucs.append(roc_auc_score(y_b[te_idx], prob))
    t1_results[method_name]["cohort_b_unmatched_auc"] = float(np.mean(fold_aucs))
    t1_results[method_name]["cohort_b_unmatched_std"] = float(np.std(fold_aucs))
    print(f"    {method_name:<10}  {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

# Standard Cohort A 5-fold CV for reference
print("\n  [Reference] Standard Cohort A 5-fold CV:")
cv_a_ref = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for method_name in ["MethCLR", "PCA", "AE", "RandInit"]:
    X_a = methods_a[method_name]
    fold_aucs = []
    for tr_idx, te_idx in cv_a_ref.split(np.zeros(N_A), y_a):
        scaler = StandardScaler().fit(X_a[tr_idx])
        clf    = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        clf.fit(scaler.transform(X_a[tr_idx]), y_a[tr_idx])
        prob   = clf.predict_proba(scaler.transform(X_a[te_idx]))[:, 1]
        if len(np.unique(y_a[te_idx])) == 2:
            fold_aucs.append(roc_auc_score(y_a[te_idx], prob))
    t1_results[f"{method_name}_cohort_a"] = {
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std":  float(np.std(fold_aucs)),
        "auc_folds": [float(x) for x in fold_aucs],
    }
    print(f"    {method_name:<10}  {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

out_t1 = os.path.join(_HERE, "rev_table1_matched_n.json")
with open(out_t1, "w") as f:
    json.dump(t1_results, f, indent=2)
print(f"\n[Task 1] Saved → {out_t1}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — PAIRED BOOTSTRAP SIGNIFICANCE TESTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print(f"TASK 2 — Paired bootstrap significance tests (n={N_BOOT:,} resamples)")
print("="*65)

# Fixed CV folds — same splits for all methods to enable pairing
cv_a_fixed = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds_a    = list(cv_a_fixed.split(np.zeros(N_A), y_a))

# Fixed B folds for matched-N bootstrap (use rep seed 0)
BOOT_REP_SEED = 0
cv_b_fixed = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BOOT_REP_SEED)
folds_b_boot = list(cv_b_fixed.split(np.zeros(N_B), y_b))

print("\n  Collecting per-sample held-out predictions (Cohort A 5-fold CV) ...")
preds_a = {}
for mname, Xa in methods_a.items():
    preds_a[mname] = cv_collect_preds(Xa, y_a, folds_a, n_match=None)
    auc = roc_auc_score(y_a, preds_a[mname])
    print(f"    {mname:<10}  AUC={auc:.4f}")

print("\n  Collecting per-sample held-out predictions "
      f"(Cohort B matched-N CV, rep_seed={BOOT_REP_SEED}) ...")
preds_b_matched = {}
for mname, Xb in methods_b.items():
    arr = np.zeros(N_B, dtype=np.float64)
    for fold_i, (tr_idx, te_idx) in enumerate(folds_b_boot):
        X_tr, y_tr = Xb[tr_idx], y_b[tr_idx]
        X_te, y_te = Xb[te_idx], y_b[te_idx]
        rng_sub = np.random.RandomState(BOOT_REP_SEED * 13 + fold_i)
        sub     = stratified_subsample(y_tr, N_MATCH, rng_sub)
        X_tr_s, y_tr_s = X_tr[sub], y_tr[sub]
        scaler  = StandardScaler().fit(X_tr_s)
        clf     = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        clf.fit(scaler.transform(X_tr_s), y_tr_s)
        arr[te_idx] = clf.predict_proba(scaler.transform(X_te))[:, 1]
    preds_b_matched[mname] = arr
    auc = roc_auc_score(y_b, arr)
    print(f"    {mname:<10}  AUC={auc:.4f} (matched-N B)")

# Run bootstrap for all pairs of interest
t2_results = {"cohort_a": {}, "cohort_b_matched_n": {}}
PAIRS = [("MethCLR", "PCA"), ("MethCLR", "AE"), ("MethCLR", "RandInit")]

print("\n  Cohort A bootstrap results:")
print(f"  {'Pair':<28} {'ΔAUC':>8}  {'95% CI':>20}  {'p-value':>10}")
print("  " + "-"*72)
for m1, m2 in PAIRS:
    delta, lo, hi, pv = bootstrap_delta_auc(
        y_a, preds_a[m1], preds_a[m2]
    )
    key = f"{m1}_vs_{m2}"
    t2_results["cohort_a"][key] = {
        "delta_auc": delta, "ci_95_lo": lo, "ci_95_hi": hi, "p_value": pv,
        "auc_m1": float(roc_auc_score(y_a, preds_a[m1])),
        "auc_m2": float(roc_auc_score(y_a, preds_a[m2])),
    }
    sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
    print(f"  {m1} vs {m2:<18} {delta:>+8.4f}  [{lo:+.4f}, {hi:+.4f}]  {pv:>10.4f} {sig}")

print("\n  Cohort B (matched-N) bootstrap results:")
print(f"  {'Pair':<28} {'ΔAUC':>8}  {'95% CI':>20}  {'p-value':>10}")
print("  " + "-"*72)
for m1, m2 in PAIRS:
    delta, lo, hi, pv = bootstrap_delta_auc(
        y_b, preds_b_matched[m1], preds_b_matched[m2]
    )
    key = f"{m1}_vs_{m2}"
    t2_results["cohort_b_matched_n"][key] = {
        "delta_auc": delta, "ci_95_lo": lo, "ci_95_hi": hi, "p_value": pv,
        "auc_m1": float(roc_auc_score(y_b, preds_b_matched[m1])),
        "auc_m2": float(roc_auc_score(y_b, preds_b_matched[m2])),
    }
    sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
    print(f"  {m1} vs {m2:<18} {delta:>+8.4f}  [{lo:+.4f}, {hi:+.4f}]  {pv:>10.4f} {sig}")

out_t2 = os.path.join(_HERE, "rev_table2_bootstrap.json")
with open(out_t2, "w") as f:
    json.dump(t2_results, f, indent=2)
print(f"\n[Task 2] Saved → {out_t2}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — SEMI-SUPERVISED FINE-TUNING CURVE (MethCLR only)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("TASK 3 — Semi-supervised fine-tuning curve")
print(f"  Supervision levels: 0%, 10%, 20%, 100%  "
      f"({N_REPS_T3} reps for 10%/20%)")
print("="*65)

X_a_methclr = emb_a_methclr   # (N_A, 128)
X_b_methclr = emb_b_methclr   # (N_B, 128)

t3_results = {}

# 0% — label-free: fit LR on ALL Cohort A, predict ALL Cohort B
# (Re-computed fresh here for consistency with probe selection)
scaler_lf = StandardScaler().fit(X_a_methclr)
clf_lf    = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
clf_lf.fit(scaler_lf.transform(X_a_methclr), y_a)
prob_lf   = clf_lf.predict_proba(scaler_lf.transform(X_b_methclr))[:, 1]
auc_lf    = float(roc_auc_score(y_b, prob_lf))
t3_results["pct_0"] = {
    "supervision_pct": 0,
    "auc_mean": auc_lf,
    "auc_std":  None,
    "note": "LR fit on all Cohort A; evaluated on all Cohort B (label-free transfer)",
}
print(f"\n  0%  (label-free): {auc_lf:.4f}  "
      f"(previous p0_eval.py result: 0.6149)")

# Flag if number deviates substantially from cached result
if abs(auc_lf - 0.6149) > 0.01:
    print(f"  [FLAG] Label-free AUC differs from cached 0.6149 by "
          f"{abs(auc_lf-0.6149):.4f} — likely due to probe selection on A vs B")

# 10% and 20% — fit LR on subsample of Cohort B, evaluate on rest
for pct in [10, 20]:
    n_sup = max(2, int(round(N_B * pct / 100)))  # number of supervised B samples
    rep_aucs = []
    for rep in range(N_REPS_T3):
        rng_t3 = np.random.RandomState(rep * 777 + pct)
        # Stratified subsample of size n_sup from B
        sup_local = stratified_subsample(y_b, n_sup, rng_t3)
        rest_mask = np.ones(N_B, dtype=bool)
        rest_mask[sup_local] = False

        X_sup, y_sup   = X_b_methclr[sup_local], y_b[sup_local]
        X_rest, y_rest = X_b_methclr[rest_mask],  y_b[rest_mask]

        if len(np.unique(y_rest)) < 2:
            continue  # skip degenerate split

        scaler_t3 = StandardScaler().fit(X_sup)
        clf_t3    = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        clf_t3.fit(scaler_t3.transform(X_sup), y_sup)
        prob_t3   = clf_t3.predict_proba(scaler_t3.transform(X_rest))[:, 1]
        if len(np.unique(y_rest)) == 2:
            rep_aucs.append(roc_auc_score(y_rest, prob_t3))

    auc_mean = float(np.mean(rep_aucs))
    auc_std  = float(np.std(rep_aucs))
    t3_results[f"pct_{pct}"] = {
        "supervision_pct": pct,
        "n_supervised":    n_sup,
        "auc_mean":        auc_mean,
        "auc_std":         auc_std,
        "rep_aucs":        [float(x) for x in rep_aucs],
        "n_reps":          len(rep_aucs),
        "note": (f"LR fit on {pct}% of Cohort B ({n_sup} samples); "
                 f"evaluated on remaining {100-pct}% ({N_B-n_sup} samples)"),
    }
    print(f"  {pct:2d}%  (n_sup={n_sup:4d}):  {auc_mean:.4f} ± {auc_std:.4f}")

# 100% — standard 5-fold within-cohort B CV (use matched-N for comparability? No — use full CV for 100%)
# The matched-N result from Task 1 is also reported here for reference
t1_methclr_matchedn = t1_results.get("MethCLR", {})
auc_100_matched = t1_methclr_matchedn.get("auc_mean")
std_100_matched = t1_methclr_matchedn.get("auc_std")

# Full within-cohort B CV (unmatched, N_B training)
auc_100_unmatched = t1_methclr_matchedn.get("cohort_b_unmatched_auc")
std_100_unmatched = t1_methclr_matchedn.get("cohort_b_unmatched_std")

t3_results["pct_100"] = {
    "supervision_pct": 100,
    "n_supervised":    N_B,
    "auc_mean_unmatched": auc_100_unmatched,
    "auc_std_unmatched":  std_100_unmatched,
    "auc_mean_matched_n": auc_100_matched,
    "auc_std_matched_n":  std_100_matched,
    "note": "100% = within-cohort Cohort B 5-fold CV (1 run, unmatched)",
}
print(f"  100% (unmatched 5-fold): {auc_100_unmatched:.4f} ± {std_100_unmatched:.4f}")
print(f"  100% (matched-N 5-fold): {auc_100_matched:.4f} ± {std_100_matched:.4f}")

out_t3 = os.path.join(_HERE, "rev_table3_semisup.json")
with open(out_t3, "w") as f:
    json.dump(t3_results, f, indent=2)
print(f"\n[Task 3] Saved → {out_t3}")


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTED OUTPUT TABLES
# ─────────────────────────────────────────────────────────────────────────────

lines = []
W = 72

def hdr(title):
    lines.append("=" * W)
    lines.append(title)
    lines.append("=" * W)

def sep():
    lines.append("-" * W)

lines.append("")
lines.append("MethCLR — Reviewer Response Analyses")
lines.append(f"N_A={N_A}  N_B={N_B}  N_MATCH={N_MATCH}  N_BOOT={N_BOOT:,}")
lines.append("")

# ── Table 1: Revised Table II (matched-N Cohort B CV) ─────────────────────
hdr("TABLE 1 (Task 1) — Revised Table II: Matched-N Cohort B CV")
lines.append(f"Cohort B 5-fold CV with training pool subsampled to N={N_MATCH}.")
lines.append(f"Repeated {N_REPS_T1}x. Mean ± std over repetition means.")
lines.append("")
lines.append(f"{'Method':<14} {'A CV (N=202)':<22} {'B matched-N CV':<22} {'B standard CV':<18}")
sep()
for mname in ["MethCLR", "PCA", "AE"]:
    a_r = t1_results.get(f"{mname}_cohort_a", {})
    b_r = t1_results.get(mname, {})
    a_str = f"{a_r['auc_mean']:.4f} ± {a_r['auc_std']:.4f}" if a_r else "—"
    b_m_str = f"{b_r['auc_mean']:.4f} ± {b_r['auc_std']:.4f}"
    b_s_str = f"{b_r['cohort_b_unmatched_auc']:.4f} ± {b_r['cohort_b_unmatched_std']:.4f}"
    lines.append(f"{mname:<14} {a_str:<22} {b_m_str:<22} {b_s_str:<18}")
# Add RandInit for A only
ri_r = t1_results.get("RandInit_cohort_a", {})
if ri_r:
    a_str = f"{ri_r['auc_mean']:.4f} ± {ri_r['auc_std']:.4f}"
    lines.append(f"{'RandInit':<14} {a_str:<22} {'N/A':<22} {'N/A':<18}")
lines.append("")
lines.append("A CV: standard 5-fold, 1 run (N≈202 training per fold)")
lines.append("B matched-N: training pool subsampled from N≈631 → N=202")
lines.append("B standard: 5-fold with full training pool N≈631")
lines.append("")

# ── Table 2: Bootstrap significance ───────────────────────────────────────
hdr("TABLE 2 (Task 2) — Bootstrap Significance (Paired, 10,000 resamples)")
lines.append(f"{'Method pair':<28} {'ΔAUC':>8}  {'95% CI':>22}  {'p-value':>10}  {'Sig':>4}")
sep()
lines.append("Cohort A 5-fold CV:")
for m1, m2 in PAIRS:
    r = t2_results["cohort_a"].get(f"{m1}_vs_{m2}", {})
    if not r:
        continue
    d, lo, hi, pv = r["delta_auc"], r["ci_95_lo"], r["ci_95_hi"], r["p_value"]
    sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
    lines.append(f"  {m1} vs {m2:<18} {d:>+8.4f}  [{lo:+.4f}, {hi:+.4f}]  {pv:>10.4f}  {sig:>4}")
lines.append("")
lines.append("Cohort B matched-N CV:")
for m1, m2 in PAIRS:
    r = t2_results["cohort_b_matched_n"].get(f"{m1}_vs_{m2}", {})
    if not r:
        continue
    d, lo, hi, pv = r["delta_auc"], r["ci_95_lo"], r["ci_95_hi"], r["p_value"]
    sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
    lines.append(f"  {m1} vs {m2:<18} {d:>+8.4f}  [{lo:+.4f}, {hi:+.4f}]  {pv:>10.4f}  {sig:>4}")
lines.append("")
lines.append("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05")
lines.append("Bootstrap over per-sample held-out CV predictions (not fold AUCs).")
lines.append("")

# ── Table 3: Semi-supervised curve ────────────────────────────────────────
hdr("TABLE 3 (Task 3) — Semi-supervised Fine-tuning Curve (MethCLR)")
lines.append(f"Frozen MethCLR encoder. LR head fit on x% of Cohort B.")
lines.append(f"10%/20%: {N_REPS_T3} reps. 0%=label-free. 100%=5-fold CV.")
lines.append("")
lines.append(f"{'Supervision':<14} {'N_sup':>7}  {'AUC':>22}  {'Note'}")
sep()
entries = [
    ("0%   (label-free)", None,    t3_results["pct_0"]["auc_mean"],  None,
     "LR on all Cohort A → Cohort B"),
    ("10%",               int(round(N_B*0.10)), t3_results["pct_10"]["auc_mean"], t3_results["pct_10"]["auc_std"],
     f"eval on 90% of B"),
    ("20%",               int(round(N_B*0.20)), t3_results["pct_20"]["auc_mean"], t3_results["pct_20"]["auc_std"],
     f"eval on 80% of B"),
    ("100% (5-fold CV, unmatched)",    N_B,    t3_results["pct_100"]["auc_mean_unmatched"], t3_results["pct_100"]["auc_std_unmatched"],
     "within-cohort B"),
    (f"100% (5-fold CV, matched-N={N_MATCH})", N_B, t3_results["pct_100"]["auc_mean_matched_n"], t3_results["pct_100"]["auc_std_matched_n"],
     "matched-N from Task 1"),
]
for label, n_sup, auc_m, auc_s, note in entries:
    n_str   = f"{n_sup:>7}" if n_sup is not None else "    N/A"
    auc_str = f"{auc_m:.4f} ± {auc_s:.4f}" if auc_s is not None else f"{auc_m:.4f}"
    lines.append(f"{label:<30} {n_str}  {auc_str:<22}  {note}")
lines.append("")
lines.append("Interpretation:")
lines.append("  0% → 10% gap shows gain from minimal Cohort B supervision.")
lines.append("  10% → 100% gap shows remaining label bottleneck.")
lines.append("")

# ── FLAGS ──────────────────────────────────────────────────────────────────
hdr("FLAGS — Results that may contradict current paper claims")
flags = []

# Flag 1: label-free MethCLR vs random-init on B
lf_methclr = t3_results["pct_0"]["auc_mean"]
lf_rand    = 0.6236  # from p0_results.json
if lf_methclr < lf_rand:
    flags.append(
        f"[FLAG] Label-free MethCLR AUC on B = {lf_methclr:.4f} is BELOW "
        f"random-init ({lf_rand:.4f}). This is already known from p0 eval "
        f"(0.615 vs 0.624) and contradicts a simple 'learned representations "
        f"generalise better' claim. Requires careful framing in paper."
    )

# Flag 2: PCA matched-N vs standard
for mname in ["PCA", "AE", "MethCLR"]:
    b_r = t1_results.get(mname, {})
    std_auc  = b_r.get("cohort_b_unmatched_auc", 0)
    match_auc = b_r.get("auc_mean", 0)
    drop = std_auc - match_auc
    if drop > 0.05:
        flags.append(
            f"[FLAG] {mname} Cohort B AUC drops {drop:.4f} under matched-N "
            f"({std_auc:.4f} → {match_auc:.4f}). This confirms the confound "
            f"was inflating that method's standard CV number."
        )

# Flag 3: if any method not sig different from MethCLR
for m1, m2 in PAIRS:
    r = t2_results["cohort_a"].get(f"{m1}_vs_{m2}", {})
    if r and r["p_value"] >= 0.05:
        flags.append(
            f"[FLAG] {m1} vs {m2} on Cohort A: NOT significant "
            f"(p={r['p_value']:.4f}). Cannot claim MethCLR is significantly "
            f"better than {m2} on Cohort A."
        )
    r = t2_results["cohort_b_matched_n"].get(f"{m1}_vs_{m2}", {})
    if r and r["p_value"] >= 0.05:
        flags.append(
            f"[FLAG] {m1} vs {m2} on Cohort B matched-N: NOT significant "
            f"(p={r['p_value']:.4f})."
        )

if not flags:
    lines.append("No contradictions detected.")
else:
    for fl in flags:
        lines.append(fl)
        lines.append("")

lines.append("")
lines.append("=" * W)
lines.append("End of reviewer analyses.")

summary = "\n".join(lines)
print("\n\n" + summary)

out_txt = os.path.join(_HERE, "rev_summary.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(summary)
print(f"\n\n[DONE] Summary → {out_txt}")
