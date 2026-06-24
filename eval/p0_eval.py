"""
MethCLR — P0 Pre-submission evaluation script
Runs locally (CPU). No retraining.

Produces:
  eval/p0_results.json  — all numbers needed to update methclr.tex

P0.1  Label-free cross-cohort transfer:
      LR fit on ALL Cohort A embeddings → evaluated on ALL Cohort B.
      Same protocol re-run for PCA and AE baselines.

P0.2  Random-initialised encoder baseline:
      Same Cohort A 5-fold CV + label-free Cohort B AUC.

P0.3  AHRR probe check:
      Is cg05575921 present in the top-20K and top-5K feature-selected sets?

P0.4  Cohort A MethCLR 5-fold CV AUC with per-fold values (to produce ± std).
"""

import os, sys, gzip, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from encoder import MLPEncoder

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.join(_HERE, "..")
CKPT_PATH    = os.path.join(_ROOT, "results", "methclr_best.pt")
MATRIX_DIR_A = os.path.join(_ROOT, "beta_matrices")
MATRIX_DIR_B = os.path.join(_ROOT, "cohort_b_beta_matrices")
SERIES_A     = os.path.join(os.path.expanduser("~"), "Downloads", "GSE85210_series_matrix.txt.gz")
SERIES_B     = os.path.join(os.path.expanduser("~"), "Downloads", "GSE224807-GPL13534_series_matrix.txt.gz")
OUT_DIR      = _HERE
os.makedirs(OUT_DIR, exist_ok=True)

TOP_PROBES   = 20_000
SEED         = 42
N_FOLDS      = 5
AE_EMBED_DIM = 128
AE_EPOCHS    = 100
AE_BATCH     = 64

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")   # inference only — CPU is fine

# ── Label loaders ─────────────────────────────────────────────────────────────

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


def load_labels_a():
    m = _parse_labels(SERIES_A, "subject status")
    print(f"[A] Labels: {len(m)}  SM={sum(m.values())}  NS={sum(1 for v in m.values() if v==0)}")
    return m

def load_labels_b():
    m = _parse_labels(SERIES_B, "smoking_status")
    print(f"[B] Labels: {len(m)}  SM={sum(m.values())}  NS={sum(1 for v in m.values() if v==0)}")
    return m


# ── Matrix loader ─────────────────────────────────────────────────────────────

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
    )
    data = np.nan_to_num(data, nan=0.5)
    labels = np.array([label_map.get(s, -1) for s in shared], dtype=np.int64)
    print(f"  -> {len(pid_list)} pipelines | {len(shared)} samples | labeled: {(labels>=0).sum()}")
    return data, labels, shared, pid_list


# ── Feature selection ─────────────────────────────────────────────────────────

def select_probes(data_a, k, matrices_a_first_file):
    """Returns top-k probe indices selected on Cohort A mean-across-pipelines."""
    mean_a     = data_a.mean(axis=0)
    probe_var  = mean_a.var(axis=0)
    top_idx    = np.sort(np.argsort(probe_var)[-k:])
    print(f"[FEAT] Selected top {k:,} probes from {data_a.shape[2]:,}")
    return top_idx


# ── Embed with MethCLR ────────────────────────────────────────────────────────

def embed_methclr(data, encoder):
    """data: (n_pipelines, n_samples, n_probes). Returns (n_samples, 128)."""
    encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(data.shape[1]):
            vecs = torch.from_numpy(data[:, i, :])
            hs = [encoder(v.unsqueeze(0))[0].squeeze(0).numpy() for v in vecs]
            out.append(np.mean(hs, axis=0))
    return np.array(out)


# ── Linear probe helpers ──────────────────────────────────────────────────────

def cv_probe(X, y, label):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    scores = cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc")
    print(f"  [{label}] folds: {' '.join(f'{s:.4f}' for s in scores)}  "
          f"mean={scores.mean():.4f} ± {scores.std():.4f}")
    return float(scores.mean()), float(scores.std()), scores.tolist()


def label_free_transfer(X_a, y_a, X_b, y_b, label):
    """Fit on ALL Cohort A, predict ALL Cohort B — no CV on B."""
    scaler = StandardScaler().fit(X_a)
    Xa_s = scaler.transform(X_a)
    Xb_s = scaler.transform(X_b)
    clf  = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    clf.fit(Xa_s, y_a)
    proba = clf.predict_proba(Xb_s)[:, 1]
    auc   = float(roc_auc_score(y_b, proba))
    print(f"  [{label}] label-free transfer AUC (B): {auc:.4f}")
    return auc


# ── Autoencoder ───────────────────────────────────────────────────────────────

class MethAE(nn.Module):
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
    ae  = MethAE(X_train_s.shape[1]).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    Xt  = torch.from_numpy(X_train_s.astype(np.float32))
    dl  = DataLoader(TensorDataset(Xt), batch_size=AE_BATCH, shuffle=True)
    for epoch in range(1, AE_EPOCHS + 1):
        ae.train()
        for (xb,) in dl:
            out, _ = ae(xb)
            loss = mse(out, xb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if epoch % 20 == 0:
            print(f"  [AE] epoch {epoch}/{AE_EPOCHS}", flush=True)
    ae.eval()
    with torch.no_grad():
        Z = ae.encode(Xt).numpy()
    return ae, Z


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = {}

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Loading labels")
    label_a = load_labels_a()
    label_b = load_labels_b()

    print("\nLoading Cohort A")
    data_a, labs_a, sids_a, _ = load_cohort(MATRIX_DIR_A, label_a)
    print("\nLoading Cohort B")
    data_b, labs_b, sids_b, _ = load_cohort(MATRIX_DIR_B, label_b)

    # ── Feature selection — fit strictly on Cohort A ──────────────────────────
    print("\n" + "="*60)
    print(f"Feature selection (TOP_PROBES={TOP_PROBES:,})")
    top_idx_20k = select_probes(data_a, TOP_PROBES, None)
    top_idx_5k  = select_probes(data_a, 5_000,      None)

    # P0.3 — AHRR probe check
    print("\n" + "="*60)
    print("P0.3 — AHRR / cg05575921 probe check")
    # Load probe names from first intersected matrix
    first_file = sorted(f for f in os.listdir(MATRIX_DIR_A) if f.endswith("_intersected.csv.gz"))[0]
    probe_names = pd.read_csv(
        os.path.join(MATRIX_DIR_A, first_file), index_col=0, compression="gzip", nrows=0
    ).index
    # Actually probe names are the row index — need to read just the index
    probe_names_full = pd.read_csv(
        os.path.join(MATRIX_DIR_A, first_file), index_col=0, compression="gzip"
    ).index.tolist()
    n_total = len(probe_names_full)

    AHRR_PROBE = "cg05575921"
    ahrr_in_all = AHRR_PROBE in probe_names_full
    ahrr_pos    = probe_names_full.index(AHRR_PROBE) if ahrr_in_all else None

    ahrr_in_20k = ahrr_pos in set(top_idx_20k.tolist()) if ahrr_pos is not None else False
    ahrr_in_5k  = ahrr_pos in set(top_idx_5k.tolist())  if ahrr_pos is not None else False

    print(f"  cg05575921 in full probe set ({n_total:,}): {ahrr_in_all}")
    if ahrr_pos is not None:
        mean_across = data_a.mean(axis=0)
        probe_var   = mean_across.var(axis=0)
        ahrr_var    = float(probe_var[ahrr_pos])
        # Rank (1=highest variance) among all probes
        ahrr_rank   = int(n_total - np.searchsorted(np.sort(probe_var), ahrr_var))
        print(f"  cg05575921 inter-sample variance rank: {ahrr_rank:,} / {n_total:,}")
        print(f"  cg05575921 in top-20K set: {ahrr_in_20k}")
        print(f"  cg05575921 in top-5K set:  {ahrr_in_5k}")
    else:
        ahrr_var = ahrr_rank = None
        print("  cg05575921 NOT FOUND in probe intersection")

    results["p0_3_ahrr"] = {
        "probe": AHRR_PROBE,
        "in_full_set": bool(ahrr_in_all),
        "position_in_full": ahrr_pos,
        "inter_sample_variance_rank": ahrr_rank,
        "total_probes": n_total,
        "in_top_20k": bool(ahrr_in_20k),
        "in_top_5k": bool(ahrr_in_5k),
        "inter_sample_variance_value": ahrr_var,
    }

    # ── Labeled subsets after feature selection ────────────────────────────────
    # Use mean-across-pipelines as the feature matrix (same as baselines)
    X_a_full = data_a[:, :, top_idx_20k].mean(axis=0)   # (n_a, 20k)
    X_b_full = data_b[:, :, top_idx_20k].mean(axis=0)   # (n_b, 20k)

    mask_a = labs_a >= 0
    mask_b = labs_b >= 0
    X_a, y_a = X_a_full[mask_a], labs_a[mask_a]
    X_b, y_b = X_b_full[mask_b], labs_b[mask_b]
    print(f"\nLabeled Cohort A: {X_a.shape}  SM={y_a.sum()}  NS={(y_a==0).sum()}")
    print(f"Labeled Cohort B: {X_b.shape}  SM={y_b.sum()}  NS={(y_b==0).sum()}")

    # ── Load trained encoder ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Loading MethCLR checkpoint")
    ckpt       = torch.load(CKPT_PATH, map_location="cpu")
    input_dim  = ckpt["input_dim"]
    ckpt_epoch = ckpt.get("epoch", None)
    ckpt_auc   = ckpt.get("auc", None)
    print(f"  epoch={ckpt_epoch} | stored AUC={ckpt_auc:.4f} | input_dim={input_dim:,}")

    encoder = MLPEncoder(input_dim=input_dim, dropout=0.0)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    # ── P0.4 — Cohort A MethCLR 5-fold CV with per-fold AUCs ─────────────────
    print("\n" + "="*60)
    print("P0.4 — MethCLR Cohort A 5-fold CV")
    data_a_sel = data_a[:, :, top_idx_20k]   # (n_pipes, n_a, 20k)

    emb_a = embed_methclr(data_a_sel[:, mask_a, :], encoder)
    auc_a_mean, auc_a_std, auc_a_folds = cv_probe(emb_a, y_a, "MethCLR/A 5-fold")

    results["p0_4_cohort_a_cv"] = {
        "auc_mean":  auc_a_mean,
        "auc_std":   auc_a_std,
        "auc_folds": auc_a_folds,
    }

    # ── P0.1 — MethCLR label-free transfer ────────────────────────────────────
    print("\n" + "="*60)
    print("P0.1 — MethCLR label-free transfer (fit all A → evaluate all B)")
    data_b_sel = data_b[:, :, top_idx_20k]
    emb_b_labeled = embed_methclr(data_b_sel[:, mask_b, :], encoder)

    auc_methclr_lf = label_free_transfer(emb_a, y_a, emb_b_labeled, y_b, "MethCLR LF")
    results["p0_1_methclr_label_free"] = {"cohort_b_auc": auc_methclr_lf}

    # ── P0.1 — PCA label-free transfer ────────────────────────────────────────
    print("\n" + "="*60)
    print("P0.1 — PCA (128 PCs) label-free transfer")
    scaler_pca = StandardScaler().fit(X_a)
    Xa_s = scaler_pca.transform(X_a)
    Xb_s = scaler_pca.transform(X_b)
    pca  = PCA(n_components=128, random_state=SEED).fit(Xa_s)
    Xa_pca = pca.transform(Xa_s)
    Xb_pca = pca.transform(Xb_s)
    print(f"  [PCA] Cohort A 5-fold CV:")
    auc_pca_a_cv, std_pca_a_cv, folds_pca_a_cv = cv_probe(Xa_pca, y_a, "PCA/A 5-fold")
    auc_pca_lf = label_free_transfer(Xa_pca, y_a, Xb_pca, y_b, "PCA LF")
    results["p0_1_pca_label_free"] = {
        "cohort_a_cv_auc": auc_pca_a_cv,
        "cohort_a_cv_std": std_pca_a_cv,
        "cohort_a_cv_folds": folds_pca_a_cv,
        "cohort_b_label_free_auc": auc_pca_lf,
    }

    # ── P0.1 — Autoencoder label-free transfer ────────────────────────────────
    print("\n" + "="*60)
    print("P0.1 — Autoencoder label-free transfer")
    ae, Xa_ae = train_ae(Xa_s)
    with torch.no_grad():
        Xb_ae = ae.encode(torch.from_numpy(Xb_s.astype(np.float32))).numpy()
    print(f"  [AE] Cohort A 5-fold CV:")
    auc_ae_a_cv, std_ae_a_cv, folds_ae_a_cv = cv_probe(Xa_ae, y_a, "AE/A 5-fold")
    auc_ae_lf = label_free_transfer(Xa_ae, y_a, Xb_ae, y_b, "AE LF")
    results["p0_1_ae_label_free"] = {
        "cohort_a_cv_auc": auc_ae_a_cv,
        "cohort_a_cv_std": std_ae_a_cv,
        "cohort_a_cv_folds": folds_ae_a_cv,
        "cohort_b_label_free_auc": auc_ae_lf,
    }

    # ── P0.2 — Random-initialised encoder ────────────────────────────────────
    print("\n" + "="*60)
    print("P0.2 — Random-initialised encoder (untrained)")
    torch.manual_seed(SEED)
    rand_enc = MLPEncoder(input_dim=input_dim, dropout=0.0)
    rand_enc.eval()

    emb_a_rand = embed_methclr(data_a_sel[:, mask_a, :], rand_enc)
    emb_b_rand = embed_methclr(data_b_sel[:, mask_b, :], rand_enc)

    print("  [RandInit] Cohort A 5-fold CV:")
    auc_rand_a_cv, std_rand_a_cv, folds_rand_a_cv = cv_probe(emb_a_rand, y_a, "RandInit/A")
    auc_rand_lf = label_free_transfer(emb_a_rand, y_a, emb_b_rand, y_b, "RandInit LF")

    results["p0_2_random_init"] = {
        "cohort_a_cv_auc":        auc_rand_a_cv,
        "cohort_a_cv_std":        std_rand_a_cv,
        "cohort_a_cv_folds":      folds_rand_a_cv,
        "cohort_b_label_free_auc": auc_rand_lf,
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print(f"{'Method':<30} {'A (5-fold CV)':>18} {'B (label-free)':>16}")
    print("-"*70)
    def _f(m, s): return f"{m:.4f} ± {s:.4f}"
    def _lf(v):   return f"{v:.4f}"
    print(f"{'MethCLR (trained)':<30} {_f(auc_a_mean, auc_a_std):>18} {_lf(auc_methclr_lf):>16}")
    print(f"{'MethCLR (random init)':<30} {_f(auc_rand_a_cv, std_rand_a_cv):>18} {_lf(auc_rand_lf):>16}")
    print(f"{'PCA (128 PCs) + LR':<30} {_f(auc_pca_a_cv, std_pca_a_cv):>18} {_lf(auc_pca_lf):>16}")
    print(f"{'Autoencoder (128) + LR':<30} {_f(auc_ae_a_cv, std_ae_a_cv):>18} {_lf(auc_ae_lf):>16}")
    print("="*70)
    print(f"\ncg05575921 (AHRR) in top-20K: {ahrr_in_20k} | in top-5K: {ahrr_in_5k}")
    if ahrr_rank is not None:
        print(f"  Variance rank: {ahrr_rank:,} / {n_total:,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "p0_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
