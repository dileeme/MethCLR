"""
MethCLR — run_baselines.py
Unsupervised baseline evaluation suite: PCA and Autoencoder

Protocol (identical for both baselines):
  1. Feature selection: top 20,000 CpGs by inter-sample variance on Cohort A mean-across-
     pipelines — probe indices fixed on A, applied to B (no data leakage).
  2. Fit unsupervised model (PCA / AE) strictly on Cohort A.
  3. Cohort A in-cohort AUC: 5-fold stratified CV with logistic regression probe.
  4. Cohort B zero-shot AUC: encode B with frozen model, 5-fold stratified CV LR probe
     (same protocol as eval_cohort_b.py used for MethCLR).

Results saved to eval/baseline_results.json (overwrites prior run).
"""

import os
import sys
import gzip
import json
import multiprocessing

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# ── Threading — set before any torch parallel ops ───────────────────────────
try:
    import psutil
    N_CORES = psutil.cpu_count(logical=False) or os.cpu_count()
except ImportError:
    N_CORES = os.cpu_count()

torch.set_num_threads(N_CORES)
torch.set_num_interop_threads(N_CORES)

# num_workers > 0 is safe on Windows only when the script runs under __main__.
# We set it here; all DataLoader construction happens inside main() which is
# called exclusively from the __main__ guard below.
NUM_WORKERS = min(4, N_CORES)

# ── Configuration ────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
MATRIX_DIR_A = os.path.join(_HERE, "beta_matrices")
MATRIX_DIR_B = os.path.join(_HERE, "cohort_b_beta_matrices")
SERIES_MAT_A = os.path.join(os.path.expanduser("~"), "Downloads",
                             "GSE85210_series_matrix.txt.gz")
SERIES_MAT_B = os.path.join(os.path.expanduser("~"), "Downloads",
                             "GSE224807-GPL13534_series_matrix.txt.gz")
OUT_DIR      = os.path.join(_HERE, "eval")
os.makedirs(OUT_DIR, exist_ok=True)

TOP_PROBES = 20_000
EMBED_DIM  = 128
AE_EPOCHS  = 100
AE_BATCH   = 64
AE_LR      = 1e-3
N_FOLDS    = 5
SEED       = 42

METHCLR_AUC_A = 0.8447
METHCLR_AUC_B = 0.7764
METHCLR_STD_B = 0.0128

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Label loaders ─────────────────────────────────────────────────────────────

def _load_labels(path: str, smoking_field: str) -> dict:
    """
    Generic GEO series-matrix label parser.
    smoking_field: substring that identifies the phenotype row
                   ('subject status' for GSE85210, 'smoking_status' for GSE224807)
    Returns dict: '{Sentrix_ID}_{Sentrix_Position}' -> int (0=NS, 1=SM)
    """
    if not os.path.exists(path):
        print(f"[WARN] Series matrix not found: {path}")
        return {}
    samples, phenotype, supp_files = [], [], []
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("!Sample_geo_accession"):
                    samples = [p.strip('"') for p in line.split("\t")[1:]]
                if line.startswith("!Sample_characteristics_ch1") and smoking_field in line:
                    phenotype = [p.strip('"') for p in line.split("\t")[1:]]
                if line.startswith("!Sample_supplementary_file"):
                    supp_files.append([p.strip('"') for p in line.split("\t")[1:]])
    except Exception as exc:
        print(f"[WARN] Could not read series matrix: {exc}")
        return {}

    if not samples or not phenotype:
        return {}

    grn_files = []
    for row in supp_files:
        matches = [x for x in row if "Grn.idat" in x]
        if matches:
            grn_files = matches
            break

    if len(grn_files) != len(samples):
        return {}

    label_map = {}
    for url, status in zip(grn_files, phenotype):
        fname = url.split("/")[-1]
        parts = fname.split("_")
        if len(parts) < 3:
            continue
        key = f"{parts[1]}_{parts[2]}"
        s = status.lower()
        if "non" in s or "ns" in s:
            label_map[key] = 0
        else:
            label_map[key] = 1
    return label_map


def load_labels_a(path: str) -> dict:
    return _load_labels(path, "subject status")


def load_labels_b(path: str) -> dict:
    return _load_labels(path, "smoking_status")


# ── Data loader ───────────────────────────────────────────────────────────────

def load_cohort(matrix_dir: str, label_map: dict):
    """
    Loads all *_intersected.csv.gz files.
    Returns (data, labels, sample_ids)
      data:       (n_pipelines, n_samples, n_probes)  float32
      labels:     (n_samples,)  int64  (-1 = no label)
      sample_ids: list[str]
    """
    files = sorted(f for f in os.listdir(matrix_dir) if f.endswith("_intersected.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No *_intersected.csv.gz in {matrix_dir}")

    matrices = {}
    for fname in files:
        pid = fname.replace("_intersected.csv.gz", "")
        df  = pd.read_csv(os.path.join(matrix_dir, fname), index_col=0, compression="gzip")
        matrices[pid] = df
        print(f"  {fname}: {df.shape}", flush=True)

    pid_list       = list(matrices.keys())
    shared_samples = sorted(set.intersection(*[set(df.columns) for df in matrices.values()]))
    data = np.stack(
        [matrices[pid][shared_samples].T.values.astype(np.float32) for pid in pid_list],
        axis=0,
    )
    data = np.nan_to_num(data, nan=0.5)

    labels = np.array([label_map.get(sid, -1) for sid in shared_samples], dtype=np.int64)
    print(f"  → {len(pid_list)} pipelines | {len(shared_samples)} samples | "
          f"labeled: {(labels >= 0).sum()}")
    return data, labels, shared_samples


# ── Linear probe (5-fold stratified CV) ──────────────────────────────────────

def linear_probe(X: np.ndarray, y: np.ndarray, label: str) -> tuple[float, float, list]:
    """
    Standardizes X, runs 5-fold stratified CV logistic regression, returns
    (mean_auc, std_auc, per_fold_aucs).
    n_jobs=-1 parallelises across all available CPU threads.
    """
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    cv     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    clf    = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    scores = cross_val_score(clf, X_s, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  [{label}] AUC folds: {' | '.join(f'{s:.4f}' for s in scores)}")
    print(f"  [{label}] Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")
    return float(scores.mean()), float(scores.std()), scores.tolist()


# ── Autoencoder architecture (exact spec) ─────────────────────────────────────

class MethAE(nn.Module):
    """
    Symmetric MLP autoencoder.
    Encoder: 20k → 512 → BN → ReLU → 256 → BN → ReLU → 128 (bottleneck)
    Decoder: 128 → 256 → BN → ReLU → 512 → BN → ReLU → 20k
    MSE reconstruction loss; downstream eval uses bottleneck (128-dim).
    """
    def __init__(self, input_dim: int, bottleneck: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z   = self.encoder(x)
        out = self.decoder(z)
        return out, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(x)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n[HW] Physical cores : {N_CORES}")
    print(f"[HW] torch threads  : {torch.get_num_threads()}")
    print(f"[HW] interop threads: {torch.get_num_interop_threads()}")
    print(f"[HW] DataLoader workers: {NUM_WORKERS}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HW] Compute device : {device}\n")

    # ── Load labels ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading labels")
    print("=" * 60)
    label_map_a = load_labels_a(SERIES_MAT_A)
    label_map_b = load_labels_b(SERIES_MAT_B)
    print(f"Cohort A labels: {len(label_map_a)} | "
          f"Smokers: {sum(label_map_a.values())} | "
          f"Non-smokers: {sum(1 for v in label_map_a.values() if v == 0)}")
    print(f"Cohort B labels: {len(label_map_b)} | "
          f"Smokers: {sum(label_map_b.values())} | "
          f"Non-smokers: {sum(1 for v in label_map_b.values() if v == 0)}")

    # ── Load Cohort A ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading Cohort A (GSE85210)")
    print("=" * 60)
    data_a, labels_a, samples_a = load_cohort(MATRIX_DIR_A, label_map_a)

    # ── Load Cohort B (optional) ─────────────────────────────────────────────
    has_cohort_b = os.path.isdir(MATRIX_DIR_B) and any(
        f.endswith("_intersected.csv.gz") for f in os.listdir(MATRIX_DIR_B)
    ) if os.path.isdir(MATRIX_DIR_B) else False

    if has_cohort_b:
        print("\n" + "=" * 60)
        print("Loading Cohort B (GSE224807)")
        print("=" * 60)
        data_b, labels_b, samples_b = load_cohort(MATRIX_DIR_B, label_map_b)
    else:
        print(f"\n[INFO] {MATRIX_DIR_B} not found — skipping cross-cohort evaluation.")
        print("[INFO] Run on Colab with cohort_b_beta_matrices/ present for full results.")
        data_b = labels_b = samples_b = None

    # ── Feature selection — fit on Cohort A only ─────────────────────────────
    print(f"\n[FEAT] Selecting top {TOP_PROBES:,} CpGs by Cohort A inter-sample variance ...")
    mean_a     = data_a.mean(axis=0)                       # (n_samples_a, n_probes)
    probe_var  = mean_a.var(axis=0)                        # (n_probes,)
    top_idx    = np.sort(np.argsort(probe_var)[-TOP_PROBES:])

    # Mean-across-pipelines feature matrix for each cohort
    X_a_full = data_a[:, :, top_idx].mean(axis=0)         # (n_a, 20k)
    mask_a   = labels_a >= 0
    X_a      = X_a_full[mask_a]
    y_a      = labels_a[mask_a]
    print(f"[FEAT] Cohort A: {X_a.shape} | SM={y_a.sum()} NS={(y_a==0).sum()}")

    if data_b is not None:
        X_b_full = data_b[:, :, top_idx].mean(axis=0)     # (n_b, 20k)
        mask_b   = labels_b >= 0
        X_b      = X_b_full[mask_b]
        y_b      = labels_b[mask_b]
        print(f"[FEAT] Cohort B: {X_b.shape} | SM={y_b.sum()} NS={(y_b==0).sum()}")

    # Scaler fit strictly on Cohort A
    scaler   = StandardScaler().fit(X_a)
    X_a_s    = scaler.transform(X_a)
    if data_b is not None:
        X_b_s = scaler.transform(X_b)

    results = {}

    # ── BASELINE 1: PCA ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Baseline 1: PCA ({EMBED_DIM} PCs) + LR")
    print("=" * 60)

    pca     = PCA(n_components=EMBED_DIM, random_state=SEED)
    X_a_pca = pca.fit_transform(X_a_s)                    # fit+transform on A
    var_exp = float(pca.explained_variance_ratio_.sum())
    print(f"[PCA] Variance explained (Cohort A): {var_exp:.4f}")

    print("\n[PCA] Cohort A — 5-fold CV linear probe:")
    auc_pca_a, std_pca_a, folds_pca_a = linear_probe(X_a_pca, y_a, "PCA / Cohort A")

    if data_b is not None:
        X_b_pca = pca.transform(X_b_s)                    # frozen — no refit
        print("\n[PCA] Cohort B — 5-fold CV linear probe (zero-shot):")
        auc_pca_b, std_pca_b, folds_pca_b = linear_probe(X_b_pca, y_b, "PCA / Cohort B")
    else:
        auc_pca_b = std_pca_b = folds_pca_b = None

    results["pca_lr"] = {
        "cohort_a_auc": auc_pca_a, "cohort_a_std": std_pca_a, "cohort_a_folds": folds_pca_a,
        "cohort_b_auc": auc_pca_b, "cohort_b_std": std_pca_b, "cohort_b_folds": folds_pca_b,
        "variance_explained": var_exp,
        "n_components": EMBED_DIM,
    }

    # ── BASELINE 2: Autoencoder ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Baseline 2: Autoencoder ({EMBED_DIM}-dim bottleneck) + LR")
    print("=" * 60)

    X_a_t  = torch.from_numpy(X_a_s.astype(np.float32))
    dataset_ae = TensorDataset(X_a_t)
    loader_ae  = DataLoader(
        dataset_ae,
        batch_size=AE_BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    ae  = MethAE(input_dim=TOP_PROBES, bottleneck=EMBED_DIM).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=AE_LR, weight_decay=1e-4)
    mse = nn.MSELoss()

    print(f"[AE] Training on Cohort A | {AE_EPOCHS} epochs | batch={AE_BATCH} | device={device}")
    for epoch in range(1, AE_EPOCHS + 1):
        ae.train()
        epoch_losses = []
        for (xb,) in loader_ae:
            xb  = xb.to(device, non_blocking=True)
            out, _ = ae(xb)
            loss = mse(out, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{AE_EPOCHS} | recon_loss={np.mean(epoch_losses):.5f}",
                  flush=True)

    # Extract bottleneck embeddings with frozen encoder
    ae.eval()
    with torch.no_grad():
        X_a_ae = ae.encode(X_a_t.to(device)).cpu().numpy()
        if data_b is not None:
            X_b_t  = torch.from_numpy(X_b_s.astype(np.float32))
            X_b_ae = ae.encode(X_b_t.to(device)).cpu().numpy()

    print("\n[AE] Cohort A — 5-fold CV linear probe:")
    auc_ae_a, std_ae_a, folds_ae_a = linear_probe(X_a_ae, y_a, "AE / Cohort A")

    if data_b is not None:
        print("\n[AE] Cohort B — 5-fold CV linear probe (zero-shot):")
        auc_ae_b, std_ae_b, folds_ae_b = linear_probe(X_b_ae, y_b, "AE / Cohort B")
    else:
        auc_ae_b = std_ae_b = folds_ae_b = None

    results["autoencoder_lr"] = {
        "cohort_a_auc": auc_ae_a, "cohort_a_std": std_ae_a, "cohort_a_folds": folds_ae_a,
        "cohort_b_auc": auc_ae_b, "cohort_b_std": std_ae_b, "cohort_b_folds": folds_ae_b,
        "ae_epochs": AE_EPOCHS, "ae_batch": AE_BATCH, "bottleneck": EMBED_DIM,
    }

    results["methclr"] = {
        "cohort_a_auc": METHCLR_AUC_A, "cohort_a_std": None,
        "cohort_b_auc": METHCLR_AUC_B, "cohort_b_std": METHCLR_STD_B,
    }

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("UNSUPERVISED BASELINE SUMMARY")
    print(f"{'Method':<30} {'Cohort A AUC':>14} {'Cohort B AUC':>14}")
    print("-" * 70)

    def _fmt(mean, std):
        if mean is None:
            return f"{'—':>14}"
        return f"{mean:.4f} ± {std:.4f}".rjust(14)

    print(f"{'PCA (128 PCs) + LR':<30} {_fmt(auc_pca_a, std_pca_a)} {_fmt(auc_pca_b, std_pca_b)}")
    print(f"{'Autoencoder (128-dim) + LR':<30} {_fmt(auc_ae_a,  std_ae_a )} {_fmt(auc_ae_b,  std_ae_b )}")
    print(f"{'MethCLR (ours)':<30} {_fmt(METHCLR_AUC_A, 0.0):<14} {_fmt(METHCLR_AUC_B, METHCLR_STD_B)}")
    print("=" * 70)
    print("Cohort A: 5-fold stratified CV | Cohort B: zero-shot transfer (frozen model)\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "baseline_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # required on Windows for num_workers > 0
    main()
