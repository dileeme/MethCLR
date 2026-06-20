"""
MethCLR — Dataset

Loads all intersected beta matrices from Google Drive, re-links smoking
labels from the series matrix, and serves (anchor, positive, label) tuples
where anchor and positive are the same biological sample processed through
two randomly sampled distinct pipelines.

Expected directory layout on Google Drive:
    /MyDrive/MethCLR/
        beta_matrices/
            norm=sesame_snp=True_sex=True_intersected.csv.gz
            norm=sesame_snp=True_sex=False_intersected.csv.gz
            ... (12 files total)
        GSE85210_series_matrix.txt.gz
"""

import os, gzip, re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────
# Label loading
# ─────────────────────────────────────────────

def load_smoking_labels(series_matrix_path: str) -> dict:
    """
    Parses GSE85210 series matrix to build Sentrix_ID+Position → smoking label map.
    Returns dict: '{Sentrix_ID}_{Sentrix_Position}' → int (0=non-smoker, 1=smoker)
    Falls back to empty dict if parsing fails — contrastive training still works,
    linear probe evaluation will be skipped.
    """
    samples, labels, supp_files = [], [], []
    opener = gzip.open if series_matrix_path.endswith(".gz") else open
    try:
        with opener(series_matrix_path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("!Sample_geo_accession"):
                    samples = [p.strip('"') for p in line.split("\t")[1:]]
                if line.startswith("!Sample_characteristics_ch1") and "subject status" in line:
                    labels = [p.strip('"') for p in line.split("\t")[1:]]
                if line.startswith("!Sample_supplementary_file"):
                    supp_files.append([p.strip('"') for p in line.split("\t")[1:]])
    except Exception as e:
        print(f"[WARN] Could not read series matrix: {e}")
        return {}

    if not samples or not labels:
        print("[WARN] Could not parse samples/labels from series matrix")
        return {}

    smoking = [0 if "non" in lb.lower() else 1 for lb in labels]

    # Parse Sentrix IDs from supplementary file FTP URLs
    sentrix_keys = []
    grn_files = []
    for row in supp_files:
        matches = [x for x in row if "Grn.idat" in x]
        if matches:
            grn_files = matches  # entire Grn row — 253 URLs
            break

    if len(grn_files) != len(samples):
        print("[WARN] Could not parse Sentrix IDs from series matrix — smoking labels unavailable")
        return {}

    for url in grn_files:
        fname = url.split("/")[-1]
        parts = fname.split("_")
        if len(parts) >= 3:
            sentrix_keys.append(f"{parts[1]}_{parts[2]}")
        else:
            sentrix_keys.append(None)

    label_map = {
        k: v for k, v in zip(sentrix_keys, smoking) if k is not None
    }
    print(f"[LABELS] Linked {len(label_map)} samples | "
          f"Smokers: {sum(label_map.values())} | "
          f"Non-smokers: {sum(1 for v in label_map.values() if v == 0)}")
    return label_map


# ─────────────────────────────────────────────
# Matrix loading
# ─────────────────────────────────────────────

def load_intersected_matrices(matrix_dir: str) -> dict:
    """
    Loads all 12 *_intersected.csv.gz files.
    Returns dict: pipeline_id → DataFrame (probes × samples)
    """
    files = sorted([
        f for f in os.listdir(matrix_dir)
        if f.endswith("_intersected.csv.gz")
    ])
    if not files:
        raise FileNotFoundError(f"No *_intersected.csv.gz files found in {matrix_dir}")

    matrices = {}
    for fname in files:
        pid = fname.replace("_intersected.csv.gz", "")
        fpath = os.path.join(matrix_dir, fname)
        print(f"  Loading {fname} ...", end=" ", flush=True)
        df = pd.read_csv(fpath, index_col=0, compression="gzip")
        matrices[pid] = df
        print(f"{df.shape}")

    print(f"\n[DATA] {len(matrices)} pipelines loaded | "
          f"Probes: {next(iter(matrices.values())).shape[0]:,} | "
          f"Samples: {next(iter(matrices.values())).shape[1]}")
    return matrices


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class MethCLRDataset(Dataset):
    def __init__(self, matrices: dict, label_map: dict, top_probes: int = 5000):
        """
        matrices:   dict of pipeline_id → DataFrame (probes × samples)
        label_map:  dict of '{Sentrix_ID}_{Sentrix_Position}' → int (0/1)
                    May be empty — labels set to -1 if unavailable.
        top_probes: number of highest-variance CpGs to retain (feature selection).
                    Reduces input from 132,337 → top_probes before encoding.
                    Variance computed on mean-across-pipelines to capture biological
                    variation between samples, not pipeline noise.
        """
        self.pipeline_ids = list(matrices.keys())
        n_pipelines = len(self.pipeline_ids)

        # Align sample columns across all pipelines
        all_cols = [set(df.columns) for df in matrices.values()]
        shared_samples = sorted(set.intersection(*all_cols))
        if not shared_samples:
            raise ValueError("No shared sample columns across pipeline matrices")

        self.sample_ids = shared_samples
        n_samples = len(self.sample_ids)

        # Stack into (n_pipelines, n_samples, n_probes) float32 array
        print(f"[DATA] Stacking {n_pipelines} × {n_samples} samples into memory ...")
        self.data = np.stack([
            matrices[pid][self.sample_ids].T.values.astype(np.float32)
            for pid in self.pipeline_ids
        ], axis=0)  # (n_pipelines, n_samples, n_probes)

        # Replace NaN with 0.5 (midpoint beta value)
        nan_count = np.isnan(self.data).sum()
        if nan_count > 0:
            print(f"[DATA] Imputing {nan_count:,} NaN values with 0.5")
            self.data = np.nan_to_num(self.data, nan=0.5)

        # Feature selection — top-N CpGs by inter-sample variance
        # Average across pipelines first to isolate biological variation,
        # then rank probes by how much they vary across samples.
        if top_probes is not None and top_probes < self.data.shape[2]:
            mean_across_pipelines = self.data.mean(axis=0)        # (n_samples, n_probes)
            probe_variance = mean_across_pipelines.var(axis=0)    # (n_probes,)
            top_idx = np.argsort(probe_variance)[-top_probes:]
            top_idx = np.sort(top_idx)                            # preserve probe order
            self.data = self.data[:, :, top_idx]
            print(f"[FEAT] Selected top {top_probes:,} probes by inter-sample variance "
                  f"(from {probe_variance.shape[0]:,})")

        # Attach smoking labels
        self.labels = np.array([
            label_map.get(sid, -1) for sid in self.sample_ids
        ], dtype=np.int64)

        labeled = (self.labels >= 0).sum()
        print(f"[DATA] Samples with smoking labels: {labeled}/{n_samples}")
        print(f"[DATA] Input dimension: {self.data.shape[2]:,} probes")
        print(f"[DATA] Stack shape: {self.data.shape}")

    @property
    def input_dim(self) -> int:
        return self.data.shape[2]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        # Sample two distinct pipelines for positive pair
        p1, p2 = np.random.choice(len(self.pipeline_ids), size=2, replace=False)
        anchor   = torch.from_numpy(self.data[p1, idx])
        positive = torch.from_numpy(self.data[p2, idx])
        label    = torch.tensor(self.labels[idx], dtype=torch.long)
        return anchor, positive, label


def get_labeled_embeddings(dataset: MethCLRDataset, encoder, device: torch.device):
    """
    Returns (embeddings, labels) for all samples that have a known smoking label.
    Uses all pipelines averaged — stable representation for linear probe eval.
    """
    encoder.eval()
    embeddings, labels = [], []

    for i, sid in enumerate(dataset.sample_ids):
        if dataset.labels[i] < 0:
            continue
        # Average embedding across all pipelines for this sample
        vecs = torch.from_numpy(dataset.data[:, i, :]).to(device)  # (n_pipelines, input_dim)
        with torch.no_grad():
            hs = []
            for v in vecs:
                h, _ = encoder(v.unsqueeze(0))
                hs.append(h.squeeze(0).cpu().numpy())
        embeddings.append(np.mean(hs, axis=0))
        labels.append(dataset.labels[i])

    return np.array(embeddings), np.array(labels)
