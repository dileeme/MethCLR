"""
MethCLR — Supplementary Analyses

1. Pipeline Separability Analysis
   Can a classifier identify which pipeline produced an embedding?
   Low AUC = pipelines are indistinguishable = MethCLR is working.
   Compared against raw beta vectors (baseline should be highly separable).

2. Silhouette Score + Trustworthiness
   Quantifies cluster quality in 128-dim embedding space and UMAP faithfulness.

3. Probe Retention Statistics
   Genomic annotation of the 20,000 selected CpGs from the Cohort B
   variance-based feature selection: chromosome distribution, relation to
   CpG islands, and gene region (TSS, body, intergenic).

Run:
    python eval/supplementary_analyses.py
"""

import os, sys, gzip
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
import torch

_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.join(_HERE, "..")
MATRIX_DIR = os.path.join(_ROOT, "cohort_b_beta_matrices")
SERIES_MAT = os.path.join(os.path.expanduser("~"), "Downloads",
                          "GSE224807-GPL13534_series_matrix.txt.gz")
MANIFEST   = os.path.join(_ROOT, "cohort_b",
                          "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz")
PROBE_INT  = os.path.join(_ROOT, "cohort_b_beta_matrices", "probe_intersection_cohortb.txt")
EMB_PATH   = os.path.join(_HERE, "cohort_b_embeddings.npy")
LABELS_PATH= os.path.join(_HERE, "cohort_b_labels.npy")
OUT_DIR    = _HERE
TOP_PROBES = 20000
SEED       = 42
np.random.seed(SEED)

results = {}

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_labels(path):
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
    sentrix_keys = []
    for url in grn_files:
        fname = url.split("/")[-1]
        parts = fname.split("_")
        sentrix_keys.append(f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else None)
    label_map = {}
    for key, status in zip(sentrix_keys, smoking):
        if key is None:
            continue
        s = status.lower()
        if "sm" in s and "ns" not in s:
            label_map[key] = 1
        elif "ns" in s:
            label_map[key] = 0
    return label_map

print("Loading matrices ...")
files = sorted([f for f in os.listdir(MATRIX_DIR) if f.endswith("_intersected.csv.gz")])
matrices = {}
for fname in files:
    pid = fname.replace("_intersected.csv.gz", "")
    df = pd.read_csv(os.path.join(MATRIX_DIR, fname), index_col=0, compression="gzip")
    matrices[pid] = df
    print(f"  {fname} {df.shape}")

pipeline_ids   = list(matrices.keys())
all_cols       = [set(df.columns) for df in matrices.values()]
shared_samples = sorted(set.intersection(*all_cols))
label_map      = load_labels(SERIES_MAT)

data = np.stack([
    matrices[pid][shared_samples].T.values.astype(np.float32)
    for pid in pipeline_ids
], axis=0)
data = np.nan_to_num(data, nan=0.5)

mean_across = data.mean(axis=0)
probe_var   = mean_across.var(axis=0)
top_idx     = np.sort(np.argsort(probe_var)[-TOP_PROBES:])
data_sel    = data[:, :, top_idx]   # (12, 789, 20000)

labels_arr  = np.array([label_map.get(sid, -1) for sid in shared_samples], dtype=np.int64)
mask        = labels_arr >= 0

# Load saved MethCLR embeddings
embeddings  = np.load(EMB_PATH)   # (789, 128)
y_smoking   = np.load(LABELS_PATH)[mask]
X_embed     = embeddings[mask]

print(f"\nData ready — {mask.sum()} labeled samples")

# ═════════════════════════════════════════════════════════════════
# ANALYSIS 1 — PIPELINE SEPARABILITY
# ═════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Analysis 1: Pipeline Separability")
print("=" * 60)
print("Can a classifier tell which pipeline produced an embedding?")
print("MethCLR goal: pipeline labels should be UNPREDICTABLE (AUC ~ 0.5)")

n_pipelines = len(pipeline_ids)
n_samples   = data_sel.shape[1]

# Build pipeline-labeled dataset:
# Each sample appears 12 times (once per pipeline), labeled by pipeline index
X_raw_flat   = data_sel.reshape(n_pipelines * n_samples, TOP_PROBES)   # raw betas, all pipelines
X_embed_flat = np.vstack([embeddings] * n_pipelines)                    # same embedding 12x
y_pipeline   = np.repeat(np.arange(n_pipelines), n_samples)

scaler_raw = StandardScaler()
X_raw_std  = scaler_raw.fit_transform(X_raw_flat)

scaler_emb = StandardScaler()
X_emb_std  = scaler_emb.fit_transform(X_embed_flat)

cv_pipe = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
clf_ovo = LogisticRegression(max_iter=500, random_state=SEED, C=1.0,
                              multi_class="ovr")

# Raw beta pipeline separability
scores_raw_pipe = cross_val_score(clf_ovo, X_raw_std, y_pipeline,
                                   cv=cv_pipe, scoring="roc_auc_ovr_weighted")
auc_raw_pipe = float(scores_raw_pipe.mean())
print(f"\nRaw Beta pipeline separability AUC:     {auc_raw_pipe:.4f} (higher = more separable)")

# MethCLR embedding pipeline separability
scores_emb_pipe = cross_val_score(clf_ovo, X_emb_std, y_pipeline,
                                   cv=cv_pipe, scoring="roc_auc_ovr_weighted")
auc_emb_pipe = float(scores_emb_pipe.mean())
print(f"MethCLR embedding pipeline separability: {auc_emb_pipe:.4f} (lower = more invariant)")
print(f"\nInvariance gain: {auc_raw_pipe - auc_emb_pipe:+.4f} reduction in pipeline AUC")

results["pipeline_separability"] = {
    "raw_beta_auc":   auc_raw_pipe,
    "methclr_auc":    auc_emb_pipe,
    "invariance_gain": float(auc_raw_pipe - auc_emb_pipe),
}

# ═════════════════════════════════════════════════════════════════
# ANALYSIS 2 — SILHOUETTE SCORE + TRUSTWORTHINESS
# ═════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Analysis 2: Silhouette Score + Trustworthiness")
print("=" * 60)

# --- Silhouette score on smoking labels in embedding space ---
scaler_s = StandardScaler()
X_emb_s  = scaler_s.fit_transform(X_embed)

sil_methclr = silhouette_score(X_emb_s, y_smoking, random_state=SEED)
print(f"MethCLR silhouette (smoking, 128-dim):  {sil_methclr:.4f}")

# Raw beta silhouette for comparison
X_raw_s  = scaler_raw.fit_transform(data_sel.mean(axis=0)[mask])
sil_raw  = silhouette_score(X_raw_s, y_smoking, random_state=SEED)
print(f"Raw beta silhouette (smoking, 20k-dim): {sil_raw:.4f}")

# --- Trustworthiness of UMAP projection ---
# Load UMAP coords computed in plot_umap.py or recompute
try:
    import umap
    print("\nComputing UMAP for trustworthiness metric ...")
    reducer = umap.UMAP(n_components=2, random_state=SEED,
                        n_neighbors=15, min_dist=0.1)
    X_umap  = reducer.fit_transform(X_emb_s)

    # Trustworthiness at multiple neighbourhood sizes
    trust_scores = {}
    for k in [5, 10, 20]:
        t = trustworthiness(X_emb_s, X_umap, n_neighbors=k)
        trust_scores[k] = float(t)
        print(f"  Trustworthiness (k={k:2d}): {t:.4f}")

    results["trustworthiness"] = trust_scores
    np.save(os.path.join(OUT_DIR, "umap_coords.npy"), X_umap)
except ImportError:
    print("[WARN] umap-learn not installed — skipping trustworthiness")
    trust_scores = {}

results["silhouette"] = {
    "methclr_128dim": float(sil_methclr),
    "raw_beta_20k":   float(sil_raw),
}
print(f"\nSilhouette improvement: {sil_methclr - sil_raw:+.4f}")

# ═════════════════════════════════════════════════════════════════
# ANALYSIS 3 — PROBE RETENTION STATISTICS
# ═════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Analysis 3: Probe Retention Statistics")
print("=" * 60)

# Get all probe names from intersection
all_probes = list(next(iter(matrices.values())).index)
selected_probes = [all_probes[i] for i in top_idx]
print(f"Total intersected probes: {len(all_probes):,}")
print(f"Selected probes:          {len(selected_probes):,} ({100*len(selected_probes)/len(all_probes):.1f}%)")

results["probe_retention"] = {
    "total_intersected": len(all_probes),
    "selected":          len(selected_probes),
    "pct_retained":      round(100 * len(selected_probes) / len(all_probes), 2),
}

# Try to annotate from manifest
manifest_loaded = False
if os.path.exists(MANIFEST):
    print("\nLoading 450K manifest for probe annotation ...")
    try:
        # The Illumina manifest CSV has a multi-line header — skip to data
        manifest = pd.read_csv(MANIFEST, compression="gzip", comment="[",
                               low_memory=False, index_col=0,
                               skiprows=7)   # skip Illumina header block
        manifest.index = manifest.index.astype(str)
        manifest_loaded = True
        print(f"Manifest loaded: {manifest.shape[0]:,} probes")
    except Exception as e:
        print(f"[WARN] Could not load manifest: {e}")

if manifest_loaded:
    sel_set = set(selected_probes)
    all_set = set(all_probes)

    def get_dist(probe_set, col):
        available = [p for p in probe_set if p in manifest.index and
                     pd.notna(manifest.loc[p, col])]
        vals = manifest.loc[available, col]
        return vals.value_counts().to_dict()

    # Chromosome distribution
    if "CHR" in manifest.columns:
        chr_dist_all = get_dist(all_set,  "CHR")
        chr_dist_sel = get_dist(sel_set,  "CHR")
        print("\nChromosome distribution (selected vs all):")
        chrs = sorted(set(chr_dist_all) | set(chr_dist_sel),
                      key=lambda x: (len(x), x))
        for c in chrs:
            n_all = chr_dist_all.get(c, 0)
            n_sel = chr_dist_sel.get(c, 0)
            pct   = 100 * n_sel / n_all if n_all > 0 else 0
            print(f"  chr{c:3s}: {n_sel:5d}/{n_all:6d} selected ({pct:.1f}%)")
        results["probe_retention"]["chr_distribution_selected"] = {
            str(k): v for k, v in chr_dist_sel.items()
        }

    # CpG island relation
    if "Relation_to_UCSC_CpG_Island" in manifest.columns:
        cpgi_col = "Relation_to_UCSC_CpG_Island"
        cpgi_all = get_dist(all_set, cpgi_col)
        cpgi_sel = get_dist(sel_set, cpgi_col)
        print("\nCpG island context (selected):")
        for region, count in sorted(cpgi_sel.items(), key=lambda x: -x[1]):
            pct_of_sel = 100 * count / len(selected_probes)
            pct_of_cat = 100 * count / cpgi_all.get(region, count)
            print(f"  {region:<25s}: {count:5d} ({pct_of_sel:.1f}% of selected, "
                  f"{pct_of_cat:.1f}% of category)")
        results["probe_retention"]["cpg_island_context"] = {
            str(k): v for k, v in cpgi_sel.items()
        }

    # Gene region (UCSC RefGene Group)
    if "UCSC_RefGene_Group" in manifest.columns:
        def first_region(val):
            if pd.isna(val):
                return "Intergenic"
            return str(val).split(";")[0]

        available_sel = [p for p in selected_probes if p in manifest.index]
        region_vals   = manifest.loc[available_sel, "UCSC_RefGene_Group"].apply(first_region)
        region_dist   = region_vals.value_counts().to_dict()
        print("\nGene region distribution (selected probes):")
        for region, count in sorted(region_dist.items(), key=lambda x: -x[1]):
            print(f"  {region:<20s}: {count:5d} ({100*count/len(available_sel):.1f}%)")
        results["probe_retention"]["gene_region_distribution"] = {
            str(k): v for k, v in region_dist.items()
        }

        # Plot probe annotation pie charts
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if "cpg_island_context" in results["probe_retention"]:
            cpgi_data = results["probe_retention"]["cpg_island_context"]
            axes[0].pie(cpgi_data.values(), labels=cpgi_data.keys(),
                        autopct="%1.1f%%", startangle=140)
            axes[0].set_title(f"CpG Island Context\n(Top {TOP_PROBES:,} selected probes)")

        axes[1].pie(region_dist.values(), labels=region_dist.keys(),
                    autopct="%1.1f%%", startangle=140)
        axes[1].set_title(f"Gene Region Distribution\n(Top {TOP_PROBES:,} selected probes)")

        plt.tight_layout()
        fig_path = os.path.join(OUT_DIR, "probe_annotation.png")
        plt.savefig(fig_path, dpi=150)
        print(f"\nProbe annotation figure saved → {fig_path}")
else:
    print("[INFO] Manifest not available — skipping genomic annotation")
    print(f"       Place GPL13534 manifest CSV in {MANIFEST} for annotation")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Pipeline separability  — Raw: {auc_raw_pipe:.4f} | MethCLR: {auc_emb_pipe:.4f}  "
      f"(Δ={auc_raw_pipe-auc_emb_pipe:+.4f})")
print(f"Silhouette (smoking)   — Raw: {sil_raw:.4f} | MethCLR: {sil_methclr:.4f}  "
      f"(Δ={sil_methclr-sil_raw:+.4f})")
if trust_scores:
    print(f"Trustworthiness (k=10) — {trust_scores.get(10, 'N/A'):.4f}")
print(f"Probe retention        — {len(selected_probes):,}/{len(all_probes):,} "
      f"({100*len(selected_probes)/len(all_probes):.1f}%)")

out_path = os.path.join(OUT_DIR, "supplementary_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAll results saved → {out_path}")
