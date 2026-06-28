"""
MethCLR — Verifiability Checks (Tasks 2 and 3)

Task 2: Exact PCA variance-explained figure, traced to the specific matrix.
Task 3: Random-init encoder Std CV and matched-N CV on Cohort B;
        MethCLR vs RandInit bootstrap significance under Std CV.

Requires:
  eval/rev_cache_*.npy          (from reviewer_analyses.py first run)
  beta_matrices/                (Cohort A intersected matrices)
  GSE85210_series_matrix.txt.gz (Cohort A labels)

Run:
    python eval/verifiability_checks.py
"""

import os, sys, gzip, json, warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.join(_HERE, "..")
MATRIX_DIR_A = os.path.join(_ROOT, "beta_matrices")
SERIES_A     = os.path.join(os.path.expanduser("~"), "Downloads",
                             "GSE85210_series_matrix.txt.gz")
CACHE_DIR    = _HERE

TOP_PROBES = 20_000
SEED       = 42
N_FOLDS    = 5
N_REPS_MN  = 20
N_BOOT     = 10_000
N_MATCH    = None   # set after loading labels
PCA_COMPONENTS = 128

np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CACHED LABELS AND RANDOM-INIT EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

print("Loading cached labels and embeddings ...")
y_a        = np.load(os.path.join(CACHE_DIR, "rev_cache_labels_a.npy"))
y_b        = np.load(os.path.join(CACHE_DIR, "rev_cache_labels_b.npy"))
emb_b_rand = np.load(os.path.join(CACHE_DIR, "rev_cache_emb_b_rand.npy"))
emb_b_mclr = np.load(os.path.join(CACHE_DIR, "rev_cache_emb_b_methclr.npy"))

N_A = len(y_a)
N_B = len(y_b)
N_MATCH = int(N_A * (N_FOLDS - 1) / N_FOLDS)  # 202

print(f"  y_a: {y_a.shape}  y_b: {y_b.shape}")
print(f"  emb_b_rand: {emb_b_rand.shape}")
print(f"  N_A={N_A}  N_B={N_B}  N_MATCH={N_MATCH}")

results = {}

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — PCA VARIANCE EXPLAINED
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("TASK 2 — PCA variance explained (128 components)")
print("="*65)

# Must reload Cohort A raw beta matrices to refit PCA on the exact same matrix.
print("\nLoading Cohort A beta matrices for PCA refit ...")

def _parse_labels_a(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    samples, phenotype, supp_files = [], [], []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("!Sample_geo_accession"):
                samples = [p.strip('"') for p in line.split("\t")[1:]]
            if line.startswith("!Sample_characteristics_ch1") and "subject status" in line:
                phenotype = [p.strip('"') for p in line.split("\t")[1:]]
            if line.startswith("!Sample_supplementary_file"):
                supp_files.append([p.strip('"') for p in line.split("\t")[1:]])
    grn = []
    for row in supp_files:
        hits = [x for x in row if "Grn.idat" in x]
        if hits:
            grn = hits; break
    out = {}
    for url, status in zip(grn, phenotype):
        parts = url.split("/")[-1].split("_")
        if len(parts) >= 3:
            key = f"{parts[1]}_{parts[2]}"
            out[key] = 0 if ("non" in status.lower()) else 1
    return out

label_map_a = _parse_labels_a(SERIES_A)

files_a = sorted(f for f in os.listdir(MATRIX_DIR_A) if f.endswith("_intersected.csv.gz"))
matrices_a = {}
for fname in files_a:
    pid = fname.replace("_intersected.csv.gz", "")
    df  = pd.read_csv(os.path.join(MATRIX_DIR_A, fname), index_col=0, compression="gzip")
    matrices_a[pid] = df
    print(f"  {fname}: {df.shape}", flush=True)

shared_a = sorted(set.intersection(*[set(df.columns) for df in matrices_a.values()]))
data_a   = np.stack(
    [matrices_a[pid][shared_a].T.values.astype(np.float32) for pid in matrices_a],
    axis=0
)  # (n_pipes, n_samples, n_probes)
data_a   = np.nan_to_num(data_a, nan=0.5)

labs_a_full = np.array([label_map_a.get(s, -1) for s in shared_a], dtype=np.int64)
mask_a      = labs_a_full >= 0

# Feature selection — identical to reviewer_analyses.py: fit on Cohort A
mean_a    = data_a.mean(axis=0)          # (n_samples, n_probes)
probe_var = mean_a.var(axis=0)           # (n_probes,)
top_idx   = np.sort(np.argsort(probe_var)[-TOP_PROBES:])
print(f"\n[FEAT] Top-{TOP_PROBES:,} probe indices selected on Cohort A "
      f"(mean-across-pipelines variance). Shape of selecting matrix: {mean_a.shape}")

# Mean-across-pipelines feature matrix for labeled Cohort A samples
X_a_raw = data_a[:, mask_a, :][:, :, top_idx].mean(axis=0)   # (N_A, 20k)
print(f"[PCA input] X_a_raw.shape = {X_a_raw.shape}  "
      f"(Cohort A, labeled only, mean across {data_a.shape[0]} pipelines, top-{TOP_PROBES:,} probes)")

scaler_a = StandardScaler().fit(X_a_raw)
X_a_s    = scaler_a.transform(X_a_raw)
print(f"[PCA input] After StandardScaler: shape = {X_a_s.shape}  "
      f"mean≈{X_a_s.mean():.2e}  std≈{X_a_s.std():.4f}")

# Fit PCA once on full labeled Cohort A (not per fold)
print(f"\nFitting PCA(n_components={PCA_COMPONENTS}, random_state={SEED}) on full Cohort A ...")
pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
pca.fit(X_a_s)

cumvar = float(np.cumsum(pca.explained_variance_ratio_)[-1])
cumvar_check = float(pca.explained_variance_ratio_.sum())

print(f"\n[RESULT Task 2]")
print(f"  np.cumsum(pca.explained_variance_ratio_)[-1] = {cumvar:.6f}")
print(f"  pca.explained_variance_ratio_.sum()          = {cumvar_check:.6f}")
print(f"  => {PCA_COMPONENTS} components explain {cumvar*100:.2f}% of variance in Cohort A")
print(f"  => Paper-reported figure '77.9%' is {'CORRECT' if abs(cumvar - 0.779) < 0.002 else 'INCORRECT — actual value: ' + str(round(cumvar*100,1)) + '%'}")
print(f"\n  Protocol: PCA fit ONCE on full labeled Cohort A (N={mask_a.sum()}, "
      f"{TOP_PROBES:,} features, StandardScaler). NOT per-fold.")

# Per-fold variance (to answer whether it varies across folds)
print(f"\n  Per-fold variance explained ({N_FOLDS}-fold CV, for completeness):")
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_vars = []
for fold_i, (tr_idx, _) in enumerate(cv.split(X_a_s, y_a)):
    pca_fold = PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit(X_a_s[tr_idx])
    fv = float(pca_fold.explained_variance_ratio_.sum())
    fold_vars.append(fv)
    print(f"    Fold {fold_i+1}: {fv:.6f} ({fv*100:.2f}%)")
print(f"  Per-fold mean ± std: {np.mean(fold_vars):.6f} ± {np.std(fold_vars):.6f}  "
      f"({np.mean(fold_vars)*100:.2f}% ± {np.std(fold_vars)*100:.2f}%)")

results["task2_pca_variance"] = {
    "cumulative_variance_explained":      cumvar,
    "pct":                                round(cumvar * 100, 4),
    "matrix":                             f"Cohort A labeled (N={mask_a.sum()}), "
                                          f"mean-across-{data_a.shape[0]}-pipelines, "
                                          f"top-{TOP_PROBES}-probes, StandardScaler",
    "fit_protocol":                       "once on full Cohort A (not per fold)",
    "n_components":                       PCA_COMPONENTS,
    "per_fold_mean":                      float(np.mean(fold_vars)),
    "per_fold_std":                       float(np.std(fold_vars)),
    "paper_figure_correct":               abs(cumvar - 0.779) < 0.002,
}

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — RANDOM-INIT ENCODER: STD CV AND MATCHED-N CV ON COHORT B
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("TASK 3 — Random-init encoder: Cohort B Std CV + matched-N CV")
print("="*65)

def stratified_subsample(y, n_target, rng):
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
        n_c   = min(n_c, len(idx_c))
        selected.extend(rng.choice(idx_c, size=n_c, replace=False).tolist())
    return np.array(selected)


def bootstrap_delta_auc(y, p1, p2, n_boot=N_BOOT, seed=SEED):
    rng     = np.random.RandomState(seed)
    n       = len(y)
    delta_obs = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    deltas  = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    deltas  = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p_val = float(np.mean(deltas <= 0) if delta_obs >= 0 else np.mean(deltas >= 0))
    return float(delta_obs), float(ci_lo), float(ci_hi), p_val


# ── (a) Standard 5-fold CV on Cohort B ────────────────────────────────────

print(f"\n(a) Std CV: 5-fold, full per-fold training pool (N≈{int(N_B*4/5)}) ...")
cv_std = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds_std = list(cv_std.split(np.zeros(N_B), y_b))

rand_preds_std = np.zeros(N_B, dtype=np.float64)
mclr_preds_std = np.zeros(N_B, dtype=np.float64)

for tr_idx, te_idx in folds_std:
    for preds_arr, emb in [(rand_preds_std, emb_b_rand), (mclr_preds_std, emb_b_mclr)]:
        scaler = StandardScaler().fit(emb[tr_idx])
        clf    = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        clf.fit(scaler.transform(emb[tr_idx]), y_b[tr_idx])
        preds_arr[te_idx] = clf.predict_proba(scaler.transform(emb[te_idx]))[:, 1]

# Per-fold AUCs for std and mean
rand_fold_aucs_std = []
mclr_fold_aucs_std = []
for _, te_idx in folds_std:
    y_te = y_b[te_idx]
    if len(np.unique(y_te)) == 2:
        rand_fold_aucs_std.append(roc_auc_score(y_te, rand_preds_std[te_idx]))
        mclr_fold_aucs_std.append(roc_auc_score(y_te, mclr_preds_std[te_idx]))

rand_std_mean = float(np.mean(rand_fold_aucs_std))
rand_std_std  = float(np.std(rand_fold_aucs_std))
mclr_std_mean = float(np.mean(mclr_fold_aucs_std))
mclr_std_std  = float(np.std(mclr_fold_aucs_std))

print(f"  RandInit Std CV AUC: {rand_std_mean:.4f} ± {rand_std_std:.4f}")
print(f"  MethCLR  Std CV AUC: {mclr_std_mean:.4f} ± {mclr_std_std:.4f}  (reference)")
print(f"  Per-fold (RandInit): {' '.join(f'{x:.4f}' for x in rand_fold_aucs_std)}")

# Bootstrap: MethCLR vs RandInit under Std CV
print(f"\n  Bootstrap (10,000 resamples): MethCLR vs RandInit, Std CV ...")
delta_std, lo_std, hi_std, pv_std = bootstrap_delta_auc(
    y_b, mclr_preds_std, rand_preds_std
)
sig_std = "***" if pv_std < 0.001 else ("**" if pv_std < 0.01 else ("*" if pv_std < 0.05 else "ns"))
print(f"  ΔAUC = {delta_std:+.4f}  95% CI [{lo_std:+.4f}, {hi_std:+.4f}]  "
      f"p={pv_std:.4f}  {sig_std}")

results["task3_rand_std_cv"] = {
    "auc_mean":       rand_std_mean,
    "auc_std":        rand_std_std,
    "fold_aucs":      rand_fold_aucs_std,
    "bootstrap_MethCLR_vs_RandInit": {
        "delta_auc": delta_std, "ci_95_lo": lo_std, "ci_95_hi": hi_std,
        "p_value": pv_std, "sig": sig_std,
        "methclr_auc": mclr_std_mean,
        "randint_auc": rand_std_mean,
    },
}

# ── (b) Matched-N CV on Cohort B ──────────────────────────────────────────

print(f"\n(b) Matched-N CV: 5-fold, training pool subsampled to N={N_MATCH}, "
      f"{N_REPS_MN} reps ...")

rep_means_rand_mn = []
for rep in range(N_REPS_MN):
    rep_seed = rep * 1000 + 7   # same seeds as reviewer_analyses.py Task 1
    cv_mn = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=rep_seed)
    fold_aucs = []
    for fold_i, (tr_idx, te_idx) in enumerate(cv_mn.split(np.zeros(N_B), y_b)):
        X_tr, y_tr = emb_b_rand[tr_idx], y_b[tr_idx]
        X_te, y_te = emb_b_rand[te_idx], y_b[te_idx]
        rng_sub    = np.random.RandomState(rep_seed * 13 + fold_i)  # same as Task 1
        sub        = stratified_subsample(y_tr, N_MATCH, rng_sub)
        X_tr_s, y_tr_s = X_tr[sub], y_tr[sub]
        scaler = StandardScaler().fit(X_tr_s)
        clf    = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        clf.fit(scaler.transform(X_tr_s), y_tr_s)
        prob = clf.predict_proba(scaler.transform(X_te))[:, 1]
        if len(np.unique(y_te)) == 2:
            fold_aucs.append(roc_auc_score(y_te, prob))
    rep_means_rand_mn.append(float(np.mean(fold_aucs)))

rand_mn_mean = float(np.mean(rep_means_rand_mn))
rand_mn_std  = float(np.std(rep_means_rand_mn))
print(f"  RandInit matched-N AUC: {rand_mn_mean:.4f} ± {rand_mn_std:.4f}")
print(f"  Rep means: {' '.join(f'{x:.4f}' for x in rep_means_rand_mn)}")

results["task3_rand_matched_n_cv"] = {
    "auc_mean":   rand_mn_mean,
    "auc_std":    rand_mn_std,
    "rep_means":  rep_means_rand_mn,
    "n_reps":     N_REPS_MN,
    "n_match":    N_MATCH,
}

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

out_path = os.path.join(_HERE, "rev_verifiability_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

W = 70
print("\n\n" + "="*W)
print("VERIFIABILITY CHECK RESULTS")
print("="*W)

print("""
TASK 2 — PCA (128 components) variance explained
-------------------------------------------------
Matrix:   Cohort A labeled only (N={}), mean-across-{}-pipelines,
          top-{:,} probes by inter-sample variance, StandardScaler.
Fit:      Once on the FULL labeled Cohort A training set.
          NOT computed per CV fold; NOT refit on fold subsets.

  np.cumsum(pca.explained_variance_ratio_)[-1] = {:.6f}
  => {:.2f}% of variance explained by 128 PCs.

  Paper figure '77.9%': {}

  Per-fold variance (for reference, not what was reported):
    mean = {:.2f}%  std = {:.2f}%  (across {} folds)
""".format(
    mask_a.sum(), data_a.shape[0], TOP_PROBES,
    cumvar, cumvar * 100,
    "CONFIRMED" if abs(cumvar - 0.779) < 0.002 else f"INCORRECT — actual = {cumvar*100:.2f}%",
    np.mean(fold_vars)*100, np.std(fold_vars)*100, N_FOLDS
))

print("""
TASK 3 — Random-init encoder: missing Cohort B protocols
---------------------------------------------------------
(a) Std CV (5-fold, full pool N≈{}):
    RandInit AUC = {:.4f} ± {:.4f}
    MethCLR  AUC = {:.4f} ± {:.4f}  (reference)
    Bootstrap MethCLR vs RandInit:
      ΔAUC = {:+.4f}  95% CI [{:+.4f}, {:+.4f}]  p={:.4f}  {}

(b) Matched-N CV (5-fold, pool subsampled to N={}, {} reps):
    RandInit AUC = {:.4f} ± {:.4f}

Updated Table IV RandInit row:
  Label-free:   0.6236        (existing)
  Std CV:       {:.4f} ± {:.4f}  (NEW)
  Matched-N CV: {:.4f} ± {:.4f}  (NEW)
""".format(
    int(N_B * 4 / 5),
    rand_std_mean, rand_std_std,
    mclr_std_mean, mclr_std_std,
    delta_std, lo_std, hi_std, pv_std, sig_std,
    N_MATCH, N_REPS_MN,
    rand_mn_mean, rand_mn_std,
    rand_std_mean, rand_std_std,
    rand_mn_mean, rand_mn_std,
))

print("="*W)
print(f"Full results → {out_path}")
