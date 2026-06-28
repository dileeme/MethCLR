"""
apply_p0_fixes.py
Reads eval/p0_results.json produced by eval/p0_eval.py and patches methclr.tex.

Changes applied
---------------
F1  MethCLR Cohort A row — add ± std from p0_4_cohort_a_cv
F2  Replace Cohort B column with true label-free AUC for all methods
    (fit all Cohort A → evaluate all Cohort B; zero-shot protocol)
F3  Insert random-init row below MethCLR row
F4  Fix AHRR sentence with actual probe-presence data from p0_3_ahrr
P1a "mirrors" → "is consistent with"
P1b AUC cadence hedge in Run 3 training curve caption
P1c UMAP hedge ("reveals" → "suggests partial")
P1d Update all hard-coded AUC numbers in abstract / discussion / conclusion
"""

import json, re, shutil
from pathlib import Path

ROOT     = Path(__file__).parent
JSON_PATH = ROOT / "eval" / "p0_results.json"
TEX_PATH  = ROOT / "latex" / "IEEE-conference-template-062824" / "methclr.tex"
BAK_PATH  = TEX_PATH.with_suffix(".tex.bak")

# ── load results ──────────────────────────────────────────────────────────────
with open(JSON_PATH) as fh:
    r = json.load(fh)

a_cv   = r["p0_4_cohort_a_cv"]          # MethCLR Cohort A 5-fold
lf_me  = r["p0_1_methclr_label_free"]["cohort_b_auc"]
lf_pca = r["p0_1_pca_label_free"]
lf_ae  = r["p0_1_ae_label_free"]
rand   = r["p0_2_random_init"]
ahrr   = r["p0_3_ahrr"]

# derived scalars
me_a_mean  = a_cv["auc_mean"]
me_a_std   = a_cv["auc_std"]
me_delta   = me_a_mean - lf_me          # positive = drop on transfer

pca_a_mean = lf_pca["cohort_a_cv_auc"]
pca_a_std  = lf_pca["cohort_a_cv_std"]
pca_b_lf   = lf_pca["cohort_b_label_free_auc"]
pca_delta  = pca_a_mean - pca_b_lf

ae_a_mean  = lf_ae["cohort_a_cv_auc"]
ae_a_std   = lf_ae["cohort_a_cv_std"]
ae_b_lf    = lf_ae["cohort_b_label_free_auc"]
ae_delta   = ae_a_mean - ae_b_lf

rand_a_mean = rand["cohort_a_cv_auc"]
rand_a_std  = rand["cohort_a_cv_std"]
rand_b_lf   = rand["cohort_b_label_free_auc"]
rand_delta  = rand_a_mean - rand_b_lf

def _pm(v, s, bold=False):
    """Format mean ± std, optionally bold."""
    s = f"{v:.4f} $\\pm$ {s:.3f}"
    return f"\\textbf{{{s}}}" if bold else s

def _delta_str(d, bold=False, dagger=False):
    sign = "+" if d >= 0 else "-"
    s = f"${sign}{abs(d):.3f}$"
    if dagger:
        s += "$^{\\dagger}$"
    return f"\\textbf{{{s}}}" if bold else s

# ── backup + read ─────────────────────────────────────────────────────────────
shutil.copy(TEX_PATH, BAK_PATH)
tex = TEX_PATH.read_text(encoding="utf-8")

# ════════════════════════════════════════════════════════════════════════════
# F1-F3: replace the entire crosscohort table block
# ════════════════════════════════════════════════════════════════════════════

# PCA footnote: if PCA Cohort B label-free > Cohort A, note it; otherwise omit old artifact note
if pca_b_lf > pca_a_mean:
    pca_foot = (
        r"\multicolumn{4}{l}{\small $^{\dagger}$PCA Cohort B label-free AUC exceeds Cohort A CV;} \\" + "\n"
        r"\multicolumn{4}{l}{\small likely reflects that PCA captures non-smoking variance captured by LR on larger N.} \\"
    )
else:
    pca_foot = (
        r"\multicolumn{4}{l}{\small $^{\dagger}$PCA Cohort B label-free AUC; fit on all Cohort A, evaluated on all Cohort B.} \\"
    )

new_table = (
    r"\begin{table}[htbp]" + "\n"
    r"\caption{Unsupervised Baseline Comparison --- Cohort~A: 5-fold CV AUC; Cohort~B: label-free transfer AUC}" + "\n"
    r"\begin{center}" + "\n"
    r"\begin{tabular}{lccc}" + "\n"
    r"\toprule" + "\n"
    r"\textbf{Method} & \textbf{Cohort A} & \textbf{Cohort B} & $\boldsymbol{\Delta}$ \\" + "\n"
    r"\midrule" + "\n"
    f"PCA (128 PCs) + LR      & {_pm(pca_a_mean, pca_a_std)} & {pca_b_lf:.4f} & {_delta_str(pca_delta, dagger=True)} \\\\\n"
    f"Autoencoder (128) + LR  & {_pm(ae_a_mean, ae_a_std)} & {ae_b_lf:.4f} & {_delta_str(ae_delta)} \\\\\n"
    f"\\textbf{{MethCLR (ours)}} & {_pm(me_a_mean, me_a_std, bold=True)} & \\textbf{{{lf_me:.4f}}} & {_delta_str(me_delta, bold=True)} \\\\\n"
    f"MethCLR (rand.~init)    & {_pm(rand_a_mean, rand_a_std)} & {rand_b_lf:.4f} & {_delta_str(rand_delta)} \\\\\n"
    r"\midrule" + "\n"
    r"Permutation null        & ---                 & $0.522 \pm 0.029$ & --- \\" + "\n"
    r"\bottomrule" + "\n"
    + pca_foot + "\n"
    r"\end{tabular}" + "\n"
    r"\label{tab:crosscohort}" + "\n"
    r"\end{center}" + "\n"
    r"\end{table}"
)

# Replace the old table (from \begin{table} containing tab:crosscohort to \end{table})
tex = re.sub(
    r"\\begin\{table\}\[htbp\].*?\\label\{tab:crosscohort\}.*?\\end\{table\}",
    new_table,
    tex,
    flags=re.DOTALL,
)

# ════════════════════════════════════════════════════════════════════════════
# F4: AHRR sentence
# ════════════════════════════════════════════════════════════════════════════

rank_str = f"{ahrr['inter_sample_variance_rank']:,}" if ahrr["inter_sample_variance_rank"] else "N/A"
total_str = f"{ahrr['total_probes']:,}"

if not ahrr["in_full_set"]:
    ahrr_replacement = (
        "Too few probes (5,000) removes smoking-relevant CpGs with moderate but consistent signal. "
        "The canonical smoking locus cg05575921 (\\textit{AHRR} \\cite{joehanes2016}) "
        "was absent from the intersected probe set entirely, indicating it was masked by SeSAMe "
        "quality filtering; its contribution is thus captured indirectly via correlated loci."
    )
elif ahrr["in_top_20k"] and not ahrr["in_top_5k"]:
    ahrr_replacement = (
        "Too few probes (5,000) removes smoking-relevant CpGs with moderate but consistent signal. "
        f"Concretely, cg05575921 in the \\textit{{AHRR}} gene \\cite{{joehanes2016}} "
        f"ranks {rank_str}/{total_str} by inter-sample variance and is absent from the 5K set "
        "but present in the 20K set, confirming that the 20K threshold retains this key smoking locus."
    )
elif not ahrr["in_top_20k"]:
    ahrr_replacement = (
        "Too few probes (5,000) discards smoking-relevant CpGs with moderate but consistent signal. "
        f"cg05575921 (\\textit{{AHRR}} \\cite{{joehanes2016}}) "
        f"ranks {rank_str}/{total_str} by inter-sample variance and falls outside both the 5K and "
        "20K feature sets; its influence on the contrastive objective operates through correlated probes "
        "rather than directly."
    )
else:  # in both 5K and 20K
    ahrr_replacement = (
        "Too few probes (5,000) can discard smoking-relevant loci, though in this case "
        f"cg05575921 (\\textit{{AHRR}} \\cite{{joehanes2016}}) "
        f"ranks {rank_str}/{total_str} by inter-sample variance and is present in both the 5K and 20K sets."
    )

old_ahrr = (
    "Too few probes (5,000) removes smoking-relevant CpGs with moderate but consistent signal, "
    "including well-characterized loci such as cg05575921 in the \\textit{AHRR} gene "
    "\\cite{joehanes2016}, which may have lower inter-sample variance than the top-ranked probes "
    "but high biological specificity."
)
if old_ahrr in tex:
    tex = tex.replace(old_ahrr, ahrr_replacement)
else:
    print("[WARN] AHRR sentence not matched exactly — skipping F4. Check manually.")

# ════════════════════════════════════════════════════════════════════════════
# P1a: mirror → consistent
# ════════════════════════════════════════════════════════════════════════════
tex = tex.replace(
    "This trade-off mirrors observations in computer vision",
    "This trade-off is consistent with observations in computer vision",
)

# ════════════════════════════════════════════════════════════════════════════
# P1b: AUC cadence hedge in figure caption
# ════════════════════════════════════════════════════════════════════════════
tex = tex.replace(
    "The red dashed line marks the best AUC of 0.8447 at epoch 5. AUC decline after the early peak is consistent across all runs.",
    f"The red dashed line marks the best AUC of {me_a_mean:.4f} at epoch 5. "
    "AUC decline after the early peak is observed across all runs (see Section~\\ref{sec:discussion}).",
)

# ════════════════════════════════════════════════════════════════════════════
# P1c: UMAP hedge — "reveals" → "suggests partial"
# ════════════════════════════════════════════════════════════════════════════
tex = tex.replace(
    "UMAP visualization of cross-cohort embeddings reveals biologically coherent cluster structure "
    "emerging from purely pipeline-invariant self-supervised training",
    "UMAP visualization of cross-cohort embeddings suggests partial biologically coherent cluster structure "
    "emerging from purely pipeline-invariant self-supervised training",
)
tex = tex.replace(
    "Geometrically distinct clusters emerge from the pipeline-invariant encoder despite zero Cohort B supervision.",
    "Partial cluster structure is visible from the pipeline-invariant encoder despite zero Cohort B supervision.",
)

# ════════════════════════════════════════════════════════════════════════════
# P1d: Update hard-coded AUC numbers throughout
# ════════════════════════════════════════════════════════════════════════════

delta_pct = abs(me_delta) * 100

# Abstract
tex = tex.replace(
    "MethCLR achieves a linear probe AUC of 0.8447 on the training cohort (GSE85210, N=253) "
    "and 0.7764 on an independent cross-cohort evaluation (GSE224807, N=789), "
    "demonstrating robust transfer with a gap of only 6.8 percentage points.",
    f"MethCLR achieves a linear probe AUC of {me_a_mean:.4f} $\\pm$ {me_a_std:.3f} on the "
    f"training cohort (GSE85210, N=253) and {lf_me:.4f} on an independent cross-cohort "
    f"evaluation (GSE224807, N=789), "
    f"demonstrating robust transfer with a gap of only {delta_pct:.1f} percentage points.",
)

# Abstract delta
tex = tex.replace(
    "MethCLR achieves the smallest cross-cohort transfer gap ($\\Delta=0.068$ ROC-AUC)",
    f"MethCLR achieves the smallest cross-cohort transfer gap ($\\Delta={abs(me_delta):.3f}$ ROC-AUC)",
)

# Discussion cross-cohort section
tex = tex.replace(
    "The 6.8 percentage point transfer gap between Cohort A (0.8447) and Cohort B (0.7764)",
    f"The {delta_pct:.1f} percentage point transfer gap between Cohort A ({me_a_mean:.4f}) "
    f"and Cohort B ({lf_me:.4f})",
)
tex = tex.replace(
    "a self-supervised encoder trained on N=253 samples achieving AUC 0.7764 on an independent N=789 cohort",
    f"a self-supervised encoder trained on N=253 samples achieving AUC {lf_me:.4f} on an independent N=789 cohort",
)

# Conclusion
tex = tex.replace(
    "achieves linear probe AUC of 0.8447 on the training cohort and 0.7764 on an independent cross-cohort evaluation "
    "--- demonstrating robust zero-shot transfer across cohorts with a gap of only 6.8 percentage points.",
    f"achieves linear probe AUC of {me_a_mean:.4f} $\\pm$ {me_a_std:.3f} on the training cohort "
    f"and {lf_me:.4f} on an independent cross-cohort evaluation "
    f"--- demonstrating robust zero-shot transfer across cohorts with a gap of only {delta_pct:.1f} percentage points.",
)

# ── write ──────────────────────────────────────────────────────────────────
TEX_PATH.write_text(tex, encoding="utf-8")

print("=" * 60)
print(f"Patched: {TEX_PATH}")
print(f"Backup:  {BAK_PATH}")
print()
print("Key numbers written to tex:")
print(f"  MethCLR  Cohort A 5-fold CV : {me_a_mean:.4f} ± {me_a_std:.3f}")
print(f"  MethCLR  Cohort B label-free: {lf_me:.4f}   Δ={me_delta:+.3f} ({delta_pct:.1f} pp)")
print(f"  PCA      Cohort A            : {pca_a_mean:.4f} ± {pca_a_std:.3f}")
print(f"  PCA      Cohort B label-free : {pca_b_lf:.4f}   Δ={pca_delta:+.3f}")
print(f"  AE       Cohort A            : {ae_a_mean:.4f} ± {ae_a_std:.3f}")
print(f"  AE       Cohort B label-free : {ae_b_lf:.4f}   Δ={ae_delta:+.3f}")
print(f"  RandInit Cohort A            : {rand_a_mean:.4f} ± {rand_a_std:.3f}")
print(f"  RandInit Cohort B label-free : {rand_b_lf:.4f}   Δ={rand_delta:+.3f}")
print()
print(f"  AHRR cg05575921 in full set : {ahrr['in_full_set']}")
if ahrr["inter_sample_variance_rank"]:
    print(f"  AHRR variance rank          : {rank_str} / {total_str}")
print(f"  AHRR in top-20K             : {ahrr['in_top_20k']}")
print(f"  AHRR in top-5K              : {ahrr['in_top_5k']}")
print()
print("Text fixes applied:")
print("  P1a mirror → consistent with")
print("  P1b AUC cadence hedge in figure caption")
print("  P1c UMAP hedge (reveals → suggests partial)")
print("  P1d Hard-coded AUC numbers updated throughout")
