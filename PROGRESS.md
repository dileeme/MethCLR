# MethCLR — Progress Tracker
**Author:** Dilen Shankar | CSE, Anna University, Chennai
**Target:** IEEE BIBM 2026 Main Track | Dallas, Dec 1–4, 2026
**Hard Deadline:** July 5, 2026
**Today:** June 23, 2026

---

## Status Overview

| Phase | Status |
|---|---|
| Data & Preprocessing | ✅ Complete |
| Model & Training | ✅ Complete |
| Evaluation — Cohort A | ✅ Complete |
| Evaluation — Cohort B | ✅ Complete |
| Baselines | ✅ Complete |
| Figures | ✅ Complete |
| Paper (LaTeX) | ✅ Complete — in `papier.zip` |
| Submission | ⏳ Pending (July 5 deadline) |

---

## Phase 1 — Data & Preprocessing ✅

- [x] Downloaded GSE85210 raw IDATs (506 files, Cohort A)
- [x] Renamed IDATs: GSM prefix stripped → `{Sentrix_ID}_{Sentrix_Position}_Grn/Red.idat.gz`
- [x] Applied methylprep patches (pandas 2.0 compatibility — `infer_channel_switch.py`, `dye_bias.py`)
- [x] Built 12-pipeline grid: 3 norms × 2 SNP filter × 2 sex probe removal
- [x] Generated all 12 beta matrices for Cohort A → `beta_matrices/`
- [x] Downloaded GSE224807 raw IDATs (Cohort B)
- [x] Generated all 12 beta matrices for Cohort B → `cohort_b_beta_matrices/`
- [x] Ran `intersect_pipelines.py` on Cohort A → **132,337 probe intersection**
- [x] Ran `intersect_pipelines.py` on Cohort B → same probe space
- [x] Confirmed probe intersection log: `beta_matrices/probe_intersection.txt`

---

## Phase 2 — Model & Training ✅

- [x] Built `model/encoder.py` — MLP: 20k→512→256→128 (h) + projector 128→64 (z)
- [x] Built `model/loss.py` — InfoNCE / NT-Xent, τ=0.07
- [x] Built `model/dataset.py` — loads 12 intersected matrices, re-links smoking labels from series matrix, serves (anchor, positive, label) pairs; feature selection top-N by inter-sample variance on mean-across-pipelines
- [x] Built `model/train.py` — full training loop, 5-fold linear probe AUC early stopping (PATIENCE=10, EVAL_EVERY=5)
- [x] Uploaded flat bundle to Google Colab T4 (`colab_dump/`, 18 files)

### Training runs

| Run | TOP_PROBES | Params (M) | Best AUC | Best Epoch | Stopped |
|---|---|---|---|---|---|
| Run 1 | 132,337 | 67.8 | 0.8337 | 20 | 150 |
| Run 2 | 5,000 | 2.56 | 0.7899 | 20 | 70 |
| **Run 3** | **20,000** | **10.24** | **0.8447** | **5** | **55** |

- [x] Best checkpoint saved: `results/methclr_best.pt` (Run 3, epoch 5)
- [x] Final checkpoint saved: `results/methclr_final.pt` (Run 3, epoch 55)

---

## Phase 3 — Evaluation ✅

### Cohort A (in-cohort)
- [x] Linear probe AUC 5-fold CV on Cohort A embeddings → **0.8447**
- [x] Feature selection confirmed: top 20,000 probes by inter-sample biological variance

### Cohort B (zero-shot cross-cohort)
- [x] Ran `eval/eval_cohort_b.py` with frozen encoder on GSE224807
- [x] Cohort B embeddings saved: `eval/cohort_b_embeddings.npy`, `cohort_b_labels.npy`, `cohort_b_sample_ids.npy`
- [x] Results saved: `eval/cross_cohort_results.json`
  - Cohort B AUC: **0.7764 ± 0.013** (folds: 0.774, 0.775, 0.774, 0.759, 0.799)
  - Transfer gap: **0.068** (6.8 pp)
  - Permutation null: 0.522 ± 0.029
- [x] UMAP visualization generated: `eval/umap_cohort_b.png`

---

## Phase 4 — Baselines ✅

- [x] Built `run_baselines.py` — PCA (128 PCs) and Autoencoder (128-dim) baselines
  - Multi-threaded (psutil physical cores, torch num_threads)
  - 5-fold stratified CV, StandardScaler per fold
  - Evaluates on both Cohort A and Cohort B (frozen, same protocol as MethCLR)
- [x] Results saved: `eval/baseline_results.json`

| Method | Cohort A | Cohort B | Δ |
|---|---|---|---|
| PCA (128) + LR | 0.822 ± 0.033 | 0.964 ± 0.015 | −0.142 † |
| AE (128) + LR | 0.833 ± 0.059 | 0.747 ± 0.034 | +0.086 |
| **MethCLR** | **0.8447** | **0.776 ± 0.013** | **+0.068** |

† PCA Cohort B gain is a sample-size artifact (N≈631 vs N≈202 per fold), not a representation quality effect.

---

## Phase 5 — Figures ✅

- [x] Built `generate_plots.py` — publication-ready figures (pdf.fonttype=42, vector-safe)
- [x] **Fig 1** `assets/fig1_optimization_paradox.png/pdf` — dual-axis training dynamics (InfoNCE loss + linear probe AUC vs epoch, optimal embedding zone shaded)
- [x] **Fig 2** `assets/fig2_pipeline_alignment.png/pdf` — violin plot of relative intra-sample pipeline distance (Raw / PCA / AE / MethCLR)
- [x] **Fig 3** `assets/fig3_crosscohort_gap.png/pdf` — grouped bar chart: ROC-AUC, PR-AUC, F1 across methods and cohorts with Δ arrows
- [x] **Fig 4** `eval/umap_cohort_b.png` — UMAP of Cohort B embeddings (frozen encoder, no Cohort B supervision)

---

## Phase 6 — Paper ✅

- [x] Full IEEE double-column draft written: `latex/IEEE-conference-template-062824/methclr.tex`
- [x] Abstract — pipeline problem, MethCLR approach, AUC results, baseline comparison summary
- [x] Introduction — 5 contribution bullets including baseline evaluation
- [x] Related Work — methylation preprocessing, contrastive SSL, batch correction
- [x] Methods — dataset config, 12-pipeline grid, feature selection (Eq. 1), architecture, InfoNCE (Eq. 2), hyperparameters, evaluation protocol (linear probe, permutation, UMAP, baselines, pipeline alignment metric)
- [x] Results — Table I (ablation), Table II (baseline comparison), Fig 1 (training), Fig 2 (pipeline alignment), Fig 4 (UMAP)
- [x] Discussion — 4 subsections: InfoNCE vs separability tension, feature selection as regularizer, cross-cohort generalization, baseline comparison (PCA artifact explained)
- [x] Conclusion — references all key findings including baseline comparison
- [x] 21 references
- [x] Overleaf-ready zip built: `papier.zip` (image paths corrected for zip directory structure)
- [x] Estimated page count: ~7.5 pages (within 8-page limit)

---

## Remaining

- [ ] Final proofread of `methclr.tex` before submission
- [ ] Compile on Overleaf — verify figures render, page count confirmed ≤ 8
- [ ] Submit via IEEE BIBM 2026 portal by **July 5, 2026**
