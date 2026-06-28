# MethCLR: Contrastive Representation Learning for Pipeline-Invariant DNA Methylation Embeddings

**Submission:** IEEE BIBM 2026 Main Conference Track (Dallas, Dec 1–4, 2026)
**Author:** Dilen | CSE, Anna University Chennai

---

MethCLR is a self-supervised contrastive learning framework for DNA methylation data. It treats alternative bioinformatics preprocessing pipelines applied to the same biological sample as data augmentations, forming positive pairs for an InfoNCE contrastive loss. The encoder learns representations that are invariant to pipeline choice, isolating true biological signal from analytical noise in Epigenome-Wide Association Studies (EWAS).

## Method

- **Positive pairs:** same biological sample processed through two of 12 distinct pipeline configurations (3 normalisation strategies × 4 SNP/sex probe filter combinations)
- **Negative pairs:** different biological samples
- **Loss:** NT-Xent / InfoNCE, temperature τ = 0.07
- **Encoder:** MLP (20,000 → 512 → 256 → 128) + projection head (128 → 64)
- **Feature selection:** top 20,000 CpGs by inter-sample variance on Cohort A mean-across-pipelines (fit on Cohort A only, applied to Cohort B)

## Datasets

| Cohort | GEO ID | N | Phenotype | Role |
|---|---|---|---|---|
| Cohort A | GSE85210 | 253 | Smoker (172) / Non-smoker (81) | Training |
| Cohort B | GSE224807 | 789 | Smoker (374) / Non-smoker (415) | Evaluation |

Both cohorts: peripheral blood, Illumina 450K array (GPL13534).
Probe intersection across all 12 pipelines: **132,337 probes**.

## Key Results

### Cohort A — 5-fold CV (N=253)

| Method | AUC |
|---|---|
| MethCLR (ours) | 0.8443 ± 0.0546 |
| PCA (128 PCs) + LR | 0.8230 ± 0.0333 |
| Autoencoder (128) + LR | 0.8219 ± 0.0259 |
| Random-init encoder + LR | 0.6663 ± 0.0876 |

MethCLR significantly outperforms random-init (ΔAUC = +0.17, p < 0.001, paired bootstrap). Differences vs PCA and AE are not significant at N=253.

### Cohort B — Within-cohort CV (matched-N protocol, N_train = 202)

| Method | AUC (matched-N) | AUC (standard) |
|---|---|---|
| MethCLR | 0.6757 ± 0.0145 | 0.7166 ± 0.0400 |
| PCA (128 PCs) + LR | 0.8993 ± 0.0106 | 0.9637 ± 0.0149 |
| Autoencoder (128) + LR | 0.6802 ± 0.0143 | 0.7372 ± 0.0194 |
| Random-init encoder + LR | 0.6622 ± 0.0175 | 0.7128 ± 0.0435 |

Matched-N protocol: each fold's training pool is stratified-subsampled to N=202 (matching Cohort A's per-fold training size), repeated 20×. Standard cross-cohort CV inflates PCA by +0.064 AUC — a sample-size confound identified and corrected here.

### Label-free Cross-cohort Transfer (zero Cohort B supervision)

| Method | AUC |
|---|---|
| MethCLR | 0.6149 |
| PCA (128 PCs) + LR | 0.6830 |
| Autoencoder (128) + LR | 0.6271 |
| Random-init encoder + LR | 0.6236 |

### Semi-supervised Curve (MethCLR, frozen encoder)

| Cohort B supervision | N | AUC |
|---|---|---|
| 0% (label-free) | 0 | 0.6149 |
| 10% | 79 | 0.6200 ± 0.0279 |
| 20% | 158 | 0.6669 ± 0.0147 |
| 100% (matched-N CV) | 789 | 0.6757 ± 0.0145 |

## Repository Structure

```
MethCLR/
├── model/
│   ├── encoder.py              # MLP encoder + projection head
│   ├── loss.py                 # InfoNCE / NT-Xent loss
│   ├── dataset.py              # Data loading, feature selection, pair construction
│   └── train.py                # Training loop (runs on Google Colab T4)
├── eval/
│   ├── eval_cohort_b.py        # Cross-cohort embedding and CV
│   ├── baselines.py            # PCA and autoencoder baselines
│   ├── p0_eval.py              # Pre-submission verification (AHRR probe, label-free transfer)
│   ├── reviewer_analyses.py    # Matched-N CV, bootstrap significance, semi-supervised curve
│   ├── verifiability_checks.py # PCA variance trace, random-init missing protocols
│   └── *.json                  # All result tables
├── new_pipeline.py             # 12-pipeline beta matrix generation (methylprep)
├── intersect_pipelines.py      # Probe intersection across pipelines
├── requirements.txt
└── CLAUDE.md                   # Full session handoff and experiment log
```

Data directories (`GSE85210_IDATs/`, `beta_matrices/`, `cohort_b_beta_matrices/`, `results/`) are excluded from version control — see `.gitignore`.

## Training

Training runs on Google Colab T4 GPU. Three runs were completed:

| Run | TOP_PROBES | First-layer params | Best AUC | Best epoch |
|---|---|---|---|---|
| Run 1 | 132,337 | 67.8M | 0.8337 | 20 |
| Run 2 | 5,000 | 2.56M | 0.7899 | 20 |
| Run 3 | 20,000 | 10.24M | **0.8447** | **5** |

Best checkpoint saved prospectively at epoch 5 via online early-stopping monitor (PATIENCE=10 evaluations of 5 epochs each; stopped at epoch 55).

## Requirements

```
torch
scikit-learn
pandas
numpy
methylprep==1.7.1
matplotlib
```

See `requirements.txt` for pinned versions. Note: methylprep 1.7.1 requires two source patches for pandas 2.0 compatibility — documented in `CLAUDE.md` Section 4.
