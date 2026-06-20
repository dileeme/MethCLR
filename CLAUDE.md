# CLAUDE.md — MethCLR Session Handoff
**Author:** Dilen | CSE, Anna University, Chennai
**Last Updated:** June 6, 2026
**Status:** 12/12 pipelines complete, intersection done — 3 training runs complete. Best: Run 3 (TOP_PROBES=20,000, AUC=0.8447). Ready for cross-cohort evaluation (Cohort B).

---

## 1. LOCKED PROJECT CONFIGURATION

- **Title:** *MethCLR: Contrastive Representation Learning for Pipeline-Invariant DNA Methylation Embeddings*
- **Target Venue:** IEEE BIBM 2026 Main Conference Track (Dallas, Dec 1–4, 2026)
- **Hard Deadline:** July 5, 2026
- **Track:** Primary: 3.g Epigenomics | Secondary: 5.a Data Mining, ML, AI

### One-Sentence Contribution
A self-supervised contrastive learning architecture that treats alternative bioinformatics preprocessing pipelines as data augmentations for the same biological sample — forcing the network to learn pipeline-invariant latent representations that isolate true biology and eliminate analytical noise in EWAS.

### Architecture
- Positive pairs: same biological sample processed through two distinct pipeline configurations from the 12-pipeline grid
- Negative pairs: different biological samples
- Loss: NT-Xent / InfoNCE (temperature τ=0.07)
- Model: lightweight MLP encoder (input_dim → 512 → 256 → 128) + projection head (128 → 64)

---

## 2. COHORT CONFIGURATION (LOCKED)

### Cohort A — Training (GSE85210)
- **Status:** 12/12 beta matrices complete, 12/12 intersected matrices complete
- **Local IDAT path:** `C:\Users\dilsh\OneDrive\Desktop\MethCLR\GSE85210_IDATs\` — 506 IDATs
- **IDATs renamed:** GSM prefix stripped. Files named `{Sentrix_ID}_{Sentrix_Position}_Grn/Red.idat.gz`
- **Series matrix:** `C:\Users\dilsh\Downloads\GSE85210_series_matrix.txt.gz`
- **Phenotype:** Binary — smoker (172) / non-smoker (81), N=253
- **Tissue:** Peripheral blood, 450K array (GPL13534)
- **WARNING:** GSM→Sentrix ID mapping could not be parsed from series matrix supplementary URLs in earlier runs. Smoking labels NOT in beta matrix column names. `dataset.py` re-links labels from series matrix at runtime.

### Cohort B — Evaluation (GSE224807)
- **Status:** NOT downloaded yet
- **Phenotype:** `smoking_status: SM` / `smoking_status: NS`
- **Platform:** GPL13534 (450K)
- **Download command:**
```powershell
curl "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE224nnn/GSE224807/suppl/GSE224807_RAW.tar" -o "$env:USERPROFILE\Downloads\GSE224807_RAW.tar"
```

---

## 3. COMPUTE ENVIRONMENT

- **OS:** Windows 11, PowerShell — WSL2 uninstalled, do NOT use it
- **Python:** 3.12, native Windows
- **Working directory:** `C:\Users\dilsh\OneDrive\Desktop\MethCLR\`
- **Training:** Google Colab T4 GPU (not local — too slow)
- **RAM:** 16 GB local

---

## 4. CRITICAL METHYLPREP PATCHES APPLIED

methylprep 1.7.1 uses deprecated pandas API removed in pandas 2.0. Source patched directly.

### Patch 1 — infer_channel_switch.py
`C:\Users\dilsh\AppData\Local\Programs\Python\Python312\Lib\site-packages\methylprep\processing\infer_channel_switch.py`
- Line 171: `oobG.append(green_in_band)` → `pd.concat([oobG, green_in_band])`
- Line 172: `oobR.append(red_in_band)` → `pd.concat([oobR, red_in_band])`
- Line 189: `lookupIG.append(lookupIR)` → `pd.concat([lookupIG, lookupIR])`

### Patch 2 — dye_bias.py
`C:\Users\dilsh\AppData\Local\Programs\Python\Python312\Lib\site-packages\methylprep\processing\dye_bias.py`
- Line 218: `data.loc[insupp] = yinterp` → `data.loc[insupp] = yinterp.astype(data.dtype)`
- Line 248: `data.loc[insupp] = yinterp` → `data.loc[insupp] = yinterp.astype(data.dtype)`

**If methylprep is ever reinstalled, reapply these patches.**

---

## 5. FOLDER STRUCTURE

```
MethCLR/
├── new_pipeline.py              ← canonical pipeline runner (v4)
├── run_norm.py                  ← single-norm runner for parallelism
├── intersect_pipelines.py       ← probe intersection across 12 pipelines
├── CLAUDE.md                    ← this file
│
├── GSE85210_IDATs/              ← 506 raw IDATs (Cohort A)
├── beta_matrices/               ← 12 raw + 12 intersected .csv.gz + probe_intersection.txt
├── model/                       ← encoder.py, loss.py, dataset.py, train.py
├── colab_dump/                  ← flat upload bundle for Colab (18 files, no subfolders)
│
├── cohort_b/                    ← GSE224807 IDATs + beta matrices (Week 4)
├── results/                     ← checkpoints, trained model weights
├── figures/                     ← training curves, UMAP plots
└── eval/                        ← linear probe results, benchmark vs ComBat
```

---

## 6. PIPELINE GRID — COMPLETE

### Normalization configs
| Label | sesame | save_uncorrected | Equivalent |
|---|---|---|---|
| `sesame` | True | False | SeSAMe (NOOB + dye correction) |
| `minfi` | False | False | preprocessIllumina equivalent |
| `raw` | True | True | Uncorrected raw betas |

### Known probe count difference
SeSAMe returns ~135,476 probes (quality mask applied). minfi/raw return ~485,000. Expected behavior.

### Probe intersection result
- **Final intersection: 132,337 probes** (bottlenecked by SeSAMe sex filter)
- This is the fixed input dimension for the MethCLR encoder
- Log: `beta_matrices/probe_intersection.txt`

### Resume command (idempotent)
```powershell
cd C:\Users\dilsh\OneDrive\Desktop\MethCLR
python new_pipeline.py
```

---

## 7. MODEL — `model/`

| File | Purpose |
|---|---|
| `encoder.py` | MLP: input_dim → 512 → 256 → 128 (h) + projector 128 → 64 (z) |
| `loss.py` | InfoNCE / NT-Xent loss, temperature τ |
| `dataset.py` | Loads 12 intersected matrices, re-links smoking labels, serves (anchor, positive, label) pairs |
| `train.py` | Full training loop — early stopping on linear probe AUC, saves checkpoints |

### Training hyperparameters (train.py)
```python
EPOCHS       = 200
BATCH_SIZE   = 128
LR           = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE  = 0.07
DROPOUT      = 0.1
EVAL_EVERY   = 5     # linear probe AUC every N epochs
PATIENCE     = 10    # stop after 50 epochs of no AUC improvement
TRAIN_SPLIT  = 0.85
TOP_PROBES   = 5000  # feature selection: top-N CpGs by inter-sample variance
```

### Feature selection
- Input reduced from 132,337 → 5,000 probes before encoding
- Selection method: top-N CpGs by inter-sample variance (computed on mean-across-pipelines to isolate biological variation, not pipeline noise)
- Reduces first-layer parameters from 67.8M → 2.56M — critical for N=253
- Implemented in `MethCLRDataset.__init__` via `top_probes` argument
- Scientifically defensible: standard practice in methylation ML

### Observed training results (run 1 — without feature selection)
- Best AUC = 0.8337 at epoch 20
- AUC dropped after epoch 20 due to p >> n overfitting in first layer
- Loss converged correctly (4.4 → 0.05 by epoch 120)
- Val loss noisy through epoch ~60, stabilised ~0.12 from epoch ~80
- Train/val loss gap at convergence: ~0.05 vs ~0.12 — expected for N=253, not severe
- Root cause of AUC drop: 67.8M first-layer parameters (132,337 × 512) on N=253 samples
- InfoNCE over-optimised pipeline invariance at the cost of linear class separability after epoch 20
- Best checkpoint: `methclr_best.pt` saved at epoch 20

### Feature selection implementation detail (for paper methods section)
- `load_intersected_matrices()` loads all 12 matrices at full size (132,337 × 253) — this output is expected and correct
- Feature selection runs inside `MethCLRDataset.__init__` AFTER stacking all pipelines
- Variance computed on mean-across-pipelines array (n_samples × n_probes) — isolates inter-sample biological variance, not pipeline-specific noise
- Top 5,000 probe indices sorted to preserve chromosomal probe order before slicing
- Console confirms: `[FEAT] Selected top 5,000 probes by inter-sample variance (from 132,337)`
- Encoder input dimension confirmed as 5,000 post-selection

### Run 2 — TOP_PROBES=5,000 (complete)
- Hyperparameters: EPOCHS=200, BATCH_SIZE=128, EVAL_EVERY=5, PATIENCE=10, TOP_PROBES=5,000
- First-layer parameters: 2.56M (5,000 × 512)
- Best AUC = 0.7899 at epoch 20
- Early stopping triggered at epoch 70
- Loss convergence: cleaner than Run 1 — train/val gap closed significantly
- AUC volatile (0.73–0.79), high variance in 5-fold CV on N=253
- Conclusion: 5,000 probes too aggressive — excluded smoking-relevant probes with moderate-but-consistent signal

### Run 3 — TOP_PROBES=20,000 (complete, best run)
- Hyperparameters: EPOCHS=200, BATCH_SIZE=128, EVAL_EVERY=5, PATIENCE=10, TOP_PROBES=20,000
- First-layer parameters: 10.24M (20,000 × 512)
- **Best AUC = 0.8447 at epoch 5** — highest across all runs
- Early stopping triggered at epoch 55
- Loss convergence: smooth, val oscillation only between epochs 20–30, converged by ~epoch 45
- AUC peaked earlier (epoch 5 vs epoch 20 in Run 1) — cleaner feature space allows faster biological signal lock-in
- Post-peak AUC drop consistent across all 3 runs — confirms genuine tension between InfoNCE pipeline-invariance objective and linear class separability (publishable finding)
- Checkpoint: `results/methclr_best.pt`

### Cross-run comparison table (for paper)
| Run | TOP_PROBES | First-layer params | Best AUC | Best epoch | Stopped at |
|---|---|---|---|---|---|
| Run 1 | 132,337 (none) | 67.8M | 0.8337 | 20 | 150 (hard cap) |
| Run 2 | 5,000 | 2.56M | 0.7899 | 20 | 70 (early stop) |
| Run 3 | 20,000 | 10.24M | **0.8447** | **5** | 55 (early stop) |

### Key scientific observations (for paper discussion section)
1. **InfoNCE vs linear separability tension:** AUC peaks early in all runs then declines as the contrastive objective optimises pipeline invariance at the cost of class discriminability. Consistent across all 3 runs — not an artifact.
2. **Feature selection sweet spot:** 20,000 probes balances smoking signal retention and parameter control for N=253. Too few (5,000) discards relevant signal; too many (132,337) causes p >> n overfitting.
3. **Contrastive loss quality:** Feature selection improves train/val loss gap (less overfitting in the contrastive objective). Run 2 had the tightest gap; Run 3 moderate; Run 1 widest.
4. **Best checkpoint is always early:** Peaks at epoch 5–20 across all runs. For downstream cross-cohort evaluation, always use `methclr_best.pt`, not `methclr_final.pt`.
5. **AUC of 0.8447 from a self-supervised model on N=253** is competitive — supervised methods on this dataset reach 0.92–0.96, so the self-supervised gap is ~5–10%, which is defensible for a contrastive representation learning approach without label supervision during training.

---

## 8. COLAB TRAINING

### Current status
- Files uploaded flat to `/content/` on Colab T4 GPU (no subfolders)
- `colab_dump/` contains 18 flat files: 12 `*_intersected.csv.gz`, `GSE85210_series_matrix.txt.gz`, `dataset.py`, `encoder.py`, `loss.py`, `train.py`
- `DRIVE_ROOT = "/content"` and `MATRIX_DIR = DRIVE_ROOT` in `train.py`
- Run 2 actively training with feature selection — upload `dataset.py` + `train.py` from `colab_dump/` to replace old versions on Colab before each new run
- Loading output showing `(132337, 253)` per file is expected — feature selection applies after stacking inside dataset, not during file load

### Run command
```python
!python /content/train.py
```

### Path fix (if DRIVE_ROOT is wrong)
```python
path = "/content/train.py"
text = open(path).read()
text = text.replace('DRIVE_ROOT   = "/content/colab_dump"', 'DRIVE_ROOT   = "/content"')
open(path, "w").write(text)
!python /content/train.py
```

### Estimated runtime
- Data loading: ~10–15 min
- Training ~100 epochs: ~5–10 min on T4
- Total: ~20–30 min

### Checkpoints saved to
`/content/checkpoints/methclr_best.pt` and `methclr_final.pt`

---

## 9. NEXT STEPS (in order)

### Immediately — DONE
- 3 training runs complete. Best model: Run 3 (TOP_PROBES=20,000, AUC=0.8447, epoch 5)
- `methclr_best.pt` saved to `results/`
- Current `train.py`: TOP_PROBES=20,000, BATCH_SIZE=128, EPOCHS=200, PATIENCE=10, EVAL_EVERY=5

### Week 4 — Cross-Cohort Evaluation
1. Download GSE224807, run same 12-pipeline grid via `new_pipeline.py` (update IDAT_DIR/OUTPUT_DIR)
2. Run `intersect_pipelines.py` on Cohort B matrices
3. Embed GSE224807 samples using trained encoder (frozen weights)
4. Linear probe AUC on smoking classification
5. Negative controls: label permutation, cg05575921 probe stability
6. Benchmark against ComBat

### Week 5 — Paper
1. 6-page IEEE conference template
2. Key figures: loss curves, linear probe AUC, UMAP embeddings, cross-cohort transfer

---

## 10. HARD CONSTRAINTS (never violate)

- Solo author — no collaborators
- No FHE/CKKS/BFV/privacy/cryptography content
- No processed beta matrices as input — raw IDATs only through own pipeline grid
- Main track requires cross-cohort generalization — cannot submit single-cohort model
- Keep model lightweight — N=253 is small, representation collapse is a real risk
- Do not add omics modalities, transformers, or phenotype imputation
- Do NOT reinstall WSL2 — all work is Windows-native Python
- Do NOT move new_pipeline.py while it is running

---

## 11. INSTRUCTIONS FOR NEXT AGENT

1. Do not reopen scope — title, architecture, cohorts are locked
2. WSL2 is gone — everything runs in Windows PowerShell natively
3. All 12 beta matrices and 12 intersected matrices are complete in `beta_matrices/`
4. If methylprep errors with `.append()` or dtype errors, reapply patches from Section 4
5. Smoking labels are NOT in beta matrix columns — re-linked from series matrix at runtime in `dataset.py`
6. Training runs on Google Colab T4 GPU — not local CPU
7. Colab upload is flat (no subfolders) — `DRIVE_ROOT = "/content"`
8. Checkpoints save to `/content/checkpoints/` on Colab — download to `results/` locally
9. Do not suggest stopping or sleeping
