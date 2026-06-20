# MethCLR: Contrastive Representation Learning for Pipeline-Invariant DNA Methylation Embeddings

MethCLR is a self-supervised contrastive learning framework designed to eliminate analytical preprocessing noise in Epigenome-Wide Association Studies (EWAS). By treating alternative bioinformatics pipeline configurations as data augmentations, MethCLR forces the neural network to map different analytical treatments of the same biological sample close together in the latent space, while pushing distinct biological samples apart. The result is a robust, pipeline-invariant representation of DNA methylation data that preserves true biological signals while filtering out technical noise.

## Core Architecture

The framework maps pipeline variations directly to the positive-pair axis in a contrastive learning setup, analogous to SimCLR/MoCo frameworks in computer vision:
- **Positive Pairs:** The same biological sample processed through two different pipeline configurations (varying normalization, cell-type adjustment, or batch handling).
- **Negative Pairs:** Distinct biological samples.
- **Loss Function:** NT-Xent (Normalized Temperature-scaled Cross Entropy) / InfoNCE loss to maximize agreement between pipeline-variant representations of identical biology.

## Repository Structure

```text
├── data/                  # Metadata tables, selected GSM lists, and verification artifacts
├── src/
│   ├── preprocessing/     # minfi-based 12-pipeline grid execution scripts (R/rpy2)
│   ├── models/            # PyTorch implementation of MethCLR and baseline architectures
│   ├── training/          # Contrastive training loops and batch generation utilities
│   └── evaluation/        # Variance decomposition, negative controls, and cross-cohort testing
├── notebooks/             # Google Colab workflow templates
├── README.md
└── requirements.txt
