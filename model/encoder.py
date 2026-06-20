"""
MethCLR — MLP Encoder

Architecture: input_dim → 512 → 256 → 128 (representation h)
Projection head: 128 → 64 (used for InfoNCE loss only)
Downstream evaluation uses h (128-dim), not z (64-dim).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
        )
        # Projection head — used during contrastive training only
        self.projector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = F.normalize(h, dim=-1)
        z = self.projector(h)
        z = F.normalize(z, dim=-1)
        return h, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns representation h only — use for downstream evaluation."""
        with torch.no_grad():
            h, _ = self.forward(x)
        return h
