"""
MethCLR — InfoNCE Loss (NT-Xent)

Given a batch of (z_i, z_j) positive pairs, treats all other
samples in the batch as negatives. Temperature τ controls
the sharpness of the distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        z_i, z_j: L2-normalized embeddings, shape (batch_size, embed_dim)
        Returns scalar loss.
        """
        batch_size = z_i.size(0)
        device = z_i.device

        # Concatenate to (2*batch_size, embed_dim)
        z = torch.cat([z_i, z_j], dim=0)

        # Pairwise cosine similarity matrix: (2B, 2B)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pair indices:
        # z_i[k] pairs with z_j[k] at position (k, k + batch_size)
        # z_j[k] pairs with z_i[k] at position (k + batch_size, k)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(batch_size, device=device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss
