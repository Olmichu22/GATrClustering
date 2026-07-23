"""
Prototype head for the clustering POC.

    event embedding -> Linear projector -> L2-normalized z in R^d
    logits_k = cos(z, c_k) / tau     with learnable prototypes c_k (k=1..K)

Prototypes are initialized from the mean embedding of the anchors of each class
(non-random init), which stabilizes the anchor CE term from the first step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int, num_clusters: int, temperature: float = 0.1):
        super().__init__()
        self.projector = nn.Linear(in_dim, proj_dim)
        self.prototypes = nn.Parameter(torch.randn(num_clusters, proj_dim))
        self.temperature = temperature
        self.num_clusters = num_clusters
        self.proj_dim = proj_dim

    def project(self, event_embedding: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized z."""
        z = self.projector(event_embedding)
        return F.normalize(z, dim=-1)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        c = F.normalize(self.prototypes, dim=-1)
        return z @ c.t() / self.temperature

    def forward(self, event_embedding: torch.Tensor):
        z = self.project(event_embedding)
        return z, self.logits(z)

    @torch.no_grad()
    def init_prototypes_from_anchors(self, z_anchor: torch.Tensor, anchor_labels: torch.Tensor):
        """Set prototype k = mean z over anchors with label k (skips empty classes)."""
        for k in range(self.num_clusters):
            mask = anchor_labels == k
            if mask.any():
                self.prototypes[k] = F.normalize(z_anchor[mask].mean(0), dim=-1)
