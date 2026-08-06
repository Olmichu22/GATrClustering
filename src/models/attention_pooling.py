"""
Attention pooling (PMA), forked from
``GATrAutoencoder/src/models/attention_pooling.py``.

Learnable seed queries attend over the (variable-length, PyG-batched) hit tokens
of each event to produce a fixed-size per-event summary. It is one-directional:
seeds attend to hits, hits never attend to the seeds.
"""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_seeds=1, dropout=0.0):
        super().__init__()
        self.num_seeds = num_seeds
        self.seed = nn.Parameter(torch.randn(1, num_seeds, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.out_dim = num_seeds * embed_dim

    def forward(self, x, batch, layer=None):
        """x: (N_total, D) tokens; batch: (N_total,) event indices. Returns (B, num_seeds*D).

        ``layer`` is accepted (and ignored) so every aggregation shares one call
        signature; only AttentionDensityPooling actually uses it."""
        device = x.device
        counts = torch.bincount(batch)
        B = counts.shape[0]
        max_len = int(counts.max().item())
        D = x.shape[-1]

        padded = torch.zeros(B, max_len, D, device=device)
        key_padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        torch.cumsum(counts, dim=0, out=offsets[1:])
        for i in range(B):
            length = int(counts[i].item())
            padded[i, :length] = x[offsets[i]:offsets[i + 1]]
            key_padding_mask[i, :length] = False

        query = self.norm_q(self.seed.expand(B, -1, -1))
        kv = self.norm_k(padded)
        out, _ = self.attn(query, kv, kv, key_padding_mask=key_padding_mask)
        return out.reshape(B, -1)
