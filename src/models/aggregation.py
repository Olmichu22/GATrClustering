"""
Switchable event-level aggregation of per-hit tokens (config: aggregation.type).

    mean      -> scatter_mean over an event's hit tokens (simplest baseline).
    attention -> AttentionPooling (PMA) with learnable seeds.
    token     -> a single learnable "aggregation token" that attends to all hits
                 of its event while NO hit attends to it (one-directional).

Design note on ``token``:
The requested semantics are "an aggregation token that attends to all hits, but
no hit attends to it." Realizing that strictly *inside* GATr's self-attention
would require a dense (N,N) additive mask per batch to break the intra-event
symmetry; with thousands of hits per event that is memory-infeasible (GATr uses
a block-diagonal xformers mask precisely to avoid it). We therefore realize the
exact same asymmetry as a post-encoder cross-attention: the token is the query,
the encoded hit tokens are keys/values, and attention is one-directional by
construction (hits never see the token). This keeps the intended inductive bias
(the token does not perturb hit representations) and is memory-feasible.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from .attention_pooling import AttentionPooling


class MeanAggregation(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.out_dim = embed_dim

    def forward(self, x, batch):
        return scatter_mean(x, batch, dim=0)


class AggregationToken(nn.Module):
    """Single learnable aggregation token; one-directional cross-attention."""

    def __init__(self, embed_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

    def forward(self, x, batch):
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

        query = self.norm_q(self.token.expand(B, -1, -1))  # (B,1,D)
        kv = self.norm_k(padded)
        out, _ = self.attn(query, kv, kv, key_padding_mask=key_padding_mask)  # (B,1,D)
        return out.squeeze(1)


def build_aggregation(cfg_agg: dict, embed_dim: int) -> nn.Module:
    agg_type = cfg_agg.get("type", "mean")
    if agg_type == "mean":
        return MeanAggregation(embed_dim)
    if agg_type == "attention":
        return AttentionPooling(
            embed_dim=embed_dim,
            num_heads=cfg_agg.get("num_heads", 4),
            num_seeds=cfg_agg.get("num_seeds", 1),
            dropout=cfg_agg.get("dropout", 0.0),
        )
    if agg_type == "token":
        return AggregationToken(
            embed_dim=embed_dim,
            num_heads=cfg_agg.get("num_heads", 4),
            dropout=cfg_agg.get("dropout", 0.0),
        )
    raise ValueError(f"Unknown aggregation.type: {agg_type}")
