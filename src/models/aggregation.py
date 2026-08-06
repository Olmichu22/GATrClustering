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
from torch_scatter import scatter_add, scatter_mean

from .attention_pooling import AttentionPooling


class MeanAggregation(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.out_dim = embed_dim

    def forward(self, x, batch, layer=None):
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

    def forward(self, x, batch, layer=None):
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


class AttentionCardinalityPooling(nn.Module):
    """Attention pooling AUGMENTED with cardinality/magnitude features.

    Softmax attention (and mean/token pooling) produce a *weighted average* of
    the hit tokens, so they are blind to the number of hits and the total
    deposited magnitude -- exactly the axis that separates a compact EM shower
    (~500 hits) from a MIP (~76 hits). Here we keep the attention summary (event
    "shape") and concatenate two size-aware branches the pooling otherwise
    discards:
        * sqrt-normalized SUM of tokens  -> grows with size, sub-linearly
        * log1p(nHits)                   -> the raw cardinality (intrinsic, not a
                                            hand-crafted physics variable)
    The size-aware branch is LayerNorm'd so it does not dominate the (unit-scale)
    attention output before the Linear head.
    """

    def __init__(self, embed_dim, num_heads=4, num_seeds=1, dropout=0.0):
        super().__init__()
        self.attn = AttentionPooling(embed_dim, num_heads, num_seeds, dropout)
        self.extra_norm = nn.LayerNorm(embed_dim + 1)  # [sum branch (D), log count (1)]
        self.out_dim = self.attn.out_dim + embed_dim + 1

    def forward(self, x, batch, layer=None):
        a = self.attn(x, batch)                                   # (B, S*D) event shape
        counts = torch.bincount(batch).clamp(min=1).to(x.dtype)   # (B,)
        s = scatter_add(x, batch, dim=0) / counts.sqrt().unsqueeze(1)  # (B, D) magnitude
        c = torch.log1p(counts).unsqueeze(1)                      # (B, 1) cardinality
        extras = self.extra_norm(torch.cat([s, c], dim=1))
        return torch.cat([a, extras], dim=1)


def _layers_per_event(layer: torch.Tensor, batch: torch.Tensor, B: int) -> torch.Tensor:
    """Number of DISTINCT active layers per event.

    ``layer`` is the per-hit layer coordinate (``k``; may be min-max scaled --
    monotone scaling preserves distinctness). We discretize to absorb float
    noise, then count unique (event, layer) pairs per event.
    """
    lay = torch.round(layer.view(-1).to(torch.float64) * 1e4).to(torch.int64)
    lay = lay - lay.min()                                   # >= 0
    base = int(lay.max().item()) + 1
    comb = batch.to(torch.int64) * base + lay               # unique per (event, layer)
    uniq = torch.unique(comb)
    ev = uniq // base
    return torch.bincount(ev, minlength=B)                  # (B,) distinct layers


class AttentionDensityPooling(nn.Module):
    """Like :class:`AttentionCardinalityPooling` but the size scalar is DENSITY
    (hits per ACTIVE LAYER) instead of ``log1p(nHits)``.

    Raw ``nHits`` grows with how many layers a shower traverses, i.e. with
    detector DEPTH -- a give-away tied to the number of layers. Dividing by the
    number of distinct active layers yields the transverse density (hits/layer),
    which describes shower COMPACTNESS independently of how deep it goes, making
    the event descriptor agnostic to the number of layers. The sqrt-normalized
    SUM branch (magnitude) is kept unchanged; only the count scalar is swapped.

    Normalization of the density scalar (``separate_norm``): the raw ratio is
    heavy-tailed and bimodal -- median ~1.9 hits/layer for MIPs but ~23 for the
    10% of events that shower, reaching ~46. Sharing one LayerNorm with the D
    magnitude features lets that tail move the normalizer's mean/variance and
    squash the magnitude branch precisely on the events that matter, and it
    enters at scale ~20 against unit-scale tokens. With ``separate_norm`` the
    magnitude branch gets its own LayerNorm(D) and the density is taken already
    scaled from the batch (``data.density``, normalized by FeatureScaler with
    stats measured online over the train split), so the two cannot interfere.
    Default is False: the original behavior, so earlier runs stay reproducible.
    """

    def __init__(self, embed_dim, num_heads=4, num_seeds=1, dropout=0.0,
                 separate_norm=False):
        super().__init__()
        self.attn = AttentionPooling(embed_dim, num_heads, num_seeds, dropout)
        self.separate_norm = bool(separate_norm)
        # separate: LayerNorm(D) on the magnitude branch only; the density is
        # already normalized upstream and is concatenated untouched.
        self.extra_norm = nn.LayerNorm(embed_dim if self.separate_norm else embed_dim + 1)
        self.out_dim = self.attn.out_dim + embed_dim + 1

    def forward(self, x, batch, layer=None, density=None):
        if layer is None and density is None:
            raise ValueError(
                "aggregation.type 'attention_density' needs per-hit layer indices "
                "(field 'k') or a precomputed per-event 'density' on the batch."
            )
        a = self.attn(x, batch)                                   # (B, S*D) event shape
        counts = torch.bincount(batch).clamp(min=1).to(x.dtype)   # (B,)
        s = scatter_add(x, batch, dim=0) / counts.sqrt().unsqueeze(1)  # (B, D) magnitude
        if density is not None:
            d = density.view(-1, 1).to(x.dtype)                   # (B, 1) already scaled
        else:
            nlayers = _layers_per_event(layer, batch, counts.shape[0]).clamp(min=1).to(x.dtype)
            d = (counts / nlayers).unsqueeze(1)                   # (B, 1) raw hits/layer
        if self.separate_norm:
            extras = torch.cat([self.extra_norm(s), d], dim=1)
        else:
            extras = self.extra_norm(torch.cat([s, d], dim=1))
        return torch.cat([a, extras], dim=1)


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
    if agg_type in ("attention_card", "attention+card"):
        return AttentionCardinalityPooling(
            embed_dim=embed_dim,
            num_heads=cfg_agg.get("num_heads", 4),
            num_seeds=cfg_agg.get("num_seeds", 1),
            dropout=cfg_agg.get("dropout", 0.0),
        )
    if agg_type in ("attention_density", "attention+density"):
        return AttentionDensityPooling(
            embed_dim=embed_dim,
            num_heads=cfg_agg.get("num_heads", 4),
            num_seeds=cfg_agg.get("num_seeds", 1),
            dropout=cfg_agg.get("dropout", 0.0),
            separate_norm=cfg_agg.get("separate_norm", False),
        )
    if agg_type == "token":
        return AggregationToken(
            embed_dim=embed_dim,
            num_heads=cfg_agg.get("num_heads", 4),
            dropout=cfg_agg.get("dropout", 0.0),
        )
    raise ValueError(f"Unknown aggregation.type: {agg_type}")
