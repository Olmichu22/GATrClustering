"""
Switchable event-level aggregation of per-hit tokens (config: aggregation.type).

    mean      -> scatter_mean over an event's hit tokens (simplest baseline).
    attention -> AttentionPooling (PMA) with learnable seeds.
    token     -> a single learnable "aggregation token" that attends to all hits
                 of its event while NO hit attends to it (one-directional).
    attention_global / attention_card / attention_density
              -> AttentionGlobalPooling: attention + magnitude branch + a
                 CONFIGURABLE list of per-event scalars (aggregation.
                 global_scalars). The three names differ only in their default
                 list, kept for backward compatibility:
                     attention_card    -> [log_nhits]
                     attention_density -> [density]
                     attention_global  -> [density]  (pensado para darla explícita)
                 e.g. `global_scalars: [density, log_nhits, energy]`.

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


#: Per-event scalars that the pooling can DERIVE by itself (everything else is
#: looked up on the batch, where the dataset put it already scaled).
DERIVED_GLOBAL_SCALARS = ("log_nhits", "density")


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


class AttentionGlobalPooling(nn.Module):
    """Attention pooling AUGMENTED with magnitude + a CONFIGURABLE list of
    per-event (global) scalars.

    Softmax attention (and mean/token pooling) produce a *weighted average* of
    the hit tokens, so they are blind to the number of hits and to the total
    deposited magnitude -- exactly the axis that separates a compact EM shower
    from a MIP. Two branches the pooling would otherwise discard are appended:

      * sqrt-normalized SUM of tokens -> magnitude, grows sub-linearly with size
        (D dims, always present)
      * ``global_scalars``            -> one column each, in the given order

    Available names (``global_scalars``):
        log_nhits : log1p(nHits). The raw cardinality. NOT scaled here: it enters
                    at ~O(7), so it belongs in the SHARED LayerNorm
                    (separate_norm: false) unless you scale it upstream.
        density   : hits per ACTIVE layer. Raw nHits grows with how DEEP a shower
                    goes, i.e. with the detector depth; dividing by the distinct
                    active layers gives transverse compactness instead. Taken
                    from ``data.density`` (already scaled by FeatureScaler) when
                    the batch carries it, else derived here from the layer index.
        energy    : beam energy, from ``data.energy`` (scaled by FeatureScaler,
                    e.g. log_z). This is the GLOBAL way to feed the energy: it
                    conditions the event embedding, as opposed to
                    ``features.extra_event_scalars``, which broadcasts the value
                    to every hit and feeds it to the ENCODER instead.
        <other>   : any other per-event field present on the batch, used as-is.

    Normalization (``separate_norm``):
        false -> ONE LayerNorm over concat(magnitude, scalars). The historical
                 behavior. Careful with heavy-tailed raw scalars: the density
                 ratio is bimodal (median ~1.9 hits/layer for MIPs, ~23 for the
                 10% that shower, reaching ~46), so its tail moves the
                 normalizer's mean/variance and squashes the magnitude branch
                 precisely on the events that matter.
        true  -> LayerNorm(D) on the magnitude branch only; the scalars are
                 concatenated untouched, which assumes they are already scaled
                 upstream (true for ``density`` and ``energy``, NOT for
                 ``log_nhits``).

    Note LayerNorm over the scalars alone is not an option: with a single column
    it maps every value to 0.
    """

    def __init__(self, embed_dim, num_heads=4, num_seeds=1, dropout=0.0,
                 separate_norm=False, global_scalars=("density",)):
        super().__init__()
        self.attn = AttentionPooling(embed_dim, num_heads, num_seeds, dropout)
        self.separate_norm = bool(separate_norm)
        self.global_scalars = [str(s) for s in (global_scalars or [])]
        n_extra = len(self.global_scalars)
        # Attribute names kept as `attn` / `extra_norm` on purpose: checkpoints of
        # earlier runs (attention_card / attention_density) load unchanged.
        self.extra_norm = nn.LayerNorm(embed_dim if self.separate_norm
                                       else embed_dim + n_extra)
        self.out_dim = self.attn.out_dim + embed_dim + n_extra

    def forward(self, x, batch, layer=None, event_scalars=None):
        event_scalars = event_scalars or {}
        a = self.attn(x, batch)                                   # (B, S*D) event shape
        counts = torch.bincount(batch).clamp(min=1).to(x.dtype)   # (B,)
        s = scatter_add(x, batch, dim=0) / counts.sqrt().unsqueeze(1)  # (B, D) magnitude

        cols = []
        for name in self.global_scalars:
            val = event_scalars.get(name)
            if val is not None:
                cols.append(val.view(-1, 1).to(x.dtype))
            elif name == "log_nhits":
                cols.append(torch.log1p(counts).unsqueeze(1))
            elif name == "density":
                # Fallback when the dataset did not precompute it: raw ratio.
                if layer is None:
                    raise ValueError(
                        "global scalar 'density' needs either a per-event "
                        "'density' on the batch or per-hit layer indices ('k')."
                    )
                nlayers = _layers_per_event(layer, batch, counts.shape[0])
                cols.append((counts / nlayers.clamp(min=1).to(x.dtype)).unsqueeze(1))
            else:
                raise ValueError(
                    f"aggregation.global_scalars names '{name}', which is neither "
                    f"a per-event field on the batch nor derivable "
                    f"{DERIVED_GLOBAL_SCALARS}."
                )

        if not cols:
            # No scalars -> extra_norm is LayerNorm(D) either way.
            extras = self.extra_norm(s)
        elif self.separate_norm:
            extras = torch.cat([self.extra_norm(s)] + cols, dim=1)
        else:
            extras = self.extra_norm(torch.cat([s] + cols, dim=1))
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
    # attention_card / attention_density son ahora ALIAS del pooling genérico con
    # una lista fija de escalares globales: reproducen exactamente el
    # comportamiento anterior (mismos parámetros, mismos nombres en el
    # state_dict), y `global_scalars` permite pedir otra combinación —incluida la
    # energía— sin tocar el código.
    if agg_type in ("attention_card", "attention+card",
                    "attention_density", "attention+density",
                    "attention_global", "attention+global"):
        default_scalars = ["log_nhits"] if agg_type.endswith("card") else ["density"]
        return AttentionGlobalPooling(
            embed_dim=embed_dim,
            num_heads=cfg_agg.get("num_heads", 4),
            num_seeds=cfg_agg.get("num_seeds", 1),
            dropout=cfg_agg.get("dropout", 0.0),
            separate_norm=cfg_agg.get("separate_norm", False),
            global_scalars=cfg_agg.get("global_scalars", default_scalars),
        )
    if agg_type == "token":
        return AggregationToken(
            embed_dim=embed_dim,
            num_heads=cfg_agg.get("num_heads", 4),
            dropout=cfg_agg.get("dropout", 0.0),
        )
    raise ValueError(f"Unknown aggregation.type: {agg_type}")
