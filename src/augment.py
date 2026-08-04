"""
Minimal augmentation for the POC: per-hit dropout (10-15%).

Produces a masked view of a PyG Batch by randomly removing hits while
(a) preserving the per-hit event index and (b) guaranteeing every event keeps
at least one hit (so no event becomes empty). Two independent dropout views of
the same batch feed the swap loss. Jitter / layer subsampling are intentionally
left out of this first iteration.
"""

from __future__ import annotations

import copy

import torch

# Per-event attributes that must NOT be filtered as node-level tensors.
_EVENT_KEYS = {"energy", "anchor_label", "class_label", "nhits_total", "ptr", "batch"}


def hit_dropout(batch, p: float):
    """Return a new Batch with a random subset of hits kept (>=1 per event)."""
    if p <= 0.0:
        return batch

    device = batch.pos.device
    N = batch.pos.shape[0]
    b = batch.batch
    B = int(b.max().item()) + 1

    keep = torch.rand(N, device=device) > p

    # Guarantee at least one kept hit per event.
    kept_per_event = torch.zeros(B, device=device).scatter_add_(
        0, b, keep.float()
    )
    empty = (kept_per_event == 0).nonzero(as_tuple=True)[0]
    if empty.numel() > 0:
        # first hit index of each event (batch is contiguous by event)
        first_idx = torch.zeros(B, dtype=torch.long, device=device)
        ones = torch.ones(N, device=device)
        counts = torch.zeros(B, device=device).scatter_add_(0, b, ones).long()
        first_idx[1:] = torch.cumsum(counts, 0)[:-1]
        keep[first_idx[empty]] = True

    new = copy.copy(batch)
    for key, val in batch.items():
        if torch.is_tensor(val) and key not in _EVENT_KEYS and val.size(0) == N:
            new[key] = val[keep]
    new.batch = b[keep]

    # recompute ptr from the new per-event counts
    new_counts = torch.bincount(new.batch, minlength=B)
    ptr = torch.zeros(B + 1, dtype=torch.long, device=device)
    torch.cumsum(new_counts, 0, out=ptr[1:])
    new.ptr = ptr
    return new
