"""
Configurable routing of features to the GATr inputs, generalizing
``GATrAutoencoder/src/utils/batch_utils.py::build_batch`` (which was hardcoded).

Config (``features``):
    mv_point           : 3 names -> multivector point (embed_point)
    mv_scalar          : 0..n names -> geometric scalar (embed_scalar, summed)
    scalars            : hit-level names -> GATr SCALAR INPUT (in_s_channels)
    thr_encoding       : 'ordinal' (thr as 1/2/3) | 'one_hot' (thr1,thr2,thr3)
    extra_event_scalars: event-level names -> broadcast to hits, SAME scalar input
"""

from __future__ import annotations

from typing import Dict, List

import torch


def expand_scalar_names(features_cfg: dict) -> List[str]:
    """Scalar column names (hit-level) after applying thr_encoding."""
    thr_encoding = features_cfg.get("thr_encoding", "ordinal")
    names: List[str] = []
    for s in features_cfg.get("scalars", []):
        if s == "thr" and thr_encoding == "one_hot":
            names += ["thr1", "thr2", "thr3"]
        else:
            names.append(s)
    return names


def compute_in_s_channels(features_cfg: dict) -> int:
    return len(expand_scalar_names(features_cfg)) + len(features_cfg.get("extra_event_scalars", []))


def compute_in_mv_channels(mv_embedding_mode: str) -> int:
    return 2 if mv_embedding_mode == "centroid" else 1


def build_inputs(batch, features_cfg: dict) -> Dict[str, torch.Tensor]:
    """Convert a PyG Batch into the tensors expected by the GATr encoder."""
    device = batch.pos.device
    N = batch.pos.shape[0]
    batch_idx = batch.batch

    # ---- multivector point ----
    mv_v_part = batch.pos  # (N,3), built from features.mv_point in the dataset

    # ---- geometric scalar (embed_scalar) ----
    mv_scalar_names = features_cfg.get("mv_scalar", [])
    if mv_scalar_names:
        cols = [batch[name] for name in mv_scalar_names]  # each (N,1)
        mv_s_part = torch.cat(cols, dim=1)  # (N, S)
    else:
        mv_s_part = torch.zeros((N, 1), dtype=torch.float32, device=device)

    # ---- GATr scalar input (hit-level + event-level extras) ----
    scalar_cols = [batch[name] for name in expand_scalar_names(features_cfg)]  # each (N,1)
    for name in features_cfg.get("extra_event_scalars", []):
        ev = batch[name]  # (B,) event-level
        scalar_cols.append(ev[batch_idx].view(-1, 1))
    scalars = (
        torch.cat(scalar_cols, dim=1)
        if scalar_cols
        else torch.zeros((N, 0), dtype=torch.float32, device=device)
    )

    return {
        "mv_v_part": mv_v_part,
        "mv_s_part": mv_s_part,
        "scalars": scalars,
        "batch_idx": batch_idx,
    }
