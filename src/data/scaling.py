"""
Configurable per-feature scaling, generalizing the global logic in
``GATrAutoencoder/src/utils/datasets.py`` (which used a single ``norm_type``).

Each feature has its own ``mode``:
    z_norm : (x - mean) / std
    minmax : (x - min) / (max - min)
    log    : log(x + eps)          (no stats required)
    none   : unchanged

Statistics source (``source``):
    online  : computed ONLY over the train-split hits/events and SAVED to a YAML
              for reuse.
    file    : loaded from an existing YAML.
    dataset : nothing is touched (the h5 is assumed already normalized).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml


def derive_stats_path(dataset_path: str) -> str:
    p = Path(dataset_path)
    return str(p.parent / f"{p.stem}_stats.yml")


def compute_feature_stats(
    values_by_feature: Dict[str, np.ndarray],
) -> Dict[str, dict]:
    """mean/std/min/max/count per feature from already-masked (train) arrays."""
    stats: Dict[str, dict] = {}
    for feat, arr in values_by_feature.items():
        a = np.asarray(arr, dtype=np.float64)
        stats[feat] = {
            "mean": float(a.mean()),
            "std": float(max(a.std(), 1e-8)),
            "min": float(a.min()),
            "max": float(a.max()),
            "count": int(a.size),
        }
    return stats


class FeatureScaler:
    """Apply per-feature scaling; manages loading/computing/saving of stats."""

    def __init__(self, scaling_cfg: dict, dataset_path: str):
        self.cfg = scaling_cfg or {}
        self.source = self.cfg.get("source", "online")
        self.modes: Dict[str, str] = {
            feat: (spec or {}).get("mode", "none")
            for feat, spec in (self.cfg.get("features", {}) or {}).items()
        }
        self.stats_path = self.cfg.get("stats_path") or derive_stats_path(dataset_path)
        self.stats: Dict[str, dict] = {}

    # ---- stats management ----
    def needs_stats(self) -> bool:
        """True if any feature uses z_norm/minmax (log/none need no stats)."""
        return any(m in ("z_norm", "minmax") for m in self.modes.values())

    def load_stats(self) -> Dict[str, dict]:
        with open(self.stats_path, "r") as fh:
            raw = yaml.safe_load(fh)
        self.stats = raw.get("features", raw) if isinstance(raw, dict) else {}
        print(f"[FeatureScaler] Loaded stats from '{self.stats_path}'")
        return self.stats

    def save_stats(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.stats_path)), exist_ok=True)
        payload = {
            "source": "online",
            "modes": self.modes,
            "features": self.stats,
        }
        with open(self.stats_path, "w") as fh:
            yaml.dump(payload, fh, default_flow_style=False, sort_keys=False)
        print(f"[FeatureScaler] Saved stats to '{self.stats_path}'")

    def resolve_stats(
        self,
        train_values_by_feature: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Populate ``self.stats`` according to ``source``."""
        if self.source == "dataset" or not self.needs_stats():
            self.stats = {}
            return
        if self.source == "file":
            self.load_stats()
            return
        # online
        if os.path.exists(self.stats_path):
            self.load_stats()
            return
        if train_values_by_feature is None:
            raise ValueError("source=online without prior stats requires train values")
        feats_needed = {
            f: v for f, v in train_values_by_feature.items()
            if self.modes.get(f) in ("z_norm", "minmax")
        }
        self.stats = compute_feature_stats(feats_needed)
        self.save_stats()

    # ---- application ----
    def apply(self, feat: str, arr: np.ndarray) -> np.ndarray:
        mode = self.modes.get(feat, "none")
        if mode == "none":
            return arr
        if mode == "log":
            return np.log(arr + 1e-6)
        s = self.stats.get(feat)
        if s is None:
            return arr
        if mode == "z_norm":
            return (arr - s["mean"]) / s["std"]
        if mode == "minmax":
            rng = s["max"] - s["min"]
            return (arr - s["min"]) / (rng if rng > 1e-8 else 1.0)
        raise ValueError(f"Unknown scaling mode for '{feat}': {mode}")
