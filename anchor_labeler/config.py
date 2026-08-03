"""
Configuration for the manual anchor labeler.

Everything the app knows about a detector lives in ONE yaml file: which h5 to
open, which hit/event fields to read, which classes exist and how a candidate
for each class is proposed. Nothing here is SDHCAL-specific -- pointing the
config at another flat (CSR) file with different field names is enough to label
a different detector/dataset.

See ``configs/labeler_sdhcal.yml`` for a documented example.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

# Fallback palette (tab10), used when a class does not set its own colour.
DEFAULT_CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


@dataclass
class ClassSpec:
    """One labelable class (e.g. electron)."""

    index: int
    name: str
    color: str
    # How events are PROPOSED for this class. ``None`` -> the class can still be
    # chosen by hand during review, it is just never auto-sampled.
    source: Optional[Dict[str, Any]] = None
    key: Optional[str] = None          # keyboard shortcut, defaults to str(index)

    @property
    def samplable(self) -> bool:
        return self.source is not None


@dataclass
class DatasetSpec:
    """Where the events live and how to read them."""

    path: str
    backend: str = "hdf5_flat"
    offsets: str = "offsets"
    # logical -> h5 key for the 3 hit coordinates
    coords: Dict[str, str] = field(default_factory=lambda: {"x": "x", "y": "y", "z": "z"})
    # per-hit field used to colour the 3D view (e.g. threshold); optional
    hit_color_field: Optional[str] = None
    hit_color_label: str = "value"
    # extra per-hit fields shown on hover
    hit_extra_fields: Dict[str, str] = field(default_factory=dict)
    # per-event fields shown in the info panel: label -> h5 key
    event_fields: Dict[str, str] = field(default_factory=dict)
    # per-event hit-count field; null -> derived from offsets
    nhits_field: Optional[str] = None
    # which data axis points RIGHT on screen (beam axis for a test beam)
    beam_axis: str = "z"


@dataclass
class SamplingSpec:
    strategy: str = "nhits_band"
    n_per_class: int = 25
    seed: int = 42
    window_std: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_per_class": self.n_per_class,
            "seed": self.seed,
            "window_std": self.window_std,
            **self.params,
        }


@dataclass
class LabelerConfig:
    name: str
    dataset: DatasetSpec
    classes: List[ClassSpec]
    sampling: SamplingSpec
    session_path: str
    export_dir: str
    autosave_seconds: float = 5.0
    raw: Dict[str, Any] = field(default_factory=dict)

    def class_by_index(self, idx: int) -> Optional[ClassSpec]:
        for c in self.classes:
            if c.index == int(idx):
                return c
        return None

    def public(self) -> Dict[str, Any]:
        """JSON-safe view handed to the browser."""
        return {
            "name": self.name,
            "dataset": {
                "path": self.dataset.path,
                "coords": self.dataset.coords,
                "beam_axis": self.dataset.beam_axis,
                "hit_color_label": self.dataset.hit_color_label,
            },
            "classes": [
                {
                    "index": c.index,
                    "name": c.name,
                    "color": c.color,
                    "key": c.key or str(c.index),
                    "samplable": c.samplable,
                }
                for c in self.classes
            ],
            "sampling": self.sampling.as_dict(),
            "session_path": self.session_path,
            "autosave_seconds": self.autosave_seconds,
        }


def _abspath(p: str, base: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def load_config(path: str) -> LabelerConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    base = os.path.dirname(os.path.abspath(path))

    ds_raw = dict(raw.get("dataset") or {})
    if "path" not in ds_raw:
        raise ValueError(f"{path}: dataset.path is required")
    ds_raw["path"] = _abspath(ds_raw["path"], base)
    known = {f.name for f in DatasetSpec.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(ds_raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown dataset keys {sorted(unknown)}")
    dataset = DatasetSpec(**ds_raw)

    classes: List[ClassSpec] = []
    for i, c in enumerate(raw.get("classes") or []):
        idx = int(c["index"]) if "index" in c else i
        classes.append(
            ClassSpec(
                index=idx,
                name=str(c.get("name", f"class{idx}")),
                color=str(c.get("color") or DEFAULT_CLASS_COLORS[idx % len(DEFAULT_CLASS_COLORS)]),
                source=c.get("source"),
                key=str(c["key"]) if c.get("key") is not None else None,
            )
        )
    if not classes:
        raise ValueError(f"{path}: at least one class is required")

    s_raw = dict(raw.get("sampling") or {})
    sampling = SamplingSpec(
        strategy=s_raw.pop("strategy", "nhits_band"),
        n_per_class=int(s_raw.pop("n_per_class", 25)),
        seed=int(s_raw.pop("seed", 42)),
        window_std=float(s_raw.pop("window_std", 1.0)),
        params=s_raw,
    )

    session_path = _abspath(raw.get("session_path") or "sessions/session.json", base)
    export_dir = _abspath(raw.get("export_dir") or "sessions/exports", base)

    return LabelerConfig(
        name=str(raw.get("name") or os.path.splitext(os.path.basename(path))[0]),
        dataset=dataset,
        classes=classes,
        sampling=sampling,
        session_path=session_path,
        export_dir=export_dir,
        autosave_seconds=float(raw.get("autosave_seconds", 5.0)),
        raw=copy.deepcopy(raw),
    )
