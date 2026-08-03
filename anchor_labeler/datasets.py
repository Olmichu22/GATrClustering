"""
Dataset backends for the labeler.

A backend only has to answer three questions:

  * how many events are there            -> ``n_events``
  * summary arrays for the sampler       -> ``nhits()``, ``mask_for_source()``
  * the hits + metadata of ONE event     -> ``event(i)``

``hdf5_flat`` implements the flat/CSR layout produced by
``src/convert/root_to_flat_h5.py`` (offsets + flat per-hit arrays + per-event
arrays). Adding another format (ROOT, parquet, npz, ...) means writing one class
with the same three methods and registering it in ``BACKENDS``; no other module
knows about HDF5.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import h5py
import numpy as np

from .config import DatasetSpec


class FlatH5Dataset:
    """Read-only view of a flat (CSR) HDF5 event file."""

    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        self._f = h5py.File(spec.path, "r")
        self._offsets = np.asarray(self._f[spec.offsets][:]).astype(np.int64)
        self.n_events = int(self._offsets.size - 1)
        self._nhits: Optional[np.ndarray] = None

    # ---- summary level -------------------------------------------------

    def has_field(self, key: str) -> bool:
        return key in self._f

    def nhits(self) -> np.ndarray:
        """Per-event hit count (from the configured field, else from offsets)."""
        if self._nhits is None:
            fld = self.spec.nhits_field
            if fld and fld in self._f:
                self._nhits = np.asarray(self._f[fld][:]).astype(np.float64)
            else:
                self._nhits = (self._offsets[1:] - self._offsets[:-1]).astype(np.float64)
        return self._nhits

    def mask_for_source(self, source: Dict[str, Any]) -> np.ndarray:
        """Boolean per-event mask of the candidate pool described by ``source``.

        Supported source types (all config-driven, no field names hardcoded):
          {type: flag,  field: is_pion}                 -> field != 0
          {type: value, field: particle_type, value: 1} -> field == value
          {type: range, field: energy, min: .., max: ..}
          {type: all}
        """
        stype = str(source.get("type", "flag"))
        if stype == "all":
            return np.ones(self.n_events, dtype=bool)
        fld = source.get("field")
        if not fld:
            raise ValueError(f"source type '{stype}' needs a 'field'")
        if fld not in self._f:
            raise KeyError(f"field '{fld}' not present in {self.spec.path}")
        vals = np.asarray(self._f[fld][:])
        if vals.shape[0] != self.n_events:
            raise ValueError(f"field '{fld}' is per-hit, not per-event")
        if stype == "flag":
            return vals.astype(bool)
        if stype == "value":
            return vals.astype(np.int64) == int(source["value"])
        if stype == "range":
            lo = float(source.get("min", -np.inf))
            hi = float(source.get("max", np.inf))
            v = vals.astype(np.float64)
            return (v >= lo) & (v <= hi)
        raise ValueError(f"unknown source type '{stype}'")

    # ---- event level ---------------------------------------------------

    def event(self, index: int) -> Dict[str, Any]:
        """Hits and metadata of one event, JSON-ready."""
        i = int(index)
        if not (0 <= i < self.n_events):
            raise IndexError(f"event {i} out of range (n_events={self.n_events})")
        a, b = int(self._offsets[i]), int(self._offsets[i + 1])
        sp = self.spec

        coords: Dict[str, list] = {}
        for logical, key in sp.coords.items():
            coords[logical] = np.asarray(self._f[key][a:b]).astype(np.float64).tolist()

        color = None
        if sp.hit_color_field and sp.hit_color_field in self._f:
            color = np.asarray(self._f[sp.hit_color_field][a:b]).astype(np.float64).tolist()

        extras: Dict[str, list] = {}
        for label, key in (sp.hit_extra_fields or {}).items():
            if key in self._f:
                extras[label] = np.asarray(self._f[key][a:b]).astype(np.float64).tolist()

        meta: Dict[str, Any] = {"nHits": b - a}
        for label, key in (sp.event_fields or {}).items():
            if key in self._f:
                v = self._f[key][i]
                meta[label] = v.item() if hasattr(v, "item") else v
        return {
            "index": i,
            "coords": coords,
            "color": color,
            "hit_extra": extras,
            "meta": meta,
        }

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


BACKENDS = {"hdf5_flat": FlatH5Dataset}


def open_dataset(spec: DatasetSpec):
    if spec.backend not in BACKENDS:
        raise ValueError(f"unknown dataset backend '{spec.backend}' "
                         f"(known: {sorted(BACKENDS)})")
    return BACKENDS[spec.backend](spec)
