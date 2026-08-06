"""
Flat (CSR) HDF5 reader decoupled through a ``field_map``.

FIXED structural contract (does not change across datasets):
    - ``offsets``: int array of length ``n_events + 1``, monotonically
      non-decreasing, with ``offsets[0] == 0`` and ``offsets[-1] == n_hits``.
    - per-hit arrays: length ``n_hits`` (concatenation over all events).
    - per-event arrays: length ``n_events``.

The h5 BRANCH NAMES do change across datasets; they are remapped through
``field_map`` (logical_name -> real_key), so the rest of the pipeline always
works with logical names (x, y, z, k, thr, energy, anchor_label, ...).

Flat ``.npz`` files (with an ``offsets`` array) are also supported.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np


def apply_hit_affine(
    hit: Dict[str, np.ndarray],
    affine: Optional[Dict[str, object]],
    tag: str = "",
) -> None:
    """In-place ``field -> scale*field + offset`` over a per-hit array dict.

    Unit/origin harmonization. Lives here (and not only in the dataset) because
    when several files are concatenated they may NOT share a frame, so the map
    has to be applied per source BEFORE the concatenation — see
    :class:`MultiFlatEventReader`.
    """
    for name, ab in (affine or {}).items():
        if name not in hit:
            raise KeyError(
                f"hit_affine names '{name}', not a hit field (have: {sorted(hit)})"
            )
        scale, offset = (float(v) for v in ab)
        hit[name] = (hit[name].astype(np.float32) * scale + offset).astype(np.float32)
        where = f" [{tag}]" if tag else ""
        print(f"[hit_affine]{where} {name} -> {scale:g}*{name} + {offset:g}")


class FlatEventReader:
    """Load per-hit and per-event arrays from a flat h5/npz using a ``field_map``."""

    def __init__(self, path: str, field_map: dict):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        self.path = path
        self.field_map = field_map

        hit_map: Dict[str, str] = dict(field_map.get("hit", {}) or {})
        event_map: Dict[str, str] = dict(field_map.get("event", {}) or {})
        offsets_key: str = field_map.get("offsets", "offsets")

        ext = os.path.splitext(path)[1].lower()
        self._is_hdf5 = ext in (".h5", ".hdf5")

        if self._is_hdf5:
            import h5py

            f = h5py.File(path, "r")
            available = set(f.keys())
        else:
            f = np.load(path, allow_pickle=False)
            available = set(f.files if hasattr(f, "files") else f.keys())

        def _load(key: str) -> np.ndarray:
            return np.asarray(f[key])

        # ---- offsets (required) ----
        if offsets_key not in available:
            raise KeyError(
                f"'{offsets_key}' (offsets) not in {path}. Keys: {sorted(available)}"
            )
        self.offsets = _load(offsets_key).astype(np.int64)
        self._validate_offsets()

        n_events = len(self.offsets) - 1
        n_hits = int(self.offsets[-1])

        # ---- per-hit arrays (those listed in field_map.hit are required) ----
        self.hit: Dict[str, np.ndarray] = {}
        for logical, real in hit_map.items():
            if real not in available:
                raise KeyError(
                    f"Per-hit branch '{real}' (logical '{logical}') not in {path}. "
                    f"Available keys: {sorted(available)}"
                )
            arr = _load(real)
            if arr.shape[0] != n_hits:
                raise ValueError(
                    f"Per-hit branch '{real}' has len {arr.shape[0]} != n_hits {n_hits}"
                )
            self.hit[logical] = arr.astype(np.float32)

        # ---- per-event arrays (all OPTIONAL; None if missing or key is null) ----
        self.event: Dict[str, Optional[np.ndarray]] = {}
        for logical, real in event_map.items():
            if real is None or real not in available:
                self.event[logical] = None
                continue
            arr = _load(real)
            if arr.shape[0] != n_events:
                raise ValueError(
                    f"Per-event branch '{real}' has len {arr.shape[0]} != n_events {n_events}"
                )
            self.event[logical] = arr

        if self._is_hdf5:
            f.close()

        self.n_events = n_events
        self.n_hits = n_hits

    def _validate_offsets(self) -> None:
        off = self.offsets
        if off.ndim != 1 or len(off) < 2:
            raise ValueError(f"'offsets' must be 1D with >=2 elements (shape={off.shape})")
        if off[0] != 0:
            raise ValueError(f"offsets[0] must be 0 (got {off[0]})")
        if np.any(np.diff(off) < 0):
            raise ValueError("'offsets' must be monotonically non-decreasing")

    def nhits_per_event(self) -> np.ndarray:
        """Number of hits per event, derived from offsets."""
        return (self.offsets[1:] - self.offsets[:-1]).astype(np.int64)

    def hit_to_event_index(self) -> np.ndarray:
        """Map hit -> event index (vectorized with np.repeat)."""
        return np.repeat(np.arange(self.n_events, dtype=np.int64), self.nhits_per_event())


def _concat_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate ``arange(start, start+length)`` for every (start, length) pair.

    Vectorized (no Python loop) so it stays cheap even with tens of thousands of
    variable-length events. Used to gather the hits of a subset of events.
    """
    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    idx = np.ones(total, dtype=np.int64)
    seg_starts = np.cumsum(lengths)[:-1]  # start offset of each segment (from 2nd)
    idx[0] = starts[0]
    # at each new segment, jump from the end of the previous range to the new start
    idx[seg_starts] = starts[1:] - (starts[:-1] + lengths[:-1]) + 1
    return np.cumsum(idx)


class MultiFlatEventReader:
    """Concatenate several flat readers into a single reader-compatible object.

    Exposes the SAME duck-typed interface as :class:`FlatEventReader`
    (``offsets``, ``n_events``, ``n_hits``, ``hit``, ``event``,
    ``nhits_per_event``, ``path``), so the rest of the pipeline (dataset,
    scaling, evaluation) is agnostic to how many files back it.

    Each source is a dict:
        path         : h5/npz path
        field_map    : logical->real remap for this source
        force_anchor : int | None -- if set, ALL events of this source get this
                       anchor class in ``event[anchor_key]`` (overriding any file
                       field). Use for simulation-as-anchors (one class per file)
                       or to blank a dataset's labels (force_anchor = -1).
        max_events   : int | None -- optional reproducible random subsample cap.
        seed         : int -- RNG seed for the subsample.
        hit_affine   : {field: [scale, offset]} | None -- unit/origin map applied
                       to THIS source only, before concatenating. Necessary
                       because the sources need not share a coordinate frame:
                       the 2012 filtered files give z = 2.8*k (30 and 70 GeV)
                       but the dedicated electron run gives z = 2.8*(k-1), one
                       layer off. A single global affine (``data.hit_affine``)
                       silently misplaces every hit of the odd one out.

    Per-hit field_map keys MUST be identical across sources. Per-event fields are
    unioned: a source missing a field is filled with ``-1`` (kept int/float by the
    dtype of the first source that provides it).
    """

    def __init__(self, sources: list, anchor_key: str = "anchor_label"):
        if not sources:
            raise ValueError("MultiFlatEventReader needs at least one source")

        readers, keeps = [], []
        # True when at least one source declared its own frame -> the dataset must
        # NOT apply data.hit_affine a second time on top of the concatenation.
        self.hit_affine_applied = any(s.get("hit_affine") for s in sources)
        for spec in sources:
            r = FlatEventReader(spec["path"], spec["field_map"])
            apply_hit_affine(r.hit, spec.get("hit_affine"),
                             tag=os.path.basename(spec["path"]))
            keep = np.arange(r.n_events, dtype=np.int64)
            max_events = spec.get("max_events")
            if max_events is not None and int(max_events) < r.n_events:
                rng = np.random.default_rng(int(spec.get("seed", 42)))
                keep = np.sort(rng.choice(r.n_events, int(max_events), replace=False))
            # per-source anchor override (constant class for the whole file)
            fa = spec.get("force_anchor")
            if fa is not None:
                r.event[anchor_key] = np.full(r.n_events, int(fa), dtype=np.int64)
            readers.append(r)
            keeps.append(keep)
            print(f"[MultiReader] '{os.path.basename(spec['path'])}': "
                  f"{keep.size}/{r.n_events} events"
                  + (f", anchor={int(fa)}" if fa is not None else ""))

        # ---- per-hit keys must match across sources ----
        hit_keys = list(readers[0].hit.keys())
        for r in readers[1:]:
            if set(r.hit.keys()) != set(hit_keys):
                raise ValueError(
                    "All sources must share the same per-hit field_map keys; got "
                    f"{sorted(hit_keys)} vs {sorted(r.hit.keys())}"
                )

        # ---- combined offsets ----
        sizes_per_src = [
            (r.offsets[1:] - r.offsets[:-1])[keep] for r, keep in zip(readers, keeps)
        ]
        all_sizes = np.concatenate(sizes_per_src) if sizes_per_src else np.empty(0, np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(all_sizes)]).astype(np.int64)
        self.n_events = int(all_sizes.size)
        self.n_hits = int(self.offsets[-1])

        # ---- per-hit arrays (gather selected events, then concatenate) ----
        self.hit: Dict[str, np.ndarray] = {}
        hit_idx_per_src = []
        for r, keep, sizes in zip(readers, keeps, sizes_per_src):
            if keep.size == r.n_events and np.array_equal(keep, np.arange(r.n_events)):
                hit_idx_per_src.append(None)  # full source -> slice directly
            else:
                hit_idx_per_src.append(_concat_ranges(r.offsets[keep], sizes))
        for name in hit_keys:
            parts = []
            for r, hidx in zip(readers, hit_idx_per_src):
                arr = r.hit[name]
                parts.append(arr if hidx is None else arr[hidx])
            self.hit[name] = np.concatenate(parts).astype(np.float32)

        # ---- per-event arrays (union of keys; fill missing with -1) ----
        event_keys = set()
        for r in readers:
            event_keys |= set(r.event.keys())
        self.event: Dict[str, Optional[np.ndarray]] = {}
        for key in event_keys:
            if all(r.event.get(key) is None for r in readers):
                self.event[key] = None
                continue
            ref = next(np.asarray(r.event[key]) for r in readers if r.event.get(key) is not None)
            parts = []
            for r, keep in zip(readers, keeps):
                arr = r.event.get(key)
                if arr is None:
                    parts.append(np.full(keep.size, -1, dtype=ref.dtype))
                else:
                    parts.append(np.asarray(arr)[keep].astype(ref.dtype, copy=False))
            self.event[key] = np.concatenate(parts)

        # ---- per-event source index: 0 = primary (TB), 1.. = anchor sources ----
        # Consumed by the (opt-in) domain-alignment loss; harmless otherwise.
        self.event["domain"] = np.concatenate([
            np.full(keep.size, src_i, dtype=np.int64)
            for src_i, keep in enumerate(keeps)
        ])

        self.path = sources[0]["path"]  # stats file is derived from the primary source

    def nhits_per_event(self) -> np.ndarray:
        return (self.offsets[1:] - self.offsets[:-1]).astype(np.int64)

    def hit_to_event_index(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_events, dtype=np.int64), self.nhits_per_event())
