"""
PyG dataset for the clustering POC, built on ``FlatEventReader`` and
``FeatureScaler``. Generalizes ``FlatSDHCALDataset`` from GATrAutoencoder:

    - reads logical names via field_map (remappable branches),
    - per-feature scaling (not a single global norm_type),
    - optional per-event fields (energy / class_label may be missing),
    - exposes anchor_label (for CE) and class_label (eval only) per event,
    - nHits_total per event (cluster diagnostic), derived from offsets.

``make_clustering_splits`` computes stats ONLY over the train split and applies
scaling once, mirroring ``make_pf_splits`` in the original.

Anchor classes can be left out without re-exporting the data::

    data:
      ignore_anchor_labels: [3]          # e.g. multi-particle -> back to -1
      drop_ignored_anchor_events: false  # true -> remove the events altogether
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from .flat_h5_reader import FlatEventReader, MultiFlatEventReader, apply_hit_affine
from .scaling import FeatureScaler

#: Target DataLoader workers for every consumer (train / evaluate / infer).
#: 8 is what every config under configs/ already sets explicitly, so the default
#: no longer diverges from practice; evaluate_clustering / infer_eval used to
#: pass nothing at all and loaded in the main process, with the GPU waiting on a
#: single thread collating variable-length events.
_TARGET_NUM_WORKERS = 8


def available_cpus() -> int:
    """CPUs this process may actually run on (respects the cgroup/affinity)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


#: The default is CAPPED by the cores this process may actually run on. Note the
#: cap reads the AFFINITY MASK, not `nproc`: measured on gaew0120 inside a job of
#: condor/manual_s15.sub, `nproc` reports 1 while `Cpus_allowed_list` is 0-255 and
#: `cpu.max` is `max 100000` (no quota). `nproc` is not lying about the affinity
#: -- it honors OMP_NUM_THREADS, which HTCondor exports as `request_cpus`
#: (default 1). So the CPU limit here is SOFT: the workers, being separate
#: processes, do get real cores. What OMP_NUM_THREADS does throttle is the
#: intra-op thread pool of torch/BLAS inside each process, and the fix for that
#: is exporting OMP_NUM_THREADS in the job environment, not `request_cpus`.
#: An explicit `train.num_workers` in the config still wins over this default.
DEFAULT_NUM_WORKERS = min(_TARGET_NUM_WORKERS, available_cpus())


def _apply_filters(event: Dict[str, Optional[np.ndarray]], n_events: int, filters: dict) -> np.ndarray:
    """Per-event boolean mask. filters: {logical_field: value | ["<=X", ">=Y"]}."""
    mask = np.ones(n_events, dtype=bool)
    for field, cond in (filters or {}).items():
        arr = event.get(field)
        if arr is None:
            raise KeyError(f"Filter on missing field '{field}'")
        arr = np.asarray(arr)
        if isinstance(cond, (list, tuple)):
            for c in cond:
                if c[:2] in ("<=", ">="):
                    op, val = c[:2], float(c[2:])
                    mask &= (arr <= val) if op == "<=" else (arr >= val)
                elif c[0] == "<":
                    mask &= arr < float(c[1:])
                elif c[0] == ">":
                    mask &= arr > float(c[1:])
        else:
            mask &= arr == cond
    return mask


class ClusteringSDHCALDataset(Dataset):
    def __init__(
        self,
        path: str,
        field_map: dict,
        features_cfg: dict,
        anchor_field: str = "anchor_label",
        eval_label_field: str = "class_label",
        filters: Optional[dict] = None,
        reader=None,
        min_hits: int = 1,
        ignore_anchor_labels: Optional[Iterable[int]] = None,
        remap_anchor_labels: Optional[Dict[int, int]] = None,
        drop_ignored_anchor_events: bool = False,
        hit_affine: Optional[Dict[str, Iterable[float]]] = None,
    ):
        super().__init__()
        # ``reader`` may be a prebuilt (possibly multi-file) reader; otherwise a
        # single-file FlatEventReader is built from ``path``/``field_map``.
        self.reader = reader if reader is not None else FlatEventReader(path, field_map)
        self.features_cfg = features_cfg
        self.anchor_field = anchor_field
        self.eval_label_field = eval_label_field
        self._path = getattr(self.reader, "path", path)

        self.offsets = self.reader.offsets
        n_events_total = self.reader.n_events

        # per-hit arrays (numpy float32); scaling is applied in-place later
        self.hit: Dict[str, np.ndarray] = dict(self.reader.hit)

        # ---- unit/origin harmonization (``data.hit_affine``) -----------------
        # Not every producer writes the same units: the filtered 2012 sample
        # stores x/y/z in cm with a different origin and a 1-based layer index,
        # while the training file is in mm with a 0-based one. Feeding the former
        # to a model trained on the latter would shrink every coordinate ~10x.
        # ``hit_affine: {field: [scale, offset]}`` maps field -> scale*field +
        # offset right after reading, BEFORE the density is derived and before
        # any per-feature scaling, so the whole pipeline sees one convention.
        # Declared in the config on purpose: a silent geometry change is exactly
        # the kind of thing that must be visible in the run's record.
        # NOTE: this is the SINGLE-FRAME path (one affine for everything). When
        # the sources do not share a frame, ``build_event_reader`` resolves an
        # affine PER FILE and the reader has already applied it before the
        # concatenation; in that case it passes hit_affine=None down here.
        apply_hit_affine(self.hit, hit_affine, tag="all sources")

        # thr one-hot (before scaling), if thr is available
        if "thr" in self.hit:
            thr = self.hit["thr"]
            self.hit["thr1"] = (thr == 1).astype(np.float32)
            self.hit["thr2"] = (thr == 2).astype(np.float32)
            self.hit["thr3"] = (thr == 3).astype(np.float32)

        # per-event arrays
        self.event = dict(self.reader.event)

        # ---- raw beam energy, kept UNSCALED --------------------------------
        # ``event['energy']`` is overwritten in place by apply_scaling_inplace
        # (log / log_z), so it can no longer answer "which energy is this event?"
        # once training starts. In a mixed 30+70 GeV run that question is needed
        # in two places that must NOT depend on the scaling choice: the
        # stratified split and the per-(class, energy) held-out accuracy. Keep a
        # copy in GeV; exposed per event as ``data.energy_gev``.
        e_raw = self.event.get("energy")
        self._energy_raw = (
            np.asarray(e_raw, dtype=np.float32) if e_raw is not None
            else np.zeros(n_events_total, dtype=np.float32)
        )
        # total nHits: derived from offsets (always correct after filtering)
        self._nhits_full = self.reader.nhits_per_event().astype(np.float32)

        # ---- derived per-event density: hits per ACTIVE layer ----------------
        # nHits alone grows with how DEEP a shower goes, i.e. with the number of
        # layers it crosses; dividing by the distinct layers actually hit gives
        # transverse compactness, which separates a dense EM shower (~23) from a
        # MIP (~1.9) without being tied to the detector depth. Derived here (not
        # read from the file) so it exists for any dataset, and registered as a
        # normal event feature so ``scaling`` handles it like energy: stats
        # computed ONLINE over the train split only.
        layer_key = (field_map.get("hit") or {}).get("k")
        if layer_key is not None and "k" in self.hit:
            self.event["density"] = self._compute_density()

        # ---- anchor classes to leave out of CE -------------------------------
        # A curated anchor file may carry classes the current head cannot take
        # (e.g. multi-particle = 3 while head.num_clusters = 3) or classes you
        # simply do not want to commit to yet. Demoting them to -1 here, ONCE,
        # is what makes them invisible everywhere downstream: CE, prototype
        # init, held-out anchor accuracy and prior_exclude_anchors all key off
        # ``anchor_label >= 0``. The events themselves stay in the unlabeled
        # pool (swap/VICReg still see them) unless drop_ignored_anchor_events.
        self._ignored_anchor_labels = sorted({int(v) for v in (ignore_anchor_labels or [])})
        ignored_mask = np.zeros(n_events_total, dtype=bool)
        anchor_arr = self.event.get(self.anchor_field)
        if anchor_arr is not None:
            anchor_arr = np.asarray(anchor_arr)
            if self._ignored_anchor_labels:
                ignored_mask = np.isin(anchor_arr, self._ignored_anchor_labels)
                if ignored_mask.any():
                    # copy: the reader's array may be shared with other views
                    anchor_arr = anchor_arr.copy()
                    anchor_arr[ignored_mask] = -1
                    self.event[self.anchor_field] = anchor_arr
                    what = "Dropping" if drop_ignored_anchor_events else "Un-labeling"
                    print(f"[Dataset] {what} {int(ignored_mask.sum())} anchors with "
                          f"label in {self._ignored_anchor_labels}")
        # ---- anchor label -> cluster index ------------------------------------
        # The labeler's class indices are a LABELING convention (0 electron,
        # 1 pion, 2 muon, 3 multip, 4 penetrante, 5 ruido) and need not be the
        # contiguous 0..K-1 the CE head requires. ``remap_anchor_labels: {5: 3}``
        # says "ruido is cluster 3 in this run". Applied AFTER ignore, so
        # ignore_anchor_labels speaks the labeler convention and a remap target
        # may reuse an ignored class's index (e.g. ignore multip=3, ruido 5->3).
        remap = {int(k): int(v) for k, v in (remap_anchor_labels or {}).items()}
        if remap:
            arr = self.event.get(self.anchor_field)
            if arr is not None:
                arr = np.asarray(arr).copy()
                src_mask = np.isin(arr, list(remap))
                if src_mask.any():
                    arr[src_mask] = np.vectorize(remap.get)(arr[src_mask])
                    self.event[self.anchor_field] = arr
                print(f"[Dataset] remapped {int(src_mask.sum())} anchor labels: {remap}")

        anchor_arr = self.event.get(self.anchor_field)
        if anchor_arr is not None:
            anchor_arr = np.asarray(anchor_arr)
            lab, cnt = np.unique(anchor_arr[anchor_arr >= 0], return_counts=True)
            print(f"[Dataset] anchor labels in use: {dict(zip(lab.tolist(), cnt.tolist()))}")

        # filters + min-hits guard (empty / near-empty events make GATr/attention
        # produce NaN, which then poisons every loss). Drop events with < min_hits.
        mask = np.ones(n_events_total, dtype=bool)
        if filters:
            mask &= _apply_filters(self.event, n_events_total, filters)
        if drop_ignored_anchor_events and ignored_mask.any():
            mask &= ~ignored_mask
        if min_hits and min_hits > 0:
            too_few = self._nhits_full < min_hits
            if too_few.any():
                print(f"[Dataset] Dropping {int(too_few.sum())} events with < {min_hits} hits")
            mask &= ~too_few
        self._event_indices = np.flatnonzero(mask).astype(np.int64)
        if self._event_indices.size != n_events_total:
            print(f"[Dataset] Kept {self._event_indices.size}/{n_events_total} events")
        self._n_events = int(self._event_indices.size)

    def _compute_density(self) -> np.ndarray:
        """hits / distinct active layers, per event (float32, >= 1)."""
        off = np.asarray(self.offsets, dtype=np.int64)
        k = np.asarray(self.hit["k"])
        n = off.size - 1
        dens = np.ones(n, dtype=np.float32)
        for i in range(n):
            s, e = int(off[i]), int(off[i + 1])
            if e <= s:
                continue
            nl = np.unique(k[s:e]).size
            dens[i] = (e - s) / max(nl, 1)
        return dens

    # ---- Dataset interface ----
    def len(self) -> int:
        return self._n_events

    def _event_scalar(self, name: str, real_idx: int, default: float) -> torch.Tensor:
        arr = self.event.get(name)
        val = float(arr[real_idx]) if arr is not None else default
        return torch.tensor([val], dtype=torch.float32)

    def get(self, idx: int) -> Data:
        real_idx = int(self._event_indices[idx])
        s, e = int(self.offsets[real_idx]), int(self.offsets[real_idx + 1])

        data = Data()
        # positions from mv_point (exactly 3)
        pt = self.features_cfg["mv_point"]
        assert len(pt) == 3, "features.mv_point must have exactly 3 components"
        pos = np.stack([self.hit[c][s:e] for c in pt], axis=1)
        data.pos = torch.from_numpy(pos.astype(np.float32))

        # all available per-hit features, by name (L,1)
        for name, arr in self.hit.items():
            data[name] = torch.from_numpy(arr[s:e].astype(np.float32)).unsqueeze(1)

        # per-event
        anchor = self.event.get(self.anchor_field)
        a_val = int(anchor[real_idx]) if anchor is not None else -1
        data.anchor_label = torch.tensor([a_val], dtype=torch.long)

        cls = self.event.get(self.eval_label_field)
        c_val = int(cls[real_idx]) if cls is not None else -1
        data.class_label = torch.tensor([c_val], dtype=torch.long)

        # source index (0 = primary/TB, 1.. = anchor files); single-file readers
        # have no 'domain' field -> everything is the primary source (0).
        dom = self.event.get("domain")
        d_val = int(dom[real_idx]) if dom is not None else 0
        data.domain = torch.tensor([d_val], dtype=torch.long)

        data.energy = self._event_scalar("energy", real_idx, 0.0)
        # Unscaled beam energy (GeV): grouping key for the per-(class, energy)
        # metrics. Never a model input — `data.energy` is the scaled one.
        data.energy_gev = torch.tensor(
            [float(self._energy_raw[real_idx])], dtype=torch.float32
        )
        # scaled density (see _compute_density); attention_density pooling reads
        # it instead of recomputing a raw, unnormalized ratio.
        data.density = self._event_scalar("density", real_idx, 0.0)
        data.nhits_total = torch.tensor([float(self._nhits_full[real_idx])], dtype=torch.float32)
        return data

    # ---- per-feature stats over a subset of events (train) ----
    def train_feature_values(self, train_idx: np.ndarray) -> Dict[str, np.ndarray]:
        """Values of each feature (hit and event) restricted to the train split."""
        ev_global = self._event_indices[train_idx]
        # hit->event mask
        sizes = self.offsets[1:] - self.offsets[:-1]
        hit_events = np.repeat(np.arange(len(self.offsets) - 1, dtype=np.int64), sizes)
        in_train = np.zeros(len(self.offsets) - 1, dtype=bool)
        in_train[ev_global] = True
        hit_mask = in_train[hit_events]

        out: Dict[str, np.ndarray] = {}
        for name, arr in self.hit.items():
            out[name] = arr[hit_mask]
        for name in ("energy", "density"):
            if self.event.get(name) is not None:
                out[name] = np.asarray(self.event[name])[ev_global].astype(np.float64)
        return out

    def apply_scaling_inplace(self, scaler: FeatureScaler) -> None:
        for name in list(self.hit.keys()):
            self.hit[name] = scaler.apply(name, self.hit[name])
        for name in ("energy", "density"):
            if self.event.get(name) is not None:
                self.event[name] = scaler.apply(name, np.asarray(self.event[name]))


def build_event_reader(data_cfg: dict):
    """Build the (single- or multi-file) reader backing the dataset.

    Multi-file mode is triggered by ``data.anchor_datasets`` (a list of extra
    files used purely as anchors, e.g. simulation with one particle class per
    file) or by ``data.force_anchor`` (override/blank the primary file's labels).
    In those cases the primary file (``data.path``) plus every anchor file are
    concatenated into one :class:`MultiFlatEventReader`; the primary keeps its own
    ``anchor_label`` field unless ``data.force_anchor`` is set (e.g. -1 to treat
    test-beam data as fully unlabeled).

    Without those keys it falls back to a plain single-file reader (unchanged
    behavior, backward compatible).
    """
    field_map = data_cfg["field_map"]
    anchor_field = data_cfg.get("anchor_field", "anchor_label")
    anchor_datasets = data_cfg.get("anchor_datasets") or []
    force_anchor = data_cfg.get("force_anchor")  # None -> keep primary's own field
    seed = int(data_cfg.get("seed", 42))

    if not anchor_datasets and force_anchor is None:
        return FlatEventReader(data_cfg["path"], field_map)

    # ---- per-file frames -------------------------------------------------
    # ``data.hit_affine`` is the DEFAULT for every source; a source may override
    # it with its own ``hit_affine``. The moment any source does, the affine is
    # resolved and applied per file inside the reader, and the (single-frame)
    # path in the dataset is skipped -- see ``_dataset_hit_affine`` below.
    global_affine = data_cfg.get("hit_affine")
    per_file = any(ds_cfg.get("hit_affine") for ds_cfg in anchor_datasets)

    sources = [{
        "path": data_cfg["path"],
        "field_map": field_map,
        "force_anchor": None if force_anchor is None else int(force_anchor),
        "max_events": data_cfg.get("max_events"),
        "seed": seed,
        "hit_affine": global_affine if per_file else None,
    }]
    for i, ds_cfg in enumerate(anchor_datasets):
        if "anchor_label" not in ds_cfg:
            raise KeyError(f"data.anchor_datasets[{i}] must set 'anchor_label' "
                           "(use null to keep the file's own anchor_label)")
        # anchor_label: null -> do NOT force a class; the file keeps its own
        # anchor_label field. That is what makes a SECOND fully-labelled primary
        # possible (e.g. the 30 GeV sample joining the 70 GeV one, each carrying
        # its own manual anchors) instead of only one-class-per-file sources.
        label = ds_cfg["anchor_label"]
        sources.append({
            "path": ds_cfg["path"],
            "field_map": ds_cfg.get("field_map") or field_map,
            "force_anchor": None if label is None else int(label),
            "max_events": ds_cfg.get("max_events"),
            "seed": int(ds_cfg.get("seed", seed)),
            "hit_affine": (ds_cfg.get("hit_affine") or global_affine) if per_file else None,
        })
    return MultiFlatEventReader(sources, anchor_key=anchor_field)


def _dataset_hit_affine(data_cfg: dict, reader) -> Optional[dict]:
    """The affine the DATASET still has to apply (None if the reader did it)."""
    if getattr(reader, "hit_affine_applied", False):
        return None
    return data_cfg.get("hit_affine")


def stratified_split(
    keys: np.ndarray,
    val_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices so that EVERY key group keeps the same val fraction.

    ``keys`` is one hashable key per event (here: anchor class x beam energy).
    With a plain global permutation, a class with ~90 anchors split across two
    energies can easily land with all of its held-out anchors at a single energy
    — and then the aggregate accuracy says nothing about generalizing across
    energies, which is the whole point of a mixed run. Groups smaller than
    ``1/val_ratio`` contribute at least one val event as long as they have >= 2.
    """
    val_parts, train_parts = [], []
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        rng.shuffle(idx)
        n_val = int(round(idx.size * val_ratio))
        if idx.size >= 2:
            n_val = min(max(n_val, 1), idx.size - 1)  # never empty, never all
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    val_idx = np.concatenate(val_parts) if val_parts else np.empty(0, np.int64)
    train_idx = np.concatenate(train_parts) if train_parts else np.empty(0, np.int64)
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def make_clustering_splits(
    data_cfg: dict,
    features_cfg: dict,
    scaling_cfg: dict,
) -> Tuple[Dataset, Dataset, ClusteringSDHCALDataset]:
    """Return (train_subset, val_subset, base_dataset) with scaling applied."""
    reader = build_event_reader(data_cfg)
    ds = ClusteringSDHCALDataset(
        path=data_cfg["path"],
        field_map=data_cfg["field_map"],
        features_cfg=features_cfg,
        anchor_field=data_cfg.get("anchor_field", "anchor_label"),
        eval_label_field=data_cfg.get("eval_label_field", "class_label"),
        filters=data_cfg.get("filters"),
        reader=reader,
        min_hits=int(data_cfg.get("min_hits", 1)),
        ignore_anchor_labels=data_cfg.get("ignore_anchor_labels"),
        remap_anchor_labels=data_cfg.get("remap_anchor_labels"),
        drop_ignored_anchor_events=bool(data_cfg.get("drop_ignored_anchor_events", False)),
        hit_affine=_dataset_hit_affine(data_cfg, reader),
    )

    N = ds.len()
    rng = np.random.default_rng(data_cfg.get("seed", 42))
    val_ratio = float(data_cfg.get("val_ratio", 0.2))

    # ---- validation split -------------------------------------------------
    # ``data.stratify_split: [anchor, energy]`` (any subset; null = old global
    # permutation). Stratifying by anchor class ALONE is not enough in a mixed
    # 30+70 GeV run: see stratified_split.
    strat = data_cfg.get("stratify_split")
    if strat:
        strat = [str(s) for s in strat]
        unknown = set(strat) - {"anchor", "energy"}
        if unknown:
            raise ValueError(f"data.stratify_split: unknown keys {sorted(unknown)}; "
                             "valid: 'anchor', 'energy'")
        parts = []
        if "anchor" in strat:
            anchor_arr = ds.event.get(ds.anchor_field)
            parts.append(
                np.asarray(anchor_arr)[ds._event_indices].astype(np.int64)
                if anchor_arr is not None
                else np.full(N, -1, dtype=np.int64)
            )
        if "energy" in strat:
            # Bin to the nearest GeV: the beam energy is discrete (30 / 70), and
            # rounding keeps a float column from exploding into N singleton keys.
            parts.append(np.rint(ds._energy_raw[ds._event_indices]).astype(np.int64))
        # Vectorized key encoding: np.unique(...,return_inverse) per column, then
        # mixed-radix combination. A Python-level hash(tuple(row)) would loop over
        # every event (millions) for no benefit.
        keys = np.zeros(N, dtype=np.int64)
        for col in parts:
            codes = np.unique(col, return_inverse=True)[1].astype(np.int64)
            keys = keys * (int(codes.max()) + 1 if codes.size else 1) + codes
        train_idx, val_idx = stratified_split(keys, val_ratio, rng)

        # Audit: anchors per (class, energy) on each side. This table is the
        # evidence that the held-out set can answer the cross-energy question;
        # it belongs in the run log, not in a notebook afterwards.
        anchor_all = ds.event.get(ds.anchor_field)
        if anchor_all is not None:
            a = np.asarray(anchor_all)[ds._event_indices]
            e = np.rint(ds._energy_raw[ds._event_indices]).astype(np.int64)
            is_val = np.zeros(N, dtype=bool)
            is_val[val_idx] = True
            print("[Split] held-out anchors per (class, energy):")
            for cls in sorted(set(a[a >= 0].tolist())):
                for en in sorted(set(e[a == cls].tolist())):
                    m = (a == cls) & (e == en)
                    n_val = int((m & is_val).sum())
                    print(f"          class {cls}  {en:>3d} GeV : "
                          f"{n_val:>5d} val / {int(m.sum()) - n_val:>6d} train")
    else:
        perm = rng.permutation(N)
        val_size = int(N * val_ratio)
        val_idx, train_idx = perm[:val_size], perm[val_size:]

    scaler = FeatureScaler(scaling_cfg, ds._path)
    if scaler.source == "online" and scaler.needs_stats():
        scaler.resolve_stats(ds.train_feature_values(train_idx))
    else:
        scaler.resolve_stats(None)
    ds.apply_scaling_inplace(scaler)

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    return train_ds, val_ds, ds
