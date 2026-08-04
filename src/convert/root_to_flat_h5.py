"""
ROOT -> flat (CSR) HDF5 converter for the clustering POC.

Environment (uproot is provided there):
    source /cvmfs/sw.hsf.org/key4hep/setup.sh

The set of exported fields is fully selectable through the config
(``configs/export_root.yml``): ``hit`` and ``event`` map a LOGICAL name (the key
written to the h5) to a ROOT branch name. Only listed branches are read, so
switching detector/format or dropping/adding a field is a config edit.

Output layout (matches the fixed CSR contract read by ``FlatEventReader``):
    offsets            int64  (n_events + 1)
    <logical hit>...   flat per-hit arrays (len n_hits), configurable dtype
    <logical event>... per-event arrays (len n_events)
    anchor_label       int64  (n_events)  -1 by default (user marks anchors)

Multiple input ROOT files are concatenated with continuous offsets.

Usage:
    source /cvmfs/sw.hsf.org/key4hep/setup.sh
    python -m src.convert.root_to_flat_h5 \
        --config configs/export_root.yml \
        --inputs /nfs/cms/arqolmo/SDHCALTBData/merged_E70GeVJun_filtered_2016.root \
        --output data/E70GeV_2016.h5
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import h5py
import numpy as np
import uproot
import yaml


def normalize_thresholds(thr: np.ndarray) -> np.ndarray:
    """Map offset-encoded thresholds (5/6/7) back to 1/2/3, PER HIT.

    Element-wise and encoding-agnostic (a hit already in {1,2,3} is left
    untouched); mirrors the authoritative reader in the SDHCAL_Energy repo.
    """
    thr = np.asarray(thr)
    if thr.size == 0:
        return thr.astype(np.int8, copy=False)
    out = thr.copy()
    mask = thr > 4
    out[mask] = thr[mask] - 4
    return out.astype(np.int8, copy=False)


def _lengths(obj_arr: np.ndarray) -> np.ndarray:
    """Per-event hit counts from a jagged (object) array of per-event arrays."""
    return np.fromiter((len(v) for v in obj_arr), dtype=np.int64, count=len(obj_arr))


def _flatten(obj_arr: np.ndarray, dtype) -> np.ndarray:
    if len(obj_arr) == 0:
        return np.array([], dtype=dtype)
    return np.concatenate([np.asarray(v) for v in obj_arr]).astype(dtype, copy=False)


def read_one_file(path: str, cfg: dict) -> Dict[str, object]:
    """Read one ROOT file into per-hit jagged arrays + per-event scalar arrays."""
    tree_name = cfg["tree"]
    hit_map: Dict[str, str] = cfg["hit"]
    event_map: Dict[str, str] = cfg.get("event", {}) or {}

    with uproot.open(path) as f:
        tree = f[tree_name]
        available = set(tree.keys())
        for logical, branch in {**hit_map, **event_map}.items():
            if branch not in available:
                raise KeyError(
                    f"Branch '{branch}' (logical '{logical}') not in tree "
                    f"'{tree_name}' of {path}"
                )
        hit_jagged = {lg: tree[br].array(library="np") for lg, br in hit_map.items()}
        event_arrays = {lg: np.asarray(tree[br].array(library="np")) for lg, br in event_map.items()}

    # per-event hit counts from a reference hit branch; verify consistency
    ref = next(iter(hit_jagged))
    counts = _lengths(hit_jagged[ref])
    for lg, arr in hit_jagged.items():
        c = _lengths(arr)
        if not np.array_equal(c, counts):
            raise ValueError(f"Per-hit branch '{lg}' has inconsistent per-event lengths in {path}")

    return {"hit_jagged": hit_jagged, "event_arrays": event_arrays, "counts": counts}


def build_anchor_label(cfg: dict, event_arrays: Dict[str, np.ndarray], n_events: int) -> np.ndarray:
    anchor_cfg = cfg.get("anchor", {}) or {}
    mode = anchor_cfg.get("mode", "none")
    if mode == "none":
        return -np.ones(n_events, dtype=np.int64)
    if mode == "from_branch":
        branch = anchor_cfg.get("branch")
        if branch is None or branch not in event_arrays:
            raise KeyError(f"anchor.mode=from_branch requires an exported event field '{branch}'")
        return np.asarray(event_arrays[branch]).astype(np.int64)
    raise ValueError(f"Unknown anchor.mode: {mode}")


def convert(inputs: List[str], output: str, cfg: dict) -> None:
    hit_dtypes = {k: np.dtype(v) for k, v in (cfg.get("hit_dtypes", {}) or {}).items()}
    thr_logical = cfg.get("thr_logical", "thr")
    do_norm = cfg.get("normalize_thr", True)

    hit_flat: Dict[str, list] = {lg: [] for lg in cfg["hit"]}
    event_cat: Dict[str, list] = {lg: [] for lg in (cfg.get("event", {}) or {})}
    all_counts: list = []

    for path in inputs:
        print(f"[convert] reading {path}")
        d = read_one_file(path, cfg)
        all_counts.append(d["counts"])
        for lg, arr in d["hit_jagged"].items():
            dt = hit_dtypes.get(lg, np.float32)
            flat = _flatten(arr, dt)
            if lg == thr_logical and do_norm:
                flat = normalize_thresholds(flat)
            hit_flat[lg].append(flat)
        for lg, arr in d["event_arrays"].items():
            event_cat[lg].append(np.asarray(arr))

    counts = np.concatenate(all_counts).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    n_events = len(counts)
    n_hits = int(offsets[-1])
    print(f"[convert] total: {n_events} events, {n_hits} hits")

    event_arrays = {lg: np.concatenate(v) for lg, v in event_cat.items()}
    anchor_label = build_anchor_label(cfg, event_arrays, n_events)

    import os

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with h5py.File(output, "w") as fout:
        fout.create_dataset("offsets", data=offsets, compression="lzf")
        for lg, parts in hit_flat.items():
            data = np.concatenate(parts) if parts else np.array([], dtype=hit_dtypes.get(lg, np.float32))
            if data.shape[0] != n_hits:
                raise ValueError(f"hit '{lg}' length {data.shape[0]} != n_hits {n_hits}")
            fout.create_dataset(lg, data=data, compression="lzf")
        for lg, arr in event_arrays.items():
            fout.create_dataset(lg, data=arr, compression="lzf")
        fout.create_dataset("anchor_label", data=anchor_label, compression="lzf")
    print(f"[convert] wrote {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--inputs", nargs="+", required=True, help="one or more ROOT files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tree", default=None, help="override tree name")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.tree:
        cfg["tree"] = args.tree
    convert(args.inputs, args.output, cfg)


if __name__ == "__main__":
    main()
