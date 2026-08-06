"""
Fill the per-event ``anchor_label`` field of a flat HDF5 (in place).

``anchor_label``: -1 = no anchor, 0..K-1 = class. It is the only labeled signal
the clustering POC trains on (anchor CE term).

Two modes (config: ``configs/anchors.yml``):

  nhits_sampling  (automatic, reproducible)
      For each class, take the events tagged for that class (mutually-exclusive
      boolean flags such as is_electron/is_pion/is_muon, or a value map over an
      integer field), keep those whose nHits is near the class mean
      (mean +/- window_std * std), and randomly sample ``n_per_class`` of them
      with a fixed ``seed``. If the band is too small it auto-widens to the
      closest N. This is the recommended path.

  manual  (fill by hand)
      You provide explicit event-index lists per class. More work, but useful if
      you want to curate anchors yourself. See ``manual:`` in the config.

Usage:
    python -m src.convert.mark_anchors --h5 data/E70GeV_2016.h5 --config configs/anchors.yml
    python -m src.convert.mark_anchors --h5 data/E70GeV_2016.h5 --clear   # all -> -1
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np
import yaml


def _nhits(f: h5py.File, field) -> np.ndarray:
    if field and field in f:
        return np.asarray(f[field][:]).astype(np.float64)
    offsets = np.asarray(f["offsets"][:]).astype(np.int64)
    return (offsets[1:] - offsets[:-1]).astype(np.float64)


def _class_masks(f: h5py.File, cfg: dict, n_events: int) -> dict:
    """Return {class_index: boolean event mask} from flags or an integer field."""
    int_source = cfg.get("int_source")
    if int_source:
        field = int_source["field"]
        vmap = {int(k): int(v) for k, v in int_source["map"].items()}
        vals = np.asarray(f[field][:]).astype(int)
        return {ci: (vals == val) for val, ci in vmap.items()}
    masks = {}
    for ci, flag_field in (cfg.get("classes") or {}).items():
        if flag_field not in f:
            raise KeyError(f"Flag field '{flag_field}' (class {ci}) not in h5")
        masks[int(ci)] = np.asarray(f[flag_field][:]).astype(bool)
    return masks


def sample_near_mean(nhits: np.ndarray, candidates: np.ndarray, n: int,
                     window_std: float, rng: np.random.Generator) -> np.ndarray:
    """Pick n events from `candidates` near the mean nHits, reproducibly."""
    if candidates.size == 0:
        return candidates
    vals = nhits[candidates]
    mean, std = vals.mean(), max(vals.std(), 1e-6)
    band = np.abs(vals - mean) <= window_std * std
    pool = candidates[band]
    if pool.size < n:
        # auto-widen: take the n closest to the mean
        order = np.argsort(np.abs(vals - mean))
        pool = candidates[order[:max(n, pool.size)]]
    if pool.size <= n:
        return pool
    return rng.choice(pool, size=n, replace=False)


def clear(h5_path: str) -> None:
    """Reset every ``anchor_label`` to -1 (TB fully unlabeled). No config needed.

    Use when anchors come from a separate file (e.g. simulation-as-anchors, see
    configs/sim_anchors.yml): the test-beam file should carry no labels.
    """
    with h5py.File(h5_path, "r+") as f:
        n_events = len(f["offsets"]) - 1
        anchor = -np.ones(n_events, dtype=np.int64)
        if "anchor_label" in f:
            del f["anchor_label"]
        f.create_dataset("anchor_label", data=anchor, compression="lzf")
        print(f"[anchors] cleared: all {n_events} events set to anchor_label=-1")


def mark(h5_path: str, cfg: dict) -> None:
    with h5py.File(h5_path, "r+") as f:
        n_events = len(f["offsets"]) - 1
        nhits = _nhits(f, cfg.get("nhits_field"))
        anchor = -np.ones(n_events, dtype=np.int64)

        mode = cfg.get("mode", "nhits_sampling")
        if mode == "manual":
            for ci, idxs in (cfg.get("manual") or {}).items():
                idxs = np.asarray(list(idxs), dtype=np.int64)
                anchor[idxs] = int(ci)
                print(f"[anchors] class {ci}: {idxs.size} manual events")
        elif mode == "nhits_sampling":
            rng = np.random.default_rng(cfg.get("seed", 42))
            n_per = int(cfg.get("n_per_class", 50))
            wstd = float(cfg.get("window_std", 0.5))
            masks = _class_masks(f, cfg, n_events)
            for ci, mask in sorted(masks.items()):
                candidates = np.flatnonzero(mask).astype(np.int64)
                sel = sample_near_mean(nhits, candidates, n_per, wstd, rng)
                anchor[sel] = ci
                m = nhits[sel]
                print(
                    f"[anchors] class {ci}: {sel.size} anchors "
                    f"(pool={candidates.size}, nHits sel mean={m.mean():.0f} "
                    f"[{m.min():.0f}-{m.max():.0f}], class mean={nhits[candidates].mean():.0f})"
                )
        else:
            raise ValueError(f"Unknown anchor mode: {mode}")

        if "anchor_label" in f:
            del f["anchor_label"]
        f.create_dataset("anchor_label", data=anchor, compression="lzf")
        print(f"[anchors] total anchors: {int((anchor >= 0).sum())} / {n_events} events")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--config", default=None,
                    help="anchor config (required unless --clear)")
    ap.add_argument("--clear", action="store_true",
                    help="reset ALL anchor_label to -1 (fully unlabeled); ignores --config")
    ap.add_argument("--seed", type=int, default=None, help="override seed")
    ap.add_argument("--n_per_class", type=int, default=None)
    args = ap.parse_args()

    if args.clear:
        clear(args.h5)
        return

    if not args.config:
        ap.error("--config is required unless --clear is given")
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.n_per_class is not None:
        cfg["n_per_class"] = args.n_per_class
    mark(args.h5, cfg)


if __name__ == "__main__":
    main()
