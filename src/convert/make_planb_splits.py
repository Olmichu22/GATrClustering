"""Build the plan-b (energy-scalar + 30/70 GeV mix) training/eval splits.

From the two filtered, cm-frame, thr-fixed files
    E70GeV_2012_filtered.h5   (833545 events, status==1: 521329)
    E30GeV_2012_filtered.h5   (1596273 events, status==1: 1009696)
this writes DISJOINT subsets (no event appears in two roles, so there is no
anchor/eval leakage):

  per energy E in {30, 70}:
    anchor_e{E}.h5, anchor_pi{E}.h5, anchor_mu{E}.h5   -- N_ANCHOR events/class,
        one particle_type per file (used with force_anchor in the config).
    unlab_{E}.h5   -- U_UNLAB representative events (all particle_types incl. the
        unlabeled -1), labels stripped later via force_anchor=-1. Excludes anchors.
  eval_30.h5       -- E_EVAL labelled 30 GeV events, disjoint from anchors AND the
        30 GeV unlabeled train pool -> clean held-out metrics.

Both files already share ONE cm coordinate frame (z=[2.8,5.6,...]) and the
thr1-dominant convention, so mixing them introduces no domain shortcut (see the
session notes). Only status==1 events are used.

Run (inside the Apptainer image, h5py available):
  python -m src.convert.make_planb_splits
"""
from __future__ import annotations

import os
from typing import Dict, List

import h5py
import numpy as np

SRC70 = "/nfs/cms/arqolmo/SDHCAL_Energy/data/filtered/E70GeV_2012_filtered.h5"
SRC30 = "/nfs/cms/arqolmo/SDHCAL_Energy/data/filtered/E30GeV_2012_filtered.h5"
OUT_DIR = "/nfs/cms/arqolmo/SDHCAL_Energy/data/filtered/planb"

# per-class anchors per energy; unlabeled-train per energy; held-out 30 GeV eval
N_ANCHOR = 300
U_UNLAB = 100_000
E_EVAL = 200_000
SEED = 1234

HIT_KEYS = ["x", "y", "z", "i", "j", "k", "thr"]
EVENT_KEYS = ["energy", "particle_type", "filter_status", "nHits_total"]
# 0=electron, 1=pion, 2=muon (particle_type convention in the filtered files)
CLASS_NAMES = {0: "e", 1: "pi", 2: "mu"}


def _load(path: str) -> Dict[str, np.ndarray]:
    d: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        d["offsets"] = f["offsets"][:].astype(np.int64)
        for k in HIT_KEYS:
            d[k] = f[k][:]
        for k in EVENT_KEYS:
            d[k] = f[k][:] if k in f else None
    return d


def _write_subset(src: Dict[str, np.ndarray], ev_idx: np.ndarray, out_path: str,
                  anchor_label: int) -> None:
    """Write the events in ``ev_idx`` (CSR-gathered) to ``out_path``."""
    ev_idx = np.sort(np.asarray(ev_idx, dtype=np.int64))
    off = src["offsets"]
    starts, ends = off[ev_idx], off[ev_idx + 1]
    lens = ends - starts
    new_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    # hit gather indices (vectorized: no per-event Python loop)
    total = int(lens.sum())
    if total:
        idx = np.ones(total, dtype=np.int64)
        seg = np.cumsum(lens)[:-1]
        idx[0] = starts[0]
        idx[seg] = starts[1:] - ends[:-1] + 1
        hit_idx = np.cumsum(idx)
    else:
        hit_idx = np.empty(0, np.int64)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as o:
        o.create_dataset("offsets", data=new_off, compression="lzf")
        for k in HIT_KEYS:
            o.create_dataset(k, data=src[k][hit_idx], compression="lzf")
        for k in EVENT_KEYS:
            if src[k] is not None:
                o.create_dataset(k, data=src[k][ev_idx], compression="lzf")
        o.create_dataset("anchor_label",
                         data=np.full(len(ev_idx), anchor_label, np.int64),
                         compression="lzf")
    ptype = src["particle_type"][ev_idx] if src["particle_type"] is not None else None
    comp = dict(zip(*[a.tolist() for a in np.unique(ptype, return_counts=True)])) if ptype is not None else {}
    print(f"  wrote {os.path.basename(out_path):18s} {len(ev_idx):>7d} events, "
          f"{len(hit_idx):>9d} hits, anchor={anchor_label}, ptype={comp}")


def process_energy(path: str, energy: int, rng: np.random.Generator,
                   want_eval: bool) -> None:
    print(f"=== {energy} GeV: {os.path.basename(path)} ===")
    src = _load(path)
    status = src["filter_status"]
    ptype = src["particle_type"]
    good = np.flatnonzero(status == 1).astype(np.int64)
    print(f"  status==1: {good.size} events")

    used = np.zeros(len(src["offsets"]) - 1, dtype=bool)

    # ---- per-class anchors (disjoint, one class per file) ----
    for cls, name in CLASS_NAMES.items():
        pool = good[(ptype[good] == cls) & ~used[good]]
        rng.shuffle(pool)
        take = pool[:N_ANCHOR]
        used[take] = True
        _write_subset(src, take, f"{OUT_DIR}/anchor_{name}{energy}.h5", anchor_label=cls)

    # ---- held-out labelled eval (30 GeV only), disjoint from anchors ----
    if want_eval:
        pool = good[(ptype[good] >= 0) & ~used[good]]  # labelled e/pi/mu only
        rng.shuffle(pool)
        take = pool[:E_EVAL]
        used[take] = True
        _write_subset(src, take, f"{OUT_DIR}/eval_{energy}.h5", anchor_label=-1)

    # ---- representative unlabeled train (all types incl. -1), disjoint ----
    pool = good[~used[good]]
    rng.shuffle(pool)
    take = pool[:U_UNLAB]
    _write_subset(src, take, f"{OUT_DIR}/unlab_{energy}.h5", anchor_label=-1)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    process_energy(SRC70, 70, rng, want_eval=False)
    process_energy(SRC30, 30, rng, want_eval=True)
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
