"""Adapt a parent-pipeline flat_with_filters.h5 (raw pad units, thr1<->thr2
SWAPPED, k 1-based) into the GATrClustering training-file schema (mm coords,
thr1-dominant, k 0-based), so a model trained on E70GeV_2012.h5 sees identical
inputs.

Transforms (derived by matching E70GeV_2012.h5 exactly):
  x = 10.408 * (i - 0.5)          # i in 1..96  -> 5.204 .. 994.4 mm
  y = 10.408 * (j - 0.5)
  z = 226.5 + 28.0 * (k_raw - 1)  # k_raw in 1..48 -> 226.5 .. 1542.5 mm
  k = k_raw - 1                    # 0-based layer, matches primary (0..47)
  thr: swap 1<->2, keep 3         # raw beam files are thr2-dominant (wrong);
                                   # primary is thr1-dominant {1:.79,2:.17,3:.04}

Per-event fields written: energy, particle_type, nHits_total, run, filter_status,
anchor_label(=-1), is_electron/is_pion/is_muon (from particle_type). offsets kept.

Usage (key4hep):
  python -m src.convert.adapt_parent_to_clustering IN.h5 --out OUT.h5 [--status-only]
"""
from __future__ import annotations
import argparse
import h5py
import numpy as np

PITCH = 10.408      # mm, transverse pad pitch (primary x step)
X0 = 0.5            # x = PITCH*(i - X0); i=1 -> 5.204
Z0 = 226.5         # mm, first layer z in primary
DZ = 28.0          # mm, layer spacing


def adapt(in_path: str, out_path: str, status_only: bool = False) -> None:
    with h5py.File(in_path, "r") as f:
        keys = list(f.keys())
        print("input keys:", keys)
        offsets = f["offsets"][:].astype(np.int64)
        i = f["i"][:].astype(np.int32)
        j = f["j"][:].astype(np.int32)
        k_raw = f["k"][:].astype(np.int32)
        thr_raw = f["thr"][:]
        n_events = len(offsets) - 1

        def ev(name, default=None):
            if name in f:
                return f[name][:]
            if default is not None:
                return np.full(n_events, default)
            return None

        energy = ev("energy")
        ptype = ev("particle_type")
        nhits_total = ev("nHits_total")
        run = f["run"][:] if "run" in f else (f["runNr"][:] if "runNr" in f else np.zeros(n_events, np.int32))
        fstatus = ev("filter_status", 1)

    # --- hit transforms ---
    x = (PITCH * (i.astype(np.float32) - X0)).astype(np.float32)
    y = (PITCH * (j.astype(np.float32) - X0)).astype(np.float32)
    # raw layer K is 1..48; a few stray hits carry K=0 -> clamp to layer 0 so
    # k stays in the primary's 0..47 range (z >= 226.5, no negative layers).
    k0 = np.clip(k_raw - 1, 0, 47).astype(np.int32)
    z = (Z0 + DZ * k0.astype(np.float32)).astype(np.float32)
    k = k0
    thr = thr_raw.astype(np.int8).copy()
    swap = thr_raw.astype(np.int8)
    thr[swap == 1] = 2
    thr[swap == 2] = 1
    # thr==3 unchanged

    # --- optional selection ---
    if status_only:
        keep = fstatus.astype(bool)
        # rebuild CSR for kept events
        lens = np.diff(offsets)
        new_lens = lens[keep]
        new_off = np.concatenate([[0], np.cumsum(new_lens)]).astype(np.int64)
        # hit mask
        hit_keep = np.zeros(offsets[-1], dtype=bool)
        for e in np.nonzero(keep)[0]:
            hit_keep[offsets[e]:offsets[e + 1]] = True
        x, y, z, k, thr = x[hit_keep], y[hit_keep], z[hit_keep], k[hit_keep], thr[hit_keep]
        i, j = i[hit_keep], j[hit_keep]
        offsets = new_off
        energy = energy[keep] if energy is not None else None
        ptype = ptype[keep] if ptype is not None else None
        nhits_total = nhits_total[keep] if nhits_total is not None else None
        run = run[keep]
        fstatus = fstatus[keep]
        n_events = int(keep.sum())
        print(f"status_only: kept {n_events} events")

    ptype = ptype.astype(np.int8) if ptype is not None else np.full(n_events, -1, np.int8)
    anchor_label = np.full(n_events, -1, np.int64)
    is_e = (ptype == 0)
    is_pi = (ptype == 1)
    is_mu = (ptype == 2)

    with h5py.File(out_path, "w") as o:
        o.create_dataset("offsets", data=offsets)
        o.create_dataset("x", data=x)
        o.create_dataset("y", data=y)
        o.create_dataset("z", data=z)
        o.create_dataset("i", data=i)
        o.create_dataset("j", data=j)
        o.create_dataset("k", data=k)
        o.create_dataset("thr", data=thr)
        if energy is not None:
            o.create_dataset("energy", data=np.asarray(energy, dtype=np.float64))
        o.create_dataset("particle_type", data=ptype)
        if nhits_total is not None:
            o.create_dataset("nHits_total", data=np.asarray(nhits_total, dtype=np.int64))
        o.create_dataset("run", data=np.asarray(run))
        o.create_dataset("filter_status", data=np.asarray(fstatus, dtype=np.int8))
        o.create_dataset("anchor_label", data=anchor_label)
        o.create_dataset("is_electron", data=is_e)
        o.create_dataset("is_pion", data=is_pi)
        o.create_dataset("is_muon", data=is_mu)

    # --- report ---
    u, c = np.unique(thr, return_counts=True)
    fr = {int(a): round(float(b) / c.sum(), 3) for a, b in zip(u, c)}
    up, cp = np.unique(ptype, return_counts=True)
    print(f"wrote {out_path}: {n_events} events, {len(x)} hits")
    print(f"  x range [{x.min():.3g},{x.max():.3g}]  z range [{z.min():.3g},{z.max():.3g}]  k [{k.min()},{k.max()}]")
    print(f"  thr fractions (should be thr1-dominant): {fr}")
    print(f"  particle_type: {dict(zip(up.tolist(), cp.tolist()))}")
    print(f"  status==1: {int(np.asarray(fstatus).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--status-only", action="store_true",
                    help="keep only filter_status==1 events")
    a = ap.parse_args()
    adapt(a.input, a.out, a.status_only)
