"""Los DataLoader de train/evaluate/infer con num_workers > 0 no rompen.

    apptainer exec -B /nfs:/nfs <img> python tests/smoke_loader_workers.py

El riesgo real no es el valor por defecto sino el fork: el dataset lleva los
arrays de hits en memoria y el h5 ya cerrado. Se comprueba que un loader con
workers y persistent_workers itera entero, devuelve los mismos eventos que uno
sin workers, y que el proceso termina sin quedarse colgado.
"""
import os
import sys
import tempfile

import h5py
import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import DEFAULT_NUM_WORKERS, make_clustering_splits  # noqa: E402

TMP = tempfile.mkdtemp()
rng = np.random.default_rng(0)
N = 400
sizes = rng.integers(5, 40, N)
off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
nh = int(off[-1])
path = TMP + "/d.h5"
with h5py.File(path, "w") as f:
    f["offsets"] = off
    for c in ("x", "y"):
        f[c] = rng.uniform(0, 100, nh).astype(np.float32)
    f["k"] = rng.integers(1, 50, nh).astype(np.float32)
    f["z"] = (2.8 * f["k"][:]).astype(np.float32)
    f["thr"] = rng.integers(1, 4, nh).astype(np.float32)
    f["energy"] = np.full(N, 70.0, np.float32)
    f["anchor_label"] = np.where(np.arange(N) < 30, 1, -1).astype(np.int64)
    f["particle_type"] = f["anchor_label"][:]

data_cfg = dict(path=path, val_ratio=0.2, seed=42,
                field_map=dict(offsets="offsets",
                               hit=dict(x="x", y="y", z="z", k="k", thr="thr"),
                               event=dict(energy="energy", anchor_label="anchor_label",
                                          class_label="particle_type")),
                anchor_field="anchor_label", eval_label_field="class_label")
features = dict(mv_point=["x", "y", "z"], mv_scalar=["thr"], scalars=[],
                thr_encoding="ordinal", extra_event_scalars=[])
scaling = dict(source="online", stats_path=TMP + "/s.yml",
               features={"x": {"mode": "none"}, "y": {"mode": "none"},
                         "z": {"mode": "none"}, "k": {"mode": "none"},
                         "thr": {"mode": "none"}, "energy": {"mode": "none"},
                         "density": {"mode": "none"}})

_, val_ds, _ = make_clustering_splits(data_cfg, features, scaling)

ok = True


def check(name, cond, extra=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else ""))
    ok &= bool(cond)


ncpu = len(os.sched_getaffinity(0))
check("DEFAULT_NUM_WORKERS > 0", DEFAULT_NUM_WORKERS > 0, str(DEFAULT_NUM_WORKERS))
check("DEFAULT_NUM_WORKERS no supera las CPUs disponibles",
      DEFAULT_NUM_WORKERS <= ncpu, "%d workers <= %d cpus" % (DEFAULT_NUM_WORKERS, ncpu))

plain = DataLoader(val_ds, batch_size=16, shuffle=False)
ref = torch.cat([b.anchor_label for b in plain])

nw = DEFAULT_NUM_WORKERS
worked = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=nw,
                    pin_memory=True, persistent_workers=nw > 0)
got = torch.cat([b.anchor_label for b in worked])
check("mismos eventos con %d workers que sin ninguno" % nw, torch.equal(ref, got),
      "%d eventos" % got.numel())

# segunda pasada: persistent_workers reutiliza los procesos
got2 = torch.cat([b.anchor_label for b in worked])
check("segunda pasada con persistent_workers", torch.equal(ref, got2))

nhits = torch.cat([b.nhits_total.view(-1) for b in worked])
check("los tensores por evento llegan intactos", nhits.min() >= 5 and nhits.max() <= 40,
      "nHits %d..%d" % (int(nhits.min()), int(nhits.max())))

del worked  # cierra los workers persistentes antes de salir
print("\n" + ("TODO OK" if ok else "HAY FALLOS"))
sys.exit(0 if ok else 1)
