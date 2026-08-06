"""Smoke test de los tres cambios de s15 (sin GPU y sin GATr).

    apptainer exec -B /nfs:/nfs <img> python tests/smoke_s15_data.py

Fabrica tres ficheros con los MISMOS marcos que los reales:
   prim70 : z = 2.8*k, k 1..50     (primario)
   prim30 : z = 2.8*k, k 0..48     (segundo primario, anchor_label propio)
   elec70 : z = 2.8*(k-1), k 1..48 (una sola clase, marco desfasado una capa)
y comprueba que tras el reader los tres caen en el MISMO marco de entrenamiento
(z_mm = 28*k' + 226.5), que el 30 GeV conserva sus etiquetas, que el split
estratifica por (clase, energia), que energy_gev sobrevive sin escalar y que el
camino antiguo (afin global + split aleatorio) sigue comportandose igual.
"""
import collections
import os
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import make_clustering_splits  # noqa: E402

TMP = tempfile.mkdtemp()


def make(path, n_ev, kmin, kmax, z_of_k, energy, labels):
    rng = np.random.default_rng(0)
    sizes = rng.integers(5, 15, n_ev)
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    nh = int(off[-1])
    k = rng.integers(kmin, kmax + 1, nh).astype(np.float32)
    with h5py.File(path, "w") as f:
        f["offsets"] = off
        f["x"] = rng.uniform(0.5, 100.0, nh).astype(np.float32)
        f["y"] = rng.uniform(0.5, 100.0, nh).astype(np.float32)
        f["k"] = k
        f["z"] = z_of_k(k).astype(np.float32)
        f["thr"] = rng.integers(1, 4, nh).astype(np.float32)
        f["energy"] = np.full(n_ev, energy, np.float32)
        f["anchor_label"] = labels
        f["particle_type"] = labels
        f["filter_status"] = np.ones(n_ev, np.int64)
    return path


lab70 = np.full(400, -1, np.int64)
lab70[:40] = 1
lab70[40:80] = 2
lab30 = np.full(600, -1, np.int64)
lab30[:30] = 0
lab30[30:60] = 1
lab30[60:90] = 2
p70 = make(TMP + "/p70.h5", 400, 1, 50, lambda k: 2.8 * k, 70.0, lab70)
p30 = make(TMP + "/p30.h5", 600, 0, 48, lambda k: 2.8 * k, 30.0, lab30)
el70 = make(TMP + "/e70.h5", 20, 1, 48, lambda k: 2.8 * (k - 1), 70.0, np.zeros(20, np.int64))

data_cfg = dict(
    path=p70, val_ratio=0.2, seed=42, force_anchor=None,
    hit_affine={"x": [10.0, -5.204], "y": [10.0, -5.204], "z": [10.0, 198.5], "k": [1.0, -1.0]},
    anchor_datasets=[
        dict(path=p30, anchor_label=None,
             hit_affine={"x": [10.0, -5.204], "y": [10.0, -5.204],
                         "z": [10.0, 226.5], "k": [1.0, 0.0]}),
        dict(path=el70, anchor_label=0,
             hit_affine={"x": [10.0, -5.204], "y": [10.0, -5.204],
                         "z": [10.0, 226.5], "k": [1.0, -1.0]}),
    ],
    stratify_split=["anchor", "energy"],
    ignore_anchor_labels=[3, 4, 5], drop_ignored_anchor_events=False,
    field_map=dict(offsets="offsets",
                   hit=dict(x="x", y="y", z="z", k="k", thr="thr"),
                   event=dict(energy="energy", anchor_label="anchor_label",
                              class_label="particle_type", filter_status="filter_status")),
    anchor_field="anchor_label", eval_label_field="class_label",
)
features = dict(mv_point=["x", "y", "z"], mv_scalar=["thr"], scalars=[],
                thr_encoding="ordinal", extra_event_scalars=["energy"])
# z y k se dejan SIN escalar: asi la comprobacion del marco se hace en mm reales.
scaling = dict(source="online", stats_path=TMP + "/stats.yml",
               features={"x": {"mode": "z_norm"}, "y": {"mode": "z_norm"},
                         "z": {"mode": "none"}, "k": {"mode": "none"},
                         "thr": {"mode": "none"}, "energy": {"mode": "log_z"},
                         "density": {"mode": "log_z"}})

train_ds, val_ds, ds = make_clustering_splits(data_cfg, features, scaling)

ok = True


def check(name, cond, extra=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else ""))
    ok &= bool(cond)


print("\n=== 1. los tres marcos colapsan en uno ===")
z, k = ds.hit["z"], ds.hit["k"]
resid = np.abs(z - (28.0 * k + 226.5))
check("z == 28*k + 226.5 en TODAS las fuentes", resid.max() < 1e-2,
      "resid max=%.4f mm" % resid.max())
check("k >= 0 en todas las fuentes (nada fuera del minmax)", k.min() >= 0,
      "k min=%.1f" % k.min())

print("\n=== 2. el 30 GeV conserva sus propias etiquetas ===")
a = np.asarray(ds.event["anchor_label"])
e = ds._energy_raw
got = collections.Counter(zip(a[a >= 0].tolist(), np.rint(e[a >= 0]).astype(int).tolist()))
print("   (clase, energia) ->", dict(sorted(got.items())))
check("electrones a 30 GeV presentes (anchor_label: null respetado)", got[(0, 30)] == 30)
check("electrones a 70 GeV forzados desde el fichero dedicado", got[(0, 70)] == 20)
check("pi/mu en ambas energias", all(got[(c, en)] > 0 for c in (1, 2) for en in (30, 70)))

print("\n=== 3. split estratificado por (clase, energia) ===")
val_idx = np.asarray(val_ds.indices)
a_ds = a[ds._event_indices]
e_ds = np.rint(e[ds._event_indices]).astype(int)
is_val = np.zeros(ds.len(), bool)
is_val[val_idx] = True
for c in (0, 1, 2):
    for en in (30, 70):
        m = (a_ds == c) & (e_ds == en)
        if m.sum():
            check("celda (clase %d, %d GeV) tiene held-out" % (c, en),
                  (m & is_val).sum() >= 1,
                  "%d/%d" % (int((m & is_val).sum()), int(m.sum())))

print("\n=== 4. energy: escalada al modelo, cruda para la metrica ===")
d0 = ds.get(0)
check("data.energy_gev es 30 o 70 (sin escalar)", float(d0.energy_gev) in (30.0, 70.0),
      str(float(d0.energy_gev)))
check("data.energy si esta escalada (log_z)",
      abs(float(d0.energy)) < 5.0 and float(d0.energy) != float(d0.energy_gev),
      "%.3f" % float(d0.energy))

print("\n=== 5. feature routing: la energia llega a la entrada escalar ===")
from src.data.feature_routing import build_inputs, compute_in_s_channels  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

check("in_s_channels = 1 (era 0 en s14a)", compute_in_s_channels(features) == 1)
batch = next(iter(DataLoader([ds.get(i) for i in range(8)], batch_size=8)))
inp = build_inputs(batch, features)
check("scalars tiene forma (n_hits, 1)", inp["scalars"].shape[1] == 1,
      str(tuple(inp["scalars"].shape)))
check("la energia se difunde por hit", len(set(inp["scalars"].view(-1).tolist())) <= 2)

print("\n=== 6. compatibilidad hacia atras (s14a: afin global, split aleatorio) ===")
cfg14 = dict(data_cfg)
cfg14["anchor_datasets"] = [dict(path=el70, anchor_label=0)]
cfg14["stratify_split"] = None
scal14 = dict(scaling)
scal14["stats_path"] = TMP + "/stats14.yml"
t2, v2, ds2 = make_clustering_splits(cfg14, features, scal14)
z2, k2 = ds2.hit["z"], ds2.hit["k"]
check("un solo afin global aplicado a todo (comportamiento s14a)",
      np.abs(z2 - (28.0 * k2 + 226.5)).max() > 1.0,
      "el desfase del fichero de electrones sigue ahi, como antes")
check("el split aleatorio sigue funcionando", len(v2) > 0 and len(t2) > 0)

print("\n" + ("TODO OK" if ok else "HAY FALLOS"))
sys.exit(0 if ok else 1)
