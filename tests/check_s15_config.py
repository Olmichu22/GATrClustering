"""Valida configs/manual_s15.yml SIN cargar los 2.4M eventos.

Comprueba que existen las tres fuentes, mide la relacion z<->k real de cada
fichero (sobre una muestra de hits) y verifica ANALITICAMENTE que el afin
declarado para ella la manda al marco de entrenamiento comun:

    z_mm = 28 * k' + 226.5,   k' = capa 0-based

Tambien comprueba el trozo del entrenador que agrupa por energia.
"""
import os
import sys

import h5py
import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CFG = os.path.join(REPO, "configs", "manual_s15.yml")
cfg = yaml.safe_load(open(CFG))
data = cfg["data"]

ok = True


def check(name, cond, extra=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else ""))
    ok &= bool(cond)


sources = [("primario 70 GeV", data["path"], data["hit_affine"])]
for ds in data["anchor_datasets"]:
    sources.append((os.path.basename(ds["path"]), ds["path"],
                    ds.get("hit_affine") or data["hit_affine"]))

print("=== fuentes y marcos ===")
for name, path, aff in sources:
    if not os.path.exists(path):
        check("existe %s" % name, False, path)
        continue
    with h5py.File(path, "r") as f:
        n = int(len(f["offsets"]) - 1)
        s = min(2_000_000, f["k"].shape[0])
        k = np.asarray(f["k"][:s], dtype=np.float64)
        z = np.asarray(f["z"][:s], dtype=np.float64)
        e = np.unique(f["energy"][: min(n, 100_000)])
    slope, intercept = np.polyfit(k, z, 1)
    # afin declarado -> marco destino
    zs, zo = (float(v) for v in aff["z"])
    ks, ko = (float(v) for v in aff["k"])
    # z_mm(k) = zs*(slope*k + intercept) + zo ;  k'(k) = ks*k + ko
    # objetivo: z_mm = 28*k' + 226.5 para todo k
    a_got, b_got = zs * slope, zs * intercept + zo          # z_mm = a_got*k + b_got
    a_want, b_want = 28.0 * ks, 28.0 * ko + 226.5           # 28*k' + 226.5
    print("  %-42s n=%-9d E=%s  z=%.3f*k%+.3f" % (name, n, e.tolist(), slope, intercept))
    check("    %s cae en el marco comun" % name,
          abs(a_got - a_want) < 1e-3 and abs(b_got - b_want) < 1e-2,
          "z_mm = %.3f*k %+.3f  (esperado %.3f*k %+.3f)" % (a_got, b_got, a_want, b_want))
    kmin_new = ks * k.min() + ko
    check("    %s no manda capas a k' < 0" % name, kmin_new >= 0, "k' min = %.1f" % kmin_new)

print("\n=== coherencia del config ===")
check("stratify_split incluye clase y energia",
      set(data.get("stratify_split") or []) == {"anchor", "energy"})
agg = cfg["model"]["aggregation"]
gs = agg.get("global_scalars") or []
check("energy entra como escalar GLOBAL (pooling), no difundida por hit",
      "energy" in gs and "energy" not in (cfg["features"].get("extra_event_scalars") or []),
      "global_scalars=%s  extra_event_scalars=%s" % (gs, cfg["features"].get("extra_event_scalars")))
check("la densidad sigue estando", "density" in gs)
check("energy tiene modo de escalado", "energy" in cfg["scaling"]["features"])
check("separate_norm con escalares ya escalados", agg.get("separate_norm") is True)
check("K = 3", cfg["model"]["head"]["num_clusters"] == 3)
check("prior con 3 componentes que suman ~1",
      abs(sum(cfg["loss"]["cluster_prior"]) - 1.0) < 1e-6)
check("el 30 GeV entra como primario (anchor_label: null)",
      any(d.get("anchor_label") is None for d in data["anchor_datasets"]))
check("stats_path nuevo (no reutiliza el de s14a)",
      "s15" in cfg["scaling"]["stats_path"])

print("\n=== entrenador: agrupacion por energia ===")
import torch  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402

from src.train_clustering import ClusteringLitModule  # noqa: E402

b = Batch.from_data_list([
    Data(anchor_label=torch.tensor([1]), energy_gev=torch.tensor([70.0])),
    Data(anchor_label=torch.tensor([0]), energy_gev=torch.tensor([30.0])),
])
en = ClusteringLitModule._batch_energy(b)
check("_batch_energy devuelve GeV enteros por evento", en.tolist() == [70, 30], str(en.tolist()))
b_old = Batch.from_data_list([Data(anchor_label=torch.tensor([1]))])
check("sin energy_gev (configs viejos) no revienta",
      ClusteringLitModule._batch_energy(b_old).tolist() == [0])

print("\n" + ("TODO OK" if ok else "HAY FALLOS"))
sys.exit(0 if ok else 1)
