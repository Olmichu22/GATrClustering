"""El early stopping configurable y la reanudacion de s15 hacen lo que dicen.

    apptainer exec -B /nfs:/nfs <img> python tests/check_s15_resume.py

Lo que de verdad puede salir mal al reanudar cambiando el monitor: que Lightning
restaure el estado del EarlyStopping viejo, cuyo best_score pertenece a
`val/loss` (~2.9). Comparando una accuracy contra eso, el run pararia en la
primera validacion. Se comprueba contra el CHECKPOINT REAL que la clave de
estado guardada corresponde al monitor viejo y no a la del callback nuevo.
"""
import os
import sys

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.train_clustering import ResumableEarlyStopping  # noqa: E402

CFG = os.path.join(REPO, "configs", "manual_s15.yml")
SUB = os.path.join(REPO, "condor", "manual_s15_resume.sub")
cfg = yaml.safe_load(open(CFG))
tcfg = cfg["train"]

ok = True


def check(name, cond, extra=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else ""))
    ok &= bool(cond)


print("=== config ===")
check("early_stop_monitor == ckpt_monitor",
      tcfg.get("early_stop_monitor") == tcfg.get("ckpt_monitor"),
      "%s vs %s" % (tcfg.get("early_stop_monitor"), tcfg.get("ckpt_monitor")))
check("early_stop_mode = max (la accuracy sube)", tcfg.get("early_stop_mode") == "max")

print("\n=== submit de reanudacion ===")
sub = open(SUB).read()
resume = [t.split("=", 1)[1] for t in sub.split() if t.startswith("RESUME=")]
check("el .sub define RESUME", bool(resume))
if resume:
    ckpt = resume[0].rstrip('"')
    check("el checkpoint existe", os.path.exists(ckpt), os.path.basename(ckpt))
    check("apunta al MEJOR, no a last.ckpt", "best-" in os.path.basename(ckpt))
    check("el .sub usa el mismo config arreglado", "CFG=configs/manual_s15.yml" in sub)
    check("OMP_NUM_THREADS fijado", "OMP_NUM_THREADS=" in sub)

    print("\n=== estado del EarlyStopping dentro del checkpoint ===")
    blob = torch.load(ckpt, map_location="cpu")
    cbs = blob.get("callbacks", {})
    es_keys = [k for k in cbs if "EarlyStopping" in str(k)]
    print("   claves guardadas:", es_keys)
    old_key = es_keys[0] if es_keys else None
    if old_key:
        print("   best_score guardado:", cbs[old_key].get("best_score"))
    # La MISMA clase que usa el entrenador: la state_key incluye el nombre de la
    # clase además del monitor, así que compararla con la base engañaría.
    new_key = ResumableEarlyStopping(monitor=tcfg["early_stop_monitor"],
                                     mode=tcfg["early_stop_mode"]).state_key
    print("   clave del callback NUEVO:", new_key)
    check("la clave cambia -> el estado viejo NO se hereda", old_key != new_key)
    check("el estado viejo era de val/loss", old_key is None or "val/loss" in str(old_key))

    print("\n=== epoca desde la que sigue ===")
    ep = blob.get("epoch")
    check("el checkpoint es de la epoca 5", ep == 5, "epoch=%s" % ep)
    print("   -> reanuda en la 6 y llega a la %s del config: %s epocas mas"
          % (tcfg["epochs"], tcfg["epochs"] - (ep + 1)))

print("\n" + ("TODO OK" if ok else "HAY FALLOS"))
sys.exit(0 if ok else 1)
