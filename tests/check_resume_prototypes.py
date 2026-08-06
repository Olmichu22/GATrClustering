"""Al reanudar, on_fit_start NO debe tocar los prototipos del checkpoint.

    apptainer exec -B /nfs:/nfs <img> python tests/check_resume_prototypes.py

Solo se ejercita la guarda, que devuelve antes de cualquier forward: por eso el
test corre sin GPU (GATr no funciona en CPU). Un trainer de mentira aporta las
dos únicas cosas que el hook mira antes de decidir: is_global_zero y ckpt_path.
"""
import os
import sys

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.train_clustering import ClusteringLitModule  # noqa: E402


class FakeTrainer:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.is_global_zero = True
        self.train_dataloader = None


cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "manual_s15.yml")))
ok = True


def check(name, cond, extra=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else ""))
    ok &= bool(cond)


class FakeBatch:
    """Lo mínimo que el hook toca de un batch."""

    def __init__(self, n, k):
        self.anchor_label = torch.arange(n) % k  # todos anclados

    def to(self, _device):
        return self


def protos_after_fit_start(ckpt_path, force=False, with_loader=True):
    """Prototipos antes/después de on_fit_start, con el forward SIMULADO.

    El forward real necesita GPU (GATr no corre en CPU), así que se sustituye por
    embeddings deterministas: lo que se está probando es la guarda de reanudación,
    no la calidad de la inicialización.
    """
    c = yaml.safe_load(yaml.safe_dump(cfg))  # copia profunda
    c["train"]["reinit_prototypes_on_resume"] = force
    m = ClusteringLitModule(c)
    K = c["model"]["head"]["num_clusters"]
    dim = c["model"]["head"]["proj_dim"]
    with torch.no_grad():  # marca reconocible: si se re-inicializa, cambia
        m.model.head.prototypes.fill_(0.123)
    before = m.model.head.prototypes.detach().clone()

    m.model.forward = lambda batch: {
        "z": torch.nn.functional.normalize(
            torch.arange(batch.anchor_label.numel() * dim, dtype=torch.float32
                         ).reshape(-1, dim), dim=1)
    }
    m._trainer = FakeTrainer(ckpt_path)
    m._train_loader = [FakeBatch(32, K)] if with_loader else None
    m.on_fit_start()
    return before, m.model.head.prototypes.detach().clone()


print("=== reanudando (ckpt_path presente) ===")
b, a = protos_after_fit_start("/ruta/al/best.ckpt")
check("los prototipos del checkpoint se conservan", torch.equal(b, a),
      "max|diff| = %.3e" % (b - a).abs().max())

print("\n=== desde cero (sin ckpt_path) ===")
b, a = protos_after_fit_start(None)
check("sin reanudar SÍ se inicializan desde los anchors", not torch.equal(b, a),
      "max|diff| = %.3f" % (b - a).abs().max())

print("\n=== escotilla de escape ===")
b, a = protos_after_fit_start("/ruta/al/best.ckpt", force=True)
check("reinit_prototypes_on_resume: true vuelve al comportamiento anterior",
      not torch.equal(b, a), "max|diff| = %.3f" % (b - a).abs().max())

print("\n=== la config de s15 no fuerza la re-init ===")
check("reinit_prototypes_on_resume ausente o false",
      not cfg["train"].get("reinit_prototypes_on_resume", False))

print("\n" + ("TODO OK" if ok else "HAY FALLOS"))
sys.exit(0 if ok else 1)
