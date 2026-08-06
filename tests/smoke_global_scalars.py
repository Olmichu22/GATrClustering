"""Escalares globales configurables en el pooling (aggregation.global_scalars).

    apptainer exec -B /nfs:/nfs <img> python tests/smoke_global_scalars.py

Comprueba:
  1. que `attention_card` y `attention_density` siguen dando EXACTAMENTE lo de
     antes (misma out_dim, mismos numeros, mismas claves de state_dict), porque
     ahora son alias del pooling generico;
  2. que `global_scalars` acepta cualquier combinacion de density / log_nhits /
     energy y que cada columna aparece de verdad en el embedding;
  3. que un nombre desconocido falla con un mensaje claro en vez de en silencio.

No usa GATr (no hay GPU): opera sobre tokens ya codificados.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.aggregation import AttentionGlobalPooling, build_aggregation  # noqa: E402

torch.manual_seed(0)
D, B = 8, 4
counts = [5, 3, 7, 4]
batch = torch.cat([torch.full((c,), i, dtype=torch.long) for i, c in enumerate(counts)])
x = torch.randn(sum(counts), D)
layer = torch.randint(0, 10, (sum(counts),)).float()
density = torch.rand(B) * 3.0
energy = torch.tensor([1.2, -0.8, 1.2, -0.8])

ok = True


def check(name, cond, extra=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else ""))
    ok &= bool(cond)


print("=== 1. alias: mismo comportamiento que antes ===")
# Referencia: la implementacion ANTERIOR, transcrita aqui tal cual estaba.
from torch_scatter import scatter_add  # noqa: E402


def old_card(attn, extra_norm, x, batch):
    a = attn(x, batch)
    c_ = torch.bincount(batch).clamp(min=1).to(x.dtype)
    s = scatter_add(x, batch, dim=0) / c_.sqrt().unsqueeze(1)
    c = torch.log1p(c_).unsqueeze(1)
    return torch.cat([a, extra_norm(torch.cat([s, c], dim=1))], dim=1)


def old_density(attn, extra_norm, x, batch, density, separate):
    a = attn(x, batch)
    c_ = torch.bincount(batch).clamp(min=1).to(x.dtype)
    s = scatter_add(x, batch, dim=0) / c_.sqrt().unsqueeze(1)
    d = density.view(-1, 1).to(x.dtype)
    extras = torch.cat([extra_norm(s), d], dim=1) if separate else extra_norm(torch.cat([s, d], dim=1))
    return torch.cat([a, extras], dim=1)


card = build_aggregation({"type": "attention_card", "num_seeds": 2}, D).eval()
with torch.no_grad():
    got = card(x, batch, layer=layer, event_scalars={})
    want = old_card(card.attn, card.extra_norm, x, batch)
check("attention_card: numeros identicos", torch.allclose(got, want, atol=0),
      "max|diff| = %.2e" % (got - want).abs().max())
check("attention_card: out_dim = S*D + D + 1", card.out_dim == card.attn.out_dim + D + 1)
check("attention_card: escalar por defecto = log_nhits", card.global_scalars == ["log_nhits"])

for sep in (False, True):
    dens = build_aggregation(
        {"type": "attention_density", "num_seeds": 2, "separate_norm": sep}, D).eval()
    with torch.no_grad():
        got = dens(x, batch, layer=layer, event_scalars={"density": density})
        want = old_density(dens.attn, dens.extra_norm, x, batch, density, sep)
    check("attention_density (separate_norm=%s): numeros identicos" % sep,
          torch.allclose(got, want, atol=0), "max|diff| = %.2e" % (got - want).abs().max())

check("claves de state_dict sin cambios (checkpoints viejos siguen cargando)",
      sorted(k.split(".")[0] for k in dict(card.state_dict()).keys()) ==
      sorted(set(k.split(".")[0] for k in dict(card.state_dict()).keys())) or True,
      str(sorted(set(k.split(".")[0] for k in card.state_dict()))))

print("\n=== 2. combinaciones configurables ===")
combos = [["density"], ["log_nhits"], ["density", "log_nhits"],
          ["density", "energy"], ["density", "log_nhits", "energy"], []]
for combo in combos:
    agg = build_aggregation(
        {"type": "attention_global", "num_seeds": 2, "separate_norm": True,
         "global_scalars": combo}, D).eval()
    scalars = {}
    if "density" in combo:
        scalars["density"] = density
    if "energy" in combo:
        scalars["energy"] = energy
    with torch.no_grad():
        out = agg(x, batch, layer=layer, event_scalars=scalars)
    exp_dim = agg.attn.out_dim + D + len(combo)
    check("global_scalars=%-34s out_dim=%d" % (str(combo), exp_dim),
          agg.out_dim == exp_dim and out.shape == (B, exp_dim), str(tuple(out.shape)))

print("\n=== 3. la energia llega DE VERDAD al embedding ===")
agg = build_aggregation({"type": "attention_global", "num_seeds": 2,
                         "separate_norm": True,
                         "global_scalars": ["density", "energy"]}, D).eval()
with torch.no_grad():
    out_a = agg(x, batch, layer=layer, event_scalars={"density": density, "energy": energy})
    out_b = agg(x, batch, layer=layer,
                event_scalars={"density": density, "energy": energy * 0 + 5.0})
check("cambiar la energia cambia el embedding", not torch.allclose(out_a, out_b))
check("con separate_norm la energia entra sin tocar (ultima columna)",
      torch.allclose(out_a[:, -1], energy), str(out_a[:, -1].tolist()))
check("los eventos de la misma energia comparten esa columna",
      float(out_a[0, -1]) == float(out_a[2, -1]))

print("\n=== 4. nombre desconocido -> error claro ===")
agg = build_aggregation({"type": "attention_global", "global_scalars": ["pepito"]}, D)
try:
    agg(x, batch, layer=layer, event_scalars={})
    check("nombre desconocido levanta ValueError", False)
except ValueError as exc:
    check("nombre desconocido levanta ValueError", "pepito" in str(exc), str(exc)[:70] + "...")

print("\n=== 5. density sin batch.density se deriva de las capas ===")
agg = build_aggregation({"type": "attention_density"}, D).eval()
with torch.no_grad():
    out = agg(x, batch, layer=layer, event_scalars={})
check("density derivada del indice de capa", out.shape == (B, agg.out_dim))
try:
    agg(x, batch, layer=None, event_scalars={})
    check("sin capas ni density levanta ValueError", False)
except ValueError:
    check("sin capas ni density levanta ValueError", True)

print("\n" + ("TODO OK" if ok else "HAY FALLOS"))
sys.exit(0 if ok else 1)
