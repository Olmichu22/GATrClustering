"""
Training entry point for the clustering POC.

    L = loss.swap.weight * L_swap(two dropout views)
      + lambda_ce        * CE(anchors)
      + lambda_vicreg    * (var + cov)(z)

Prototypes are initialized (non-random) from the mean anchor embedding of each
class before the loop. lambda_ce is high at the start (anchors are the most
reliable signal). No Sinkhorn / no proportion prior in this first iteration.

Run inside the Apptainer image gatr_v9.sif (see GATrAutoencoder/CLAUDE.md).
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

from .augment import hit_dropout
from .data.dataset import make_clustering_splits
from .losses.swap_loss import swap_loss
from .losses.vicreg import vicreg_loss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .models.clustering_model import ClusteringModel
from .plots import plot_latent_pca


def setup_wandb(cfg: dict):
    """Init W&B if enabled (config train.wandb); returns the module or None."""
    wcfg = cfg["train"].get("wandb", {}) or {}
    if not wcfg.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; skipping")
        return None
    wandb.init(
        project=wcfg.get("project", "gatr-clustering"),
        entity=wcfg.get("entity"),
        name=wcfg.get("name"),
        mode=wcfg.get("mode", "online"),
        config=cfg,
    )
    return wandb


def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


@torch.no_grad()
def init_prototypes(model, loader, device):
    """Collect z of all anchor events over the train loader and init prototypes."""
    model.eval()
    zs, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        anchor = batch.anchor_label
        mask = anchor >= 0
        if mask.any():
            out = model(batch)
            zs.append(out["z"][mask].detach())
            labels.append(anchor[mask])
    if zs:
        z = torch.cat(zs, 0)
        lab = torch.cat(labels, 0)
        model.head.init_prototypes_from_anchors(z, lab)
        print(f"[init] Prototypes initialized from {z.shape[0]} anchor events.")
    else:
        print("[init] WARNING: no anchor events found; prototypes stay random.")


def anchor_ce(logits, anchor_label):
    mask = anchor_label >= 0
    if not mask.any():
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[mask], anchor_label[mask])


def train(cfg: dict):
    tcfg = cfg["train"]
    device = torch.device(tcfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    os.makedirs(tcfg["out_dir"], exist_ok=True)

    train_ds, val_ds, _ = make_clustering_splits(cfg["data"], cfg["features"], cfg["scaling"])
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=tcfg.get("num_workers", 4),
    )
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model = ClusteringModel(cfg["model"], cfg["features"]).to(device)
    init_prototypes(model, train_loader, device)

    wb = setup_wandb(cfg)
    if wb is not None:
        wb.log({"model/params": sum(p.numel() for p in model.parameters())})

    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 1e-4)
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])

    p_drop = cfg["augment"]["hit_dropout"]
    lcfg = cfg["loss"]
    swap_w = lcfg["swap"]["weight"]
    sharpen = lcfg["swap"].get("sharpen_temp", 0.25)
    lam_ce = lcfg["lambda_ce"]
    lam_vic = lcfg["lambda_vicreg"]
    vcfg = lcfg["vicreg"]

    for epoch in range(tcfg["epochs"]):
        model.train()
        agg = {"loss": 0.0, "swap": 0.0, "ce": 0.0, "vic": 0.0, "n": 0}
        for batch in train_loader:
            batch = batch.to(device)
            view_a = hit_dropout(batch, p_drop)
            view_b = hit_dropout(batch, p_drop)

            out_a = model(view_a)
            out_b = model(view_b)

            l_swap = swap_loss(out_a["logits"], out_b["logits"], sharpen_temp=sharpen)
            l_ce = anchor_ce(out_a["logits"], batch.anchor_label)
            l_vic, _, _ = vicreg_loss(
                out_a["z"], vcfg["var_weight"], vcfg["cov_weight"], vcfg.get("var_gamma", 1.0)
            )
            loss = swap_w * l_swap + lam_ce * l_ce + lam_vic * l_vic

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            agg["loss"] += float(loss)
            agg["swap"] += float(l_swap)
            agg["ce"] += float(l_ce)
            agg["vic"] += float(l_vic)
            agg["n"] += 1
        sched.step()

        n = max(agg["n"], 1)
        lr_now = sched.get_last_lr()[0]
        print(
            f"[epoch {epoch:03d}] loss={agg['loss']/n:.4f} "
            f"swap={agg['swap']/n:.4f} ce={agg['ce']/n:.4f} vic={agg['vic']/n:.4f}"
        )
        if wb is not None:
            wb.log({
                "loss/total": agg["loss"] / n, "loss/swap": agg["swap"] / n,
                "loss/ce": agg["ce"] / n, "loss/vic": agg["vic"] / n,
                "lr": lr_now, "epoch": epoch,
            })

        if (epoch + 1) % tcfg.get("plot_every", 5) == 0 or epoch + 1 == tcfg["epochs"]:
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                os.path.join(tcfg["out_dir"], "last.ckpt"),
            )
            z, cluster, anchor = collect_latent(
                model, val_loader, device, max_events=tcfg.get("plot_max_events", 3000)
            )
            # held-out anchor accuracy (val-split anchors never entered CE / proto init)
            amask = anchor >= 0
            acc = float((cluster[amask] == anchor[amask]).mean()) if amask.any() else float("nan")
            if amask.any():
                print(f"           val held-out anchor acc={acc:.3f} ({int(amask.sum())} anchors)")
            proto = model.head.prototypes.detach().cpu().numpy()
            fig = plot_latent_pca(
                z, cluster, anchor, proto, epoch,
                os.path.join(tcfg["out_dir"], "pca", f"epoch_{epoch:03d}.png"),
            )
            if wb is not None:
                log = {"latent/pca": wb.Image(fig), "epoch": epoch}
                if amask.any():
                    log["val/heldout_anchor_acc"] = acc
                wb.log(log)
            plt.close(fig)

    if wb is not None:
        wb.finish()
    print(f"[done] checkpoint at {os.path.join(tcfg['out_dir'], 'last.ckpt')}")


@torch.no_grad()
def collect_latent(model, loader, device, max_events=3000):
    """Collect z, assigned cluster and anchor label over (a capped subset of) a loader."""
    was_training = model.training
    model.eval()
    Z, CL, AN, seen = [], [], [], 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        Z.append(out["z"].cpu().numpy())
        CL.append(out["logits"].argmax(1).cpu().numpy())
        AN.append(batch.anchor_label.cpu().numpy())
        seen += out["z"].shape[0]
        if seen >= max_events:
            break
    if was_training:
        model.train()
    import numpy as np
    return np.concatenate(Z), np.concatenate(CL), np.concatenate(AN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--data_path", default=None, help="override data.path")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=None,
                    help="enable W&B logging (overrides config)")
    ap.add_argument("--no_wandb", dest="wandb", action="store_false",
                    help="disable W&B logging (overrides config)")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_mode", default=None, help="online | offline | disabled")
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    if args.data_path:
        cfg["data"]["path"] = args.data_path
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.out_dir:
        cfg["train"]["out_dir"] = args.out_dir
    if args.device:
        cfg["train"]["device"] = args.device

    cfg["train"].setdefault("wandb", {})
    if args.wandb is not None:
        cfg["train"]["wandb"]["enabled"] = args.wandb
    if args.wandb_project:
        cfg["train"]["wandb"]["project"] = args.wandb_project
    if args.wandb_mode:
        cfg["train"]["wandb"]["mode"] = args.wandb_mode
    if args.run_name:
        cfg["train"]["wandb"]["name"] = args.run_name
    train(cfg)


if __name__ == "__main__":
    main()
