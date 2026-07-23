"""
Evaluation for the clustering POC. With almost no real labels available, the
PRIMARY diagnostic is the nHits(total) distribution split by assigned cluster
(e/pi/mu separate well in hit count). Secondary checks use the few labels.

Produces (saved to ``out_dir``):
    1. Anchor sanity: accuracy of anchors vs their prototypes.
    2. nHits(total) distribution colored by assigned cluster  (PRIMARY).
    3. t-SNE of z colored by cluster (and by class_label where available).
    4. Confusion matrix on the few labeled non-anchor events (Hungarian-aligned).
    5. Where 'extra'/unlabeled events fall across clusters.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data.dataset import make_clustering_splits  # noqa: E402
from .models.clustering_model import ClusteringModel  # noqa: E402


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    Z, cl, anc, lab, nh = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        Z.append(out["z"].cpu().numpy())
        cl.append(out["logits"].argmax(1).cpu().numpy())
        anc.append(batch.anchor_label.cpu().numpy())
        lab.append(batch.class_label.cpu().numpy())
        nh.append(batch.nhits_total.cpu().numpy())
    return (
        np.concatenate(Z),
        np.concatenate(cl),
        np.concatenate(anc),
        np.concatenate(lab),
        np.concatenate(nh),
    )


def anchor_sanity(cluster, anchor):
    mask = anchor >= 0
    if not mask.any():
        print("[eval] no anchors present")
        return
    acc = float((cluster[mask] == anchor[mask]).mean())
    print(f"[eval] anchor sanity accuracy: {acc:.3f} ({int(mask.sum())} anchors)")


def plot_nhits_by_cluster(nhits, cluster, K, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.histogram_bin_edges(nhits, bins=40)
    for k in range(K):
        m = cluster == k
        if m.any():
            ax.hist(nhits[m], bins=bins, histtype="step", linewidth=2, label=f"cluster {k}")
    ax.set_xlabel("nHits (total)")
    ax.set_ylabel("events")
    ax.set_title("nHits distribution by assigned cluster (PRIMARY diagnostic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[eval] wrote {out_path}")


def plot_tsne(z, cluster, label, out_path, max_pts=4000):
    from sklearn.manifold import TSNE

    n = z.shape[0]
    idx = np.random.default_rng(0).choice(n, size=min(n, max_pts), replace=False)
    emb = TSNE(n_components=2, init="pca", perplexity=30).fit_transform(z[idx])

    has_label = (label[idx] >= 0).any()
    fig, axes = plt.subplots(1, 2 if has_label else 1, figsize=(12 if has_label else 6, 5), squeeze=False)
    axes[0][0].scatter(emb[:, 0], emb[:, 1], c=cluster[idx], s=6, cmap="tab10")
    axes[0][0].set_title("t-SNE of z (color = assigned cluster)")
    if has_label:
        lab = label[idx]
        m = lab >= 0
        axes[0][1].scatter(emb[m, 0], emb[m, 1], c=lab[m], s=8, cmap="tab10")
        axes[0][1].set_title("t-SNE of z (color = true class, where known)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[eval] wrote {out_path}")


def confusion_labeled(cluster, anchor, label, K, out_path):
    """Confusion on labeled NON-anchor events, Hungarian cluster<->class aligned."""
    from scipy.optimize import linear_sum_assignment

    mask = (label >= 0) & (anchor < 0)
    if not mask.any():
        print("[eval] no labeled non-anchor events for confusion matrix")
        return
    cl, la = cluster[mask], label[mask]
    classes = np.unique(la)
    C = np.zeros((K, len(classes)), dtype=int)
    for i, k in enumerate(range(K)):
        for j, c in enumerate(classes):
            C[i, j] = int(((cl == k) & (la == c)).sum())
    row, col = linear_sum_assignment(-C)
    acc = C[row, col].sum() / C.sum()
    print(f"[eval] labeled non-anchor accuracy (Hungarian): {acc:.3f} on {int(mask.sum())} events")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(C, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"class {c}" for c in classes])
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"cluster {k}" for k in range(K)])
    for i in range(K):
        for j in range(len(classes)):
            ax.text(j, i, C[i, j], ha="center", va="center")
    fig.colorbar(im)
    ax.set_title(f"confusion (labeled non-anchor), acc={acc:.2f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[eval] wrote {out_path}")


def evaluate(cfg, ckpt_path, out_dir, device_str):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    _, val_ds, _ = make_clustering_splits(cfg["data"], cfg["features"], cfg["scaling"])
    loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = ClusteringModel(cfg["model"], cfg["features"]).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    z, cluster, anchor, label, nhits = collect(model, loader, device)
    K = cfg["model"]["head"]["num_clusters"]
    nhits = nhits.reshape(-1)

    anchor_sanity(cluster, anchor)
    plot_nhits_by_cluster(nhits, cluster, K, os.path.join(out_dir, "nhits_by_cluster.png"))
    plot_tsne(z, cluster, label, os.path.join(out_dir, "tsne_z.png"))
    confusion_labeled(cluster, anchor, label, K, os.path.join(out_dir, "confusion.png"))

    # cluster occupancy of unlabeled events
    unl = label < 0
    if unl.any():
        occ = np.bincount(cluster[unl], minlength=K)
        print(f"[eval] unlabeled event occupancy per cluster: {occ.tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default=None, help="defaults to cfg stored in the checkpoint")
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--out_dir", default="results/eval")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.cfg:
        with open(args.cfg) as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = torch.load(args.ckpt, map_location="cpu")["cfg"]
    if args.data_path:
        cfg["data"]["path"] = args.data_path
    evaluate(cfg, args.ckpt, args.out_dir, args.device)


if __name__ == "__main__":
    main()
