"""
Evaluation for the clustering POC.

IMPORTANT: there is NO trustworthy ground truth. ``particle_type`` is just another
noisy classifier, so no confusion matrix is computed. Validation uses only:

  Anchor-based (held-out):
    Anchors that landed in the validation split never entered the anchor CE term
    nor the prototype initialization (both run over the train split only), so
    they form a genuine held-out set. We report their classification accuracy
    against the learned prototypes.

  Unsupervised:
    1. nHits(total) distribution split by assigned cluster (PRIMARY -- e/pi/mu
       separate in hit count).
    2. Silhouette score of z w.r.t. the assigned clusters (separation).
    3. Cluster occupancy (are clusters used, or collapsed?).
    4. t-SNE of z colored by assigned cluster, anchors highlighted.
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
from .plots import plot_latent_pca  # noqa: E402


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    Z, cl, anc, nh = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        Z.append(out["z"].cpu().numpy())
        cl.append(out["logits"].argmax(1).cpu().numpy())
        anc.append(batch.anchor_label.cpu().numpy())
        nh.append(batch.nhits_total.cpu().numpy())
    return (np.concatenate(Z), np.concatenate(cl), np.concatenate(anc),
            np.concatenate(nh).reshape(-1))


def heldout_anchor_accuracy(cluster, anchor):
    """Accuracy on validation-split anchors (held out from CE + prototype init)."""
    mask = anchor >= 0
    if not mask.any():
        print("[eval] no held-out anchors in the validation split")
        return
    acc = float((cluster[mask] == anchor[mask]).mean())
    print(f"[eval] held-out anchor accuracy: {acc:.3f} on {int(mask.sum())} val anchors")


def silhouette(z, cluster, max_pts=3000):
    from sklearn.metrics import silhouette_score

    K = len(np.unique(cluster))
    if K < 2 or z.shape[0] <= K:
        print(f"[eval] silhouette skipped (clusters used = {K})")
        return
    n = z.shape[0]
    idx = np.random.default_rng(0).choice(n, size=min(n, max_pts), replace=False)
    s = silhouette_score(z[idx], cluster[idx])
    print(f"[eval] silhouette score (z vs assigned cluster): {s:.3f}")


def cluster_occupancy(cluster, K):
    occ = np.bincount(cluster, minlength=K)
    frac = occ / max(occ.sum(), 1)
    print(f"[eval] cluster occupancy: {occ.tolist()}  (fractions {np.round(frac,3).tolist()})")


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


def plot_tsne(z, cluster, anchor, out_path, max_pts=4000):
    from sklearn.manifold import TSNE

    n = z.shape[0]
    idx = np.random.default_rng(0).choice(n, size=min(n, max_pts), replace=False)
    emb = TSNE(n_components=2, init="pca", perplexity=30).fit_transform(z[idx])
    cl, an = cluster[idx], anchor[idx]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(emb[:, 0], emb[:, 1], c=cl, s=6, cmap="tab10", alpha=0.4, linewidths=0)
    am = an >= 0
    if am.any():
        ax.scatter(emb[am, 0], emb[am, 1], c=an[am], s=70, cmap="tab10",
                   edgecolors="black", linewidths=0.8, label="anchors")
        ax.legend(fontsize=8)
    ax.set_title("t-SNE of z (color = assigned cluster; edged = anchors)")
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

    z, cluster, anchor, nhits = collect(model, loader, device)
    K = cfg["model"]["head"]["num_clusters"]

    # ---- anchor-based (held-out) ----
    heldout_anchor_accuracy(cluster, anchor)
    # ---- unsupervised ----
    silhouette(z, cluster)
    cluster_occupancy(cluster, K)
    plot_nhits_by_cluster(nhits, cluster, K, os.path.join(out_dir, "nhits_by_cluster.png"))
    plot_tsne(z, cluster, anchor, os.path.join(out_dir, "tsne_z.png"))
    proto = model.head.prototypes.detach().cpu().numpy()
    fig = plot_latent_pca(z, cluster, anchor, proto, state.get("epoch", -1),
                          os.path.join(out_dir, "pca_z.png"))
    plt.close(fig)


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
