"""
Training-time diagnostic plots.

``plot_latent_pca`` renders a 2D PCA of the event latent space z: all events are
colored by their assigned cluster (argmax of the cosine logits), the anchors are
highlighted (larger, black-edged markers colored by their true anchor class),
and the learnable prototypes are overlaid as stars. Watching this across epochs
shows how events settle around the prototypes and whether anchors stay in their
own group.
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_latent_pca(z, cluster, anchor, prototypes, epoch, out_path, title=None):
    """
    z          : (M, d)   L2-normalized event embeddings
    cluster    : (M,)     assigned cluster per event (argmax logits)
    anchor     : (M,)     anchor class per event (-1 = not an anchor)
    prototypes : (K, d)   current prototype vectors (any norm)

    Saves the figure to ``out_path`` and RETURNS it (not closed) so callers can
    also log it (e.g. wandb.Image). Remember to ``plt.close(fig)`` afterwards.
    """
    from sklearn.decomposition import PCA

    z = np.asarray(z)
    cluster = np.asarray(cluster)
    anchor = np.asarray(anchor)
    proto = np.asarray(prototypes)
    proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    pca = PCA(n_components=2).fit(z)
    e = pca.transform(z)
    ep = pca.transform(proto)
    K = proto.shape[0]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    # all events, colored by assigned cluster
    ax.scatter(e[:, 0], e[:, 1], c=cluster, cmap="tab10", s=6, alpha=0.35,
               vmin=0, vmax=max(K - 1, 1), linewidths=0)

    # anchors highlighted by their true class
    amask = anchor >= 0
    if amask.any():
        ax.scatter(e[amask, 0], e[amask, 1], c=anchor[amask], cmap="tab10",
                   s=70, edgecolors="black", linewidths=0.8, vmin=0, vmax=max(K - 1, 1),
                   label="anchors")

    # prototypes as stars
    ax.scatter(ep[:, 0], ep[:, 1], c=range(K), cmap="tab10", marker="*", s=420,
               edgecolors="black", linewidths=1.2, vmin=0, vmax=max(K - 1, 1),
               label="prototypes")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title or f"latent PCA (epoch {epoch}) — color=cluster, edged=anchors, stars=prototypes")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    return fig
