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


def cluster_colors(K):
    """Stable discrete color per cluster index (tab10), so plots and legends agree."""
    cmap = plt.get_cmap("tab10")
    return [cmap(k % 10) for k in range(K)]


def cluster_label(k, class_names=None):
    """Legend label for cluster k, e.g. '0 · electron' when names are known."""
    if class_names is not None and k < len(class_names) and class_names[k]:
        return f"{k} · {class_names[k]}"
    return f"cluster {k}"


def plot_latent_pca(z, cluster, anchor, prototypes, epoch, out_path, title=None,
                    class_names=None):
    """
    z          : (M, d)   L2-normalized event embeddings
    cluster    : (M,)     assigned cluster per event (argmax logits)
    anchor     : (M,)     anchor class per event (-1 = not an anchor)
    prototypes : (K, d)   current prototype vectors (any norm)
    class_names: optional list mapping cluster index -> particle name (e.g. from
                 anchors.yml); when given, the legend shows the particle per cluster.

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

    colors = cluster_colors(K)

    # all events + prototypes, per cluster so the legend names the particle
    for k in range(K):
        m = cluster == k
        if m.any():
            ax.scatter(e[m, 0], e[m, 1], color=colors[k], s=6, alpha=0.35,
                       linewidths=0, label=cluster_label(k, class_names))
    # anchors highlighted by their true class (same color, black edge)
    amask = anchor >= 0
    for k in range(K):
        mk = amask & (anchor == k)
        if mk.any():
            ax.scatter(e[mk, 0], e[mk, 1], color=colors[k], s=70,
                       edgecolors="black", linewidths=0.8)
    # prototypes as stars, colored per cluster
    ax.scatter(ep[:, 0], ep[:, 1], color=colors, marker="*", s=420,
               edgecolors="black", linewidths=1.2)

    # legend proxies for the marker roles (color-independent)
    from matplotlib.lines import Line2D
    role_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="0.6",
               markeredgecolor="black", markersize=8, label="anchors (edged)"),
        Line2D([0], [0], marker="*", linestyle="", markerfacecolor="0.6",
               markeredgecolor="black", markersize=15, label="prototypes"),
    ]
    cluster_handles, cluster_labels = ax.get_legend_handles_labels()
    ax.legend(cluster_handles + role_handles,
              cluster_labels + [h.get_label() for h in role_handles],
              loc="best", fontsize=8, title="cluster · particle")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title or f"latent PCA (epoch {epoch}) — color=cluster, edged=anchors, stars=prototypes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    return fig
