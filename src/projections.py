"""2D projections of the clustering latent space.

Pure numpy (no sklearn / torch), so the same code runs inside the GATr container
(evaluate_clustering.py, which writes the .npz) and outside it (the Dash
explorer, which only re-reads what was written).

Why more than PCA:

* **t-SNE is a poor fit for these latents.** z lives on the unit hypersphere and
  the clusters are extremely unbalanced (the muon cluster is ~97% of the sample).
  t-SNE's perplexity-based neighbourhoods then shred the dominant cluster into
  arbitrary islands, it has no out-of-sample transform (the prototypes have to be
  smuggled into the same fit), it is not deterministic and the distances between
  the blobs it draws are meaningless. Small but physically real populations (a
  0.3% electron cluster) become indistinguishable from t-SNE artefacts.

* **`proto` (prototype-plane projection) is the proposed alternative.** The head
  assigns a cluster from the cosine similarities z.p_k, so those K numbers are
  *exactly* the information the decision uses. We project the (row-centred)
  similarity vector onto a regular 2D K-gon: prototype k sits at vertex k and an
  event lands at the barycentre of the vertices weighted by how much the model
  prefers each prototype. It is linear, deterministic, out-of-sample exact (the
  prototypes map through the same formula) and for K=3 it is *lossless* -- a
  row-centred 3-vector already lives in a plane, so nothing is thrown away.
  Distances mean something: centre = undecided, vertex = confidently that
  cluster, edge = a two-cluster ambiguity.
"""
from __future__ import annotations

import numpy as np

# name -> axis labels used by the explorer and the eval plots
METHOD_AXES = {
    "pca": ("PC 1", "PC 2"),
    "proto": ("prototype-plane 1", "prototype-plane 2"),
    "tsne": ("t-SNE 1", "t-SNE 2"),
    "umap": ("UMAP 1", "UMAP 2"),
}

METHOD_LABELS = {
    "pca": "PCA (linear)",
    "proto": "Prototype plane",
    "tsne": "t-SNE",
    "umap": "UMAP",
}


def _normalize(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-8)


def pca_2d(z, proto=None):
    """Plain 2-component PCA fitted on z; proto projected with the same map."""
    z = np.asarray(z, dtype=np.float64)
    mu = z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(z - mu, full_matrices=False)
    comp = vt[:2]                                    # (2, d)
    emb = (z - mu) @ comp.T
    pemb = ((np.asarray(proto, dtype=np.float64) - mu) @ comp.T
            if proto is not None and len(proto) else np.zeros((0, 2)))
    return emb.astype(np.float32), pemb.astype(np.float32)


def _polygon(K):
    """K unit vectors on a circle, vertex k at angle 90deg + k*360/K."""
    ang = np.pi / 2 + 2 * np.pi * np.arange(K) / K
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)      # (K, 2)


def proto_plane_2d(z, proto):
    """Project onto the plane spanned by the prototype similarities.

    s_ik = <z_i, p_k/|p_k|>; the row-centred s is mapped onto a regular K-gon so
    prototype k pulls towards vertex k. Lossless for K=3; for K>3 it is the usual
    simplex-to-plane projection: exact along the 2 shown directions, lossy in the
    remaining K-3.
    """
    z = np.asarray(z, dtype=np.float64)
    p = _normalize(np.asarray(proto, dtype=np.float64), axis=1)
    K = p.shape[0]
    V = _polygon(K)

    def _map(x):
        s = _normalize(x, axis=1) @ p.T                      # (N, K) cosines
        return (s - s.mean(axis=1, keepdims=True)) @ V

    emb, pemb = _map(z), _map(p)
    if K < 3:
        # degenerate: the K-gon spans a single axis -> take the leading PCA
        # direction of the residual (z minus its prototype-plane part) for y.
        res = z - (z @ p.T) @ p
        _, _, vt = np.linalg.svd(res - res.mean(0, keepdims=True), full_matrices=False)
        emb[:, 1] = res @ vt[0]
        pemb[:, 1] = (p - (p @ p.T) @ p) @ vt[0]
    scale = float(np.percentile(np.abs(emb), 99)) or 1.0
    return (emb / scale).astype(np.float32), (pemb / scale).astype(np.float32)


def tsne_2d(z, proto=None, perplexity=30, random_state=0):
    """t-SNE, kept only for comparison. No out-of-sample transform, so the
    prototypes are embedded by fitting once on [events; prototypes] stacked."""
    from sklearn.manifold import TSNE

    z = np.asarray(z, dtype=np.float32)
    n_p = 0 if proto is None else len(proto)
    stacked = z if not n_p else np.concatenate(
        [z, _normalize(np.asarray(proto, np.float32), 1)], 0)
    perp = float(min(perplexity, max(5, (stacked.shape[0] - 1) / 3)))
    emb = TSNE(n_components=2, init="pca", perplexity=perp,
               random_state=random_state).fit_transform(stacked)
    split = len(emb) - n_p
    return emb[:split].astype(np.float32), emb[split:].astype(np.float32)


def umap_2d(z, proto=None, random_state=0):
    """UMAP when umap-learn is installed (it is not, in the GATr container).
    Unlike t-SNE it has a real transform(), so prototypes map out-of-sample."""
    import umap

    z = np.asarray(z, dtype=np.float32)
    red = umap.UMAP(n_components=2, metric="cosine", random_state=random_state).fit(z)
    emb = red.embedding_.astype(np.float32)
    pemb = (red.transform(_normalize(np.asarray(proto, np.float32), 1)).astype(np.float32)
            if proto is not None and len(proto) else np.zeros((0, 2), np.float32))
    return emb, pemb


_FUNCS = {"pca": pca_2d, "proto": proto_plane_2d, "tsne": tsne_2d, "umap": umap_2d}


def compute(z, proto, methods=("pca", "proto")):
    """{method: (emb (N,2), proto_emb (K,2))} for every method that works here.

    A method that raises (missing dependency, degenerate input) is skipped with a
    warning instead of killing the whole evaluation.
    """
    out = {}
    for m in methods:
        fn = _FUNCS.get(m)
        if fn is None:
            print(f"[proj] unknown projection '{m}', skipped")
            continue
        try:
            out[m] = fn(z, proto)
            print(f"[proj] {m}: ok")
        except Exception as exc:                             # noqa: BLE001
            print(f"[proj] {m} failed ({type(exc).__name__}: {exc}), skipped")
    if not out:
        out["pca"] = pca_2d(z, proto)
    return out
