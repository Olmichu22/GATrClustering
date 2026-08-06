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
    4. 2D projections of z (PCA + prototype plane; t-SNE opt-in via
       --projections) colored by assigned cluster, anchors highlighted.
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

from .data.dataset import DEFAULT_NUM_WORKERS, make_clustering_splits  # noqa: E402
from .models.clustering_model import ClusteringModel  # noqa: E402
from .plots import plot_latent_pca, cluster_colors, cluster_label  # noqa: E402
from . import projections  # noqa: E402

DEFAULT_PROJECTIONS = ("pca", "proto")


def load_class_names(anchors_path, K):
    """Map cluster index -> particle name from the anchors.yml used for marking.

    Cluster k corresponds to anchor_label k (prototypes are initialized per anchor
    label), so anchors.yml's ``classes: {k: is_<particle>}`` gives the cluster ->
    particle mapping. Falls back to ``cluster k`` for any missing entry.
    """
    names = [None] * K
    if anchors_path and os.path.exists(anchors_path):
        with open(anchors_path) as fh:
            acfg = yaml.safe_load(fh) or {}
        for k, flag in (acfg.get("classes") or {}).items():
            try:
                k = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= k < K:
                nm = str(flag)
                names[k] = nm[3:] if nm.startswith("is_") else nm
    return names


def _torch_load(ckpt_path, map_location):
    """torch.load with weights_only compat across torch versions.

    Lightning 2.x checkpoints contain non-tensor objects (cfg, hparams), so we
    must load with weights_only=False. Older torch has no such kwarg.
    """
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
        # torch too old to know the weights_only kwarg
        return torch.load(ckpt_path, map_location=map_location)


def load_ckpt(ckpt_path, map_location="cpu"):
    """Load a checkpoint and return ``(state_dict, cfg)`` ready for
    ``ClusteringModel(cfg["model"], cfg["features"]).load_state_dict(state_dict)``.

    Supports both formats transparently:
      * Lightning (new): weights under ``ckpt["state_dict"]`` with a ``model.``
        prefix (LightningModule wraps the model as ``self.model``); cfg under
        ``ckpt["hyper_parameters"]`` (from ``save_hyperparameters(cfg)``).
      * Manual loop (old): weights under ``ckpt["model"]``; cfg under ``ckpt["cfg"]``.
    """
    ckpt = _torch_load(ckpt_path, map_location)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # ---- Lightning format ----
        raw = ckpt["state_dict"]
        prefix = "model."
        state_dict = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in raw.items()
        }
        cfg = ckpt.get("hyper_parameters")
        # save_hyperparameters(cfg) -> hyper_parameters IS the cfg. But if it was
        # wrapped (e.g. save_hyperparameters(cfg=cfg)) unwrap the "cfg" sub-key.
        if isinstance(cfg, dict) and "cfg" in cfg and "model" not in cfg:
            cfg = cfg["cfg"]
        return state_dict, cfg

    if isinstance(ckpt, dict) and "model" in ckpt:
        # ---- old manual-loop format ----
        return ckpt["model"], ckpt.get("cfg")

    raise KeyError(
        f"Unrecognized checkpoint format: top-level keys = "
        f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
    )


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


def cluster_occupancy(cluster, K, class_names=None):
    occ = np.bincount(cluster, minlength=K)
    frac = occ / max(occ.sum(), 1)
    print(f"[eval] cluster occupancy: {occ.tolist()}  (fractions {np.round(frac,3).tolist()})")
    for k in range(K):
        print(f"[eval]   {cluster_label(k, class_names)}: {int(occ[k])} events ({frac[k]:.3f})")


def plot_nhits_by_cluster(nhits, cluster, K, out_path, class_names=None, clip_pct=99.5):
    """nHits per assigned cluster, with the axis on the bulk of the distribution.

    A handful of 5000-hit events used to set the range for all 40 bins, so each
    bin was ~130 hits wide and the entire population piled into the first few
    while most of the axis showed single-event tails. The upper limit is now the
    LARGEST per-cluster ``clip_pct`` percentile: taking it per cluster (rather
    than one global percentile) keeps every cluster's own bulk on screen even
    when one of them lives at much higher occupancy. The clipped events are
    counted and reported in the caption, never silently dropped.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = cluster_colors(K)

    hi = 0.0
    for k in range(K):
        m = cluster == k
        if m.any():
            hi = max(hi, float(np.percentile(nhits[m], clip_pct)))
    lo = float(np.min(nhits)) if nhits.size else 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = float(np.max(nhits)) if nhits.size else 1.0
    n_out = int((nhits > hi).sum())

    bins = np.histogram_bin_edges(nhits[nhits <= hi], bins=40, range=(lo, hi))
    for k in range(K):
        m = cluster == k
        if m.any():
            ax.hist(nhits[m], bins=bins, histtype="step", linewidth=2,
                    color=colors[k], label=cluster_label(k, class_names))
    ax.set_yscale("log")   # muon cluster dwarfs e/pi in linear scale -> log by default
    ax.set_xlim(lo, hi)
    ax.set_xlabel("nHits (total)")
    ax.set_ylabel("events")
    title = "nHits distribution by assigned cluster (PRIMARY diagnostic)"
    if n_out:
        title += (f"\naxis clipped at p{clip_pct:g} per cluster = {hi:.0f} hits; "
                  f"{n_out} event{'s' if n_out != 1 else ''} above, not shown")
    ax.set_title(title, fontsize=10)
    ax.legend(title="cluster · particle")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[eval] wrote {out_path}")


def plot_projection(emb, proto_emb, cluster, anchor, K, out_path, method,
                    class_names=None):
    """One 2D projection of z (see src/projections.py) as a png.

    Same visual grammar as the training PCA plot: color = assigned cluster,
    black-edged markers = anchors (colored by their TRUE class), stars =
    prototypes.
    """
    xlab, ylab = projections.METHOD_AXES.get(method, ("dim 1", "dim 2"))
    colors = cluster_colors(K)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    # events per cluster -> legend names the particle
    for k in range(K):
        m = cluster == k
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1], color=colors[k], s=6, alpha=0.4,
                       linewidths=0, label=cluster_label(k, class_names))
    # anchors highlighted by their true class (same color, black edge)
    am = anchor >= 0
    for k in range(K):
        mk = am & (anchor == k)
        if mk.any():
            ax.scatter(emb[mk, 0], emb[mk, 1], color=colors[k], s=70,
                       edgecolors="black", linewidths=0.8)
    if proto_emb is not None and len(proto_emb):
        ax.scatter(proto_emb[:, 0], proto_emb[:, 1],
                   color=[colors[k] for k in range(len(proto_emb))],
                   s=260, marker="*", edgecolors="black", linewidths=1.0, zorder=5)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.legend(fontsize=8, title="cluster · particle")
    ax.set_title(f"{projections.METHOD_LABELS.get(method, method)} of z "
                 "(color = assigned cluster; edged = anchors; star = prototype)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[eval] wrote {out_path}")


def _gather_raw_hits(base_ds, val_idx, sel, max_hits):
    """Raw (unscaled) per-hit xyz + thr for the selected val events.

    ``base_ds.reader.hit`` keeps the ORIGINAL arrays: scaling in
    ``apply_scaling_inplace`` assigns fresh arrays to ``base_ds.hit`` and never
    touches ``base_ds.reader.hit``. So we read physically-meaningful coordinates
    here regardless of the z_norm/minmax applied to the model inputs.

    Returns (hits_xyz (S, max_hits, 3), hits_thr (S, max_hits), hits_len (S,)).
    """
    reader = base_ds.reader
    offsets = base_ds.offsets
    # logical names for the 3 point coords, in the order the model uses them
    px, py, pz = base_ds.features_cfg["mv_point"]
    xr, yr, zr = reader.hit[px], reader.hit[py], reader.hit[pz]
    thr = reader.hit.get("thr")

    S = len(sel)
    hits_xyz = np.zeros((S, max_hits, 3), dtype=np.float32)
    hits_thr = np.zeros((S, max_hits), dtype=np.float32)
    hits_len = np.zeros(S, dtype=np.int32)
    for i, di in enumerate(sel):
        real = int(base_ds._event_indices[int(val_idx[di])])
        s, e = int(offsets[real]), int(offsets[real + 1])
        n = min(e - s, max_hits)
        hits_xyz[i, :n, 0] = xr[s:s + n]
        hits_xyz[i, :n, 1] = yr[s:s + n]
        hits_xyz[i, :n, 2] = zr[s:s + n]
        if thr is not None:
            hits_thr[i, :n] = thr[s:s + n]
        hits_len[i] = n
    return hits_xyz, hits_thr, hits_len


def explorer_selection(cluster, anchor, K, max_events, seed=0):
    """Which events the explorer/pngs should show: a STRATIFIED sample.

    Taking the first (or a uniform random) N events hides exactly what we look
    at these plots for. In the s14a full-sample inference the electron cluster
    holds 30 of 833 545 events, so a 6000-event uniform dump contains none of
    them and the view says nothing about where the electrons went.

    Rule: every anchor first, then every event of any cluster smaller than the
    per-cluster cap (max_events // K) -- rare clusters come in whole -- and the
    remaining budget split over the big clusters in proportion to their size, so
    the majority keeps its relative density. Deterministic given ``seed``.

    NOTE: the resulting point density is therefore NOT the cluster prior; read
    occupancy off assignments_full.npz / the printed cross-tab instead.
    """
    rng = np.random.default_rng(seed)
    n = len(cluster)
    if n <= max_events:
        return np.arange(n)

    keep = [np.flatnonzero(anchor >= 0)[:max_events]]
    taken = set(keep[0].tolist())
    budget = max_events - len(keep[0])
    cap = max(1, budget // max(1, K))

    pools = {k: np.array([i for i in np.flatnonzero(cluster == k) if i not in taken])
             for k in range(K)}
    big = []
    for k in range(K):
        pool = pools[k]
        if len(pool) <= cap:                       # rare cluster -> take it whole
            keep.append(pool)
            budget -= len(pool)
        else:
            big.append(k)
    total_big = sum(len(pools[k]) for k in big) or 1
    for k in big:
        share = int(round(budget * len(pools[k]) / total_big))
        if share > 0:
            keep.append(rng.choice(pools[k], size=min(share, len(pools[k])),
                                   replace=False))
    sel = np.unique(np.concatenate([k for k in keep if len(k)]))
    if len(sel) > max_events:
        sel = np.sort(rng.choice(sel, size=max_events, replace=False))
    counts = {int(k): int((cluster[sel] == k).sum()) for k in range(K)}
    print(f"[eval] explorer selection: {len(sel)} events, per-cluster {counts}, "
          f"{int((anchor[sel] >= 0).sum())} anchors (stratified, not the prior)")
    return sel


def export_latent_explorer(model, base_ds, val_idx, z, cluster, anchor,
                           K, class_names, out_path, max_events=3000,
                           projs=None, methods=DEFAULT_PROJECTIONS, sel=None):
    """Dump everything the (torch-free) latent_explorer_demo.py needs into an npz.

    Latent geometry (one or more 2D projections of z + the co-embedded
    prototypes) plus, for each plotted event, its RAW hits (for the 3D shower
    panel), energy, nHits and anchor flag. Event order matches the val loader
    (shuffle=False -> ``val_idx`` order), so ``z[i]`` and the i-th gathered event
    refer to the same shower.

    Every projection is stored under ``emb2d_<method>`` / ``proto2d_<method>``
    and listed in ``emb_methods``; the explorer offers them as a toggle. The
    first one is also written to the legacy ``emb_2d`` / ``proto_2d`` keys.
    ``projs`` may carry embeddings already computed by the caller (so the pngs
    and the explorer show the exact same map).
    """
    if sel is None:
        sel = explorer_selection(cluster, anchor, K, max_events)
    sel = np.asarray(sel)
    zc, clc, anc = z[sel], cluster[sel], anchor[sel]

    proto = model.head.prototypes.detach().cpu().numpy()
    proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    if projs is None:
        projs = projections.compute(zc, proto, methods)
    proj_arrays = {}
    for m, (emb, pemb) in projs.items():
        proj_arrays[f"emb2d_{m}"] = np.asarray(emb, np.float32)[:len(sel)]
        proj_arrays[f"proto2d_{m}"] = np.asarray(pemb, np.float32)
    first = next(iter(projs))
    emb_2d = proj_arrays[f"emb2d_{first}"]
    proto_2d = proj_arrays[f"proto2d_{first}"]

    # Raw per-event fields (unscaled) aligned to sel.
    reader = base_ds.reader
    real_ids = np.array([int(base_ds._event_indices[int(val_idx[i])]) for i in sel])
    nhits = reader.nhits_per_event()[real_ids].astype(np.int32)
    e_arr = reader.event.get("energy")
    energies = (np.asarray(e_arr)[real_ids].astype(np.float32)
                if e_arr is not None else np.zeros(len(sel), np.float32))

    max_hits = int(nhits.max()) if len(nhits) else 0
    hits_xyz, hits_thr, hits_len = _gather_raw_hits(base_ds, val_idx, sel, max_hits)

    names = np.array([class_names[k] or f"cluster {k}" for k in range(K)])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        **proj_arrays,
        emb_2d=emb_2d,
        emb_method=np.array([first]),
        emb_methods=np.array(list(projs.keys())),
        proto_2d=proto_2d,
        cluster=clc.astype(np.int32),
        anchor=anc.astype(np.int32),
        is_anchor=(anc >= 0),
        energies=energies,
        n_hits=nhits,
        hits_xyz=hits_xyz,
        hits_thr=hits_thr,
        hits_len=hits_len,
        class_names=names,
        K=np.array([K], dtype=np.int32),
        sel_idx=sel.astype(np.int64),   # position in the val/inference order
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[eval] wrote {out_path}  ({size_mb:.1f} MB, {len(sel)} events, "
          f"projections: {', '.join(projs)})")


def evaluate(cfg, ckpt_path, out_dir, device_str, anchors_path=None,
             explorer_out=None, explorer_max_events=3000, nhits_clip_pct=99.5,
             methods=DEFAULT_PROJECTIONS):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    _, val_ds, _ = make_clustering_splits(cfg["data"], cfg["features"], cfg["scaling"])
    # Hasta ahora sin workers: la evaluación cargaba en el proceso principal,
    # con la GPU esperando a que un solo hilo hiciera el collate de eventos de
    # longitud variable. Toma el mismo `train.num_workers` que el entrenamiento.
    nw = int(cfg["train"].get("num_workers", DEFAULT_NUM_WORKERS))
    loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=nw, pin_memory=True, persistent_workers=nw > 0)

    model = ClusteringModel(cfg["model"], cfg["features"]).to(device)
    state_dict, _ = load_ckpt(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    z, cluster, anchor, nhits = collect(model, loader, device)
    K = cfg["model"]["head"]["num_clusters"]

    # Per-event assignments for EVERY evaluated event. The plots subsample
    # (t-SNE 4000, explorer dump 3000-5000) and the printed occupancy is lost if
    # the batch system does not capture stdout, so without this there is no way
    # to answer questions about a specific region of the distribution -- a band
    # holding 0.3% of the sample lands in single digits in any subsample.
    np.savez_compressed(os.path.join(out_dir, "assignments.npz"),
                        cluster=cluster.astype(np.int16),
                        anchor=anchor.astype(np.int16),
                        nhits=nhits.astype(np.int32),
                        K=np.array([K], dtype=np.int32))
    print(f"[eval] wrote {os.path.join(out_dir, 'assignments.npz')} "
          f"({len(cluster)} events)")
    class_names = load_class_names(anchors_path, K)
    print(f"[eval] cluster -> particle map: "
          f"{ {k: (class_names[k] or f'cluster {k}') for k in range(K)} }")

    # ---- anchor-based (held-out) ----
    heldout_anchor_accuracy(cluster, anchor)
    # ---- unsupervised ----
    silhouette(z, cluster)
    cluster_occupancy(cluster, K, class_names)
    plot_nhits_by_cluster(nhits, cluster, K, os.path.join(out_dir, "nhits_by_cluster.png"),
                          class_names, clip_pct=nhits_clip_pct)
    proto = model.head.prototypes.detach().cpu().numpy()
    fig = plot_latent_pca(z, cluster, anchor, proto, -1,
                          os.path.join(out_dir, "pca_z.png"), class_names=class_names)
    plt.close(fig)

    # ---- 2D projections: one png each, and the same arrays reused for the
    # explorer dump so the png and the interactive view are the SAME map. They
    # run on a STRATIFIED subsample (rare clusters kept whole) -- t-SNE would not
    # scale past a few thousand points anyway.
    sel = explorer_selection(cluster, anchor, K, explorer_max_events)
    protoN = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    projs = projections.compute(z[sel], protoN, methods)
    for m, (emb, pemb) in projs.items():
        plot_projection(emb, pemb, cluster[sel], anchor[sel], K,
                        os.path.join(out_dir, f"proj_{m}.png"), m, class_names)

    # ---- latent-space explorer dump (loaded by latent_explorer_demo.py) ----
    if explorer_out:
        # val_ds is a Subset: recover the base dataset and the val indices to
        # align each z[i] with the RAW hits of the same event.
        base_ds = val_ds.dataset
        val_idx = np.asarray(val_ds.indices)
        export_latent_explorer(model, base_ds, val_idx, z, cluster, anchor,
                               K, class_names, explorer_out, explorer_max_events,
                               projs=projs, sel=sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default=None, help="defaults to cfg stored in the checkpoint")
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--out_dir", default="results/eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--anchors", default="configs/anchors.yml",
                    help="anchors.yml used for marking; gives the cluster -> particle names")
    ap.add_argument("--explorer_out", default=None,
                    help="path for the latent_explorer.npz dump (loaded by "
                         "latent_explorer_demo.py); default '<out_dir>/latent_explorer.npz'. "
                         "Pass 'none' to skip it.")
    ap.add_argument("--explorer_max_events", type=int, default=3000,
                    help="cap on events written to the explorer dump")
    ap.add_argument("--projections", default=",".join(DEFAULT_PROJECTIONS),
                    help="comma-separated 2D projections of z to compute "
                         "(pca, proto, tsne, umap). They all end up in the "
                         "explorer npz as a toggle. t-SNE is OFF by default: it "
                         "distorts these hypersphere latents (see src/projections.py)")
    ap.add_argument("--nhits_clip_pct", type=float, default=99.5,
                    help="nHits histogram: clip the x axis at this per-cluster "
                         "percentile so single-event tails do not set the range "
                         "(100 = show everything)")
    args = ap.parse_args()

    if args.cfg:
        with open(args.cfg) as fh:
            cfg = yaml.safe_load(fh)
    else:
        _, cfg = load_ckpt(args.ckpt, map_location="cpu")
    if args.data_path:
        cfg["data"]["path"] = args.data_path

    explorer_out = args.explorer_out
    if explorer_out is None:
        explorer_out = os.path.join(args.out_dir, "latent_explorer.npz")
    elif explorer_out.lower() == "none":
        explorer_out = None

    methods = tuple(m.strip() for m in args.projections.split(",") if m.strip())
    evaluate(cfg, args.ckpt, args.out_dir, args.device, anchors_path=args.anchors,
             explorer_out=explorer_out, explorer_max_events=args.explorer_max_events,
             nhits_clip_pct=args.nhits_clip_pct, methods=methods or DEFAULT_PROJECTIONS)


if __name__ == "__main__":
    main()
