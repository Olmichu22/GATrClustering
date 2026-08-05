"""Run a trained clustering model over ALL events of an eval h5 and dump a
compact npz (cluster, particle_type, energy, nhits, anchor, filter_status) for
cross-tab analysis. Unlike evaluate_clustering.py this does NOT split train/val
and it keeps the particle_type label per event.

CRITICAL: scaling uses the TRAINING stats file (data seen as at train time), not
recomputed on the eval set. Point cfg.scaling.stats_path at the training stats.

Usage (GPU container):
  python -m src.infer_eval --ckpt CKPT --data_path ADAPTED.h5 \
      --stats_path data/sim_anchors_s10_stats.yml --out OUT.npz [--device cuda:0]
"""
from __future__ import annotations
import argparse, copy, os
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import make_clustering_splits
from src.models.clustering_model import ClusteringModel
from src.evaluate_clustering import (load_ckpt, load_class_names, plot_projection,
                                     export_latent_explorer, explorer_selection,
                                     DEFAULT_PROJECTIONS)
from src import projections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--stats_path", default=None,
                    help="training stats yml; scaling reuses it instead of recomputing on eval set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--explorer_out", default=None,
                    help="also dump a latent_explorer.npz (browse the run with "
                         "src/latent_explorer_demo.py). Off by default.")
    ap.add_argument("--explorer_max_events", type=int, default=5000,
                    help="events kept in the explorer dump / projection plots")
    ap.add_argument("--projections", default=",".join(DEFAULT_PROJECTIONS),
                    help="2D projections of z for the dump and the pngs "
                         "(pca, proto, tsne, umap)")
    ap.add_argument("--anchors", default="configs/anchors.yml",
                    help="anchors.yml giving the cluster -> particle names")
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    state_dict, cfg = load_ckpt(a.ckpt, map_location=device)
    if cfg is None:
        raise SystemExit("checkpoint has no cfg; pass --cfg (not implemented) ")

    cfg = copy.deepcopy(cfg)
    # single-file eval: drop anchor datasets, evaluate on the whole file
    cfg["data"]["path"] = a.data_path
    cfg["data"]["anchor_datasets"] = None
    cfg["data"]["force_anchor"] = None
    cfg["data"]["val_ratio"] = 1.0          # val_idx = everything
    cfg["data"]["filters"] = None
    if a.stats_path:
        cfg["scaling"]["stats_path"] = a.stats_path
    # ensure we reuse existing stats, never recompute on the eval set
    cfg["scaling"]["source"] = "online"

    _, val_ds, base = make_clustering_splits(cfg["data"], cfg["features"], cfg["scaling"])
    val_idx = np.asarray(val_ds.indices)
    loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False)

    model = ClusteringModel(cfg["model"], cfg["features"]).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # z is only needed for the (optional) explorer dump + projection plots. It
    # has to be kept for EVERY event, not just the first few thousand: the
    # explorer picks a stratified subsample afterwards, and the rare clusters it
    # must include (30 electrons in 833k) can sit anywhere in the file.
    # 833k x 64 float32 is ~200 MB, which is affordable.
    want_z = bool(a.explorer_out)
    z_keep = []

    clusters, ptypes, anchors, nhits, energies = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            if want_z:
                z_keep.append(out["z"].detach().cpu().numpy().astype(np.float32))
            clusters.append(out["logits"].argmax(1).cpu().numpy())
            ptypes.append(batch.class_label.cpu().numpy().reshape(-1))
            anchors.append(batch.anchor_label.cpu().numpy().reshape(-1))
            nhits.append(batch.nhits_total.cpu().numpy().reshape(-1))
            energies.append(batch.energy.cpu().numpy().reshape(-1))
    cluster = np.concatenate(clusters)
    ptype = np.concatenate(ptypes)
    anchor = np.concatenate(anchors)
    nhit = np.concatenate(nhits)
    energy = np.concatenate(energies)

    # filter_status aligned via real event ids (val loader is shuffle=False -> val_idx order)
    import h5py
    real_ids = base._event_indices[val_idx]
    fstatus = np.ones(len(cluster), dtype=np.int8)
    with h5py.File(a.data_path, "r") as f:
        if "filter_status" in f:
            fs = f["filter_status"][:]
            fstatus = fs[real_ids].astype(np.int8)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, cluster=cluster.astype(np.int32),
                        particle_type=ptype.astype(np.int32),
                        anchor=anchor.astype(np.int32),
                        nhits=nhit.astype(np.float32),
                        energy=energy.astype(np.float32),
                        filter_status=fstatus)
    print(f"[infer] wrote {a.out}: {len(cluster)} events")

    # ---- optional: 2D projections + latent explorer dump for this full-sample
    # inference (same npz format the eval writes, so latent_explorer_demo.py
    # opens it unchanged).
    if want_z:
        z = np.concatenate(z_keep)
        K = int(cfg["model"]["head"]["num_clusters"])
        class_names = load_class_names(a.anchors, K)
        sel = explorer_selection(cluster, anchor, K, a.explorer_max_events)
        proto = model.head.prototypes.detach().cpu().numpy()
        proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
        methods = tuple(m.strip() for m in a.projections.split(",") if m.strip())
        projs = projections.compute(z[sel], proto, methods or DEFAULT_PROJECTIONS)
        out_dir = os.path.dirname(os.path.abspath(a.explorer_out))
        os.makedirs(out_dir, exist_ok=True)
        for m, (emb, pemb) in projs.items():
            plot_projection(emb, pemb, cluster[sel], anchor[sel], K,
                            os.path.join(out_dir, f"proj_{m}.png"), m, class_names)
        export_latent_explorer(model, base, val_idx, z, cluster, anchor,
                               K, class_names, a.explorer_out,
                               max_events=a.explorer_max_events, projs=projs,
                               sel=sel)
    # quick console cross-tab (status==1, non-anchor)
    m = (fstatus == 1) & (anchor < 0)
    print(f"[infer] status==1 & non-anchor: {int(m.sum())} events")
    for pt, name in [(0, "e"), (1, "pi"), (2, "mu")]:
        sel = m & (ptype == pt)
        if sel.sum():
            u, c = np.unique(cluster[sel], return_counts=True)
            frac = {int(k): round(float(v) / sel.sum(), 3) for k, v in zip(u, c)}
            print(f"  {name} (N={int(sel.sum())}): cluster share {frac}")


if __name__ == "__main__":
    main()
