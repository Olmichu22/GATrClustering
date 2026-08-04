# GATrClustering

Unsupervised, anchor-guided clustering of SDHCAL shower events (e/π/μ) on graphs
of hits, with a **GATr** backbone. Forked from `../GATrAutoencoder`.

The whole pipeline is **modular and parametric** (see `configs/poc.yml`): feature
routing (multivector point / geometric scalar / GATr scalar input), per-feature
scaling, and switchable aggregation — so switching detector/conditions means
editing the YAML, not the code.

## Layout

```
configs/poc.yml                 experiment config (data + features + scaling + model + loss + train)
src/data/flat_h5_reader.py      fixed CSR contract + remappable field_map
src/data/scaling.py             per-feature z_norm/minmax/log/none; online/file/dataset stats
src/data/dataset.py             PyG dataset + make_clustering_splits (train-only stats)
src/data/feature_routing.py     point / geometric-scalar / GATr-scalar routing; thr ordinal|one_hot
src/models/gatr_module.py       forked GATr encoder (generalized geometric-scalar embedding)
src/models/attention_pooling.py forked PMA pooling
src/models/aggregation.py       mean | attention | token (asymmetric aggregation token)
src/models/prototype_head.py    projection to z + cosine logits vs learnable prototypes
src/models/clustering_model.py  encoder -> aggregation -> prototype head
src/losses/swap_loss.py         SwAV-style swap loss (no Sinkhorn)
src/losses/vicreg.py            variance + covariance regularization
src/augment.py                  per-hit dropout (two views)
src/train_clustering.py         training + anchor-based prototype init
src/evaluate_clustering.py      nHits-by-cluster (primary), t-SNE, confusion
```

## Data contract

Flat CSR HDF5 (or npz): an `offsets` array (`len n_events+1`), per-hit arrays
(`len n_hits`) and per-event arrays (`len n_events`). Branch names are remapped
via `data.field_map` in the config. Anchors are a per-event field
(`anchor_label`: `-1` = no anchor, `0..K-1` = class) that the user marks.

## Run (inside the Apptainer image)

```bash
apptainer exec --nv -B /nfs/cms/arqolmo/GPU_train/mlpf/extlib \
  -B /nfs:/nfs --pwd "$PWD" /nfs/cms/arqolmo/GPU_train/mlpf/gatr_v9.sif bash

# train  (W&B on by default; logs losses, lr, held-out anchor acc, latent PCA)
export WANDB_API_KEY=...        # or:  --wandb_mode offline  /  --no_wandb
python -m src.train_clustering --cfg configs/poc.yml --data_path /path/to/dataset.h5

# evaluate
python -m src.evaluate_clustering --ckpt results/poc_run1/last.ckpt \
  --data_path /path/to/dataset.h5 --out_dir results/eval
```
