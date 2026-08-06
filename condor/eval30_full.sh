#!/bin/bash
set -euo pipefail
# Standard evaluation (plots + latent_explorer.npz) of the BEST run (s10b,
# density seed123) on the 30 GeV eval set, inside the Apptainer image. Uses the
# dedicated eval config configs/eval30_s10b.yml (anchors stripped, data = 30 GeV,
# scaling reuses the s10b training stats). Outputs to results/eval30/s10b_full/:
#   nhits_by_cluster.png, tsne_z.png, pca_z.png, and latent_explorer.npz.

IMG="/nfs/cms/arqolmo/GPU_train/mlpf/gatr_v9.sif"
REPO="/nfs/cms/arqolmo/SDHCAL_Energy/GATrClustering"
CKPT="${CKPT:-$REPO/results/sim_anchors_s10b/20260802_132631_s10b_density_seed123/checkpoints/best-epoch019-acc1.0000.ckpt}"
CFG="${CFG:-configs/eval30_s10b.yml}"
OUT="${OUT:-results/eval30/s10b_full}"
ANCHORS="${ANCHORS:-configs/anchors_eval30.yml}"
MAXEV="${MAXEV:-6000}"
DEVICE="${DEVICE:-cuda:0}"

export SINGULARITYENV_PYTHONUSERBASE="/nfs/cms/arqolmo/GPU_train/mlpf/extlib"
export SINGULARITYENV_PATH="$SINGULARITYENV_PYTHONUSERBASE/bin:$PATH"
export SINGULARITYENV_PYTHONPATH="$SINGULARITYENV_PYTHONUSERBASE/lib/python3.8/site-packages:$REPO:${PYTHONPATH:-}"
export SINGULARITYENV_WANDB_MODE="disabled"

CMD="cd $REPO; python -m src.evaluate_clustering \
  --ckpt '$CKPT' --cfg '$CFG' --out_dir '$OUT' \
  --anchors '$ANCHORS' --explorer_max_events $MAXEV --device $DEVICE"

apptainer exec --nv \
  -B /nfs/cms/arqolmo/GPU_train/mlpf/extlib -B /nfs:/nfs -B /pnfs:/pnfs --pwd "$REPO" \
  "$IMG" bash -lc "$CMD"
