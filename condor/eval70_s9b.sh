#!/bin/bash
set -euo pipefail
# In-domain sanity check: run s9b inference on the 70 GeV TRAINING file
# (E70GeV_2012.h5, already in the training schema) to compare the nHits-by-cluster
# structure against the 30 GeV out-of-domain eval. Reuses the s9b training stats.

IMG="/nfs/cms/arqolmo/GPU_train/mlpf/gatr_v9.sif"
REPO="/nfs/cms/arqolmo/SDHCAL_Energy/GATrClustering"
DATA="${DATA:-$REPO/data/E70GeV_2012.h5}"
CKPT="${CKPT:-$REPO/results/sim_anchors_s9b/20260802_111209_s9b_dedicated_thrfix_seed123/checkpoints/best-epoch019-acc1.0000.ckpt}"
STATS="${STATS:-$REPO/data/sim_anchors_s9b_stats.yml}"
OUT="${OUT:-$REPO/results/eval70/s9b_70gev.npz}"
DEVICE="${DEVICE:-cuda:0}"

export SINGULARITYENV_PYTHONUSERBASE="/nfs/cms/arqolmo/GPU_train/mlpf/extlib"
export SINGULARITYENV_PATH="$SINGULARITYENV_PYTHONUSERBASE/bin:$PATH"
export SINGULARITYENV_PYTHONPATH="$SINGULARITYENV_PYTHONUSERBASE/lib/python3.8/site-packages:$REPO:${PYTHONPATH:-}"
export SINGULARITYENV_WANDB_MODE="disabled"

CMD="cd $REPO; mkdir -p $(dirname "$OUT"); \
python -m src.infer_eval --ckpt '$CKPT' --data_path '$DATA' --stats_path '$STATS' \
  --out '$OUT' --device $DEVICE"

apptainer exec --nv \
  -B /nfs/cms/arqolmo/GPU_train/mlpf/extlib -B /nfs:/nfs -B /pnfs:/pnfs --pwd "$REPO" \
  "$IMG" bash -lc "$CMD"
