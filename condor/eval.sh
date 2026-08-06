#!/bin/bash
set -euo pipefail

# Evaluation-only executable (condor/train.sh trains AND evaluates; this runs
# just the eval, on a checkpoint that already exists and on any dataset).
#
# Env:
#   CKPT      checkpoint to load                (required)
#   CFG       config to use                     (default: the cfg in the ckpt)
#   DATA      dataset h5; overrides cfg.data.path
#   OUT       output directory for the plots + latent_explorer.npz
#   DEVICE    cuda:0
#   EXPLORER_MAX  events written to latent_explorer.npz (default 3000)

IMG="/nfs/cms/arqolmo/GPU_train/mlpf/gatr_v9.sif"
REPO="/nfs/cms/arqolmo/SDHCAL_Energy/GATrClustering"

export SINGULARITYENV_PYTHONUSERBASE="/nfs/cms/arqolmo/GPU_train/mlpf/extlib"
export SINGULARITYENV_PATH="$SINGULARITYENV_PYTHONUSERBASE/bin:$PATH"
export SINGULARITYENV_PYTHONPATH="$SINGULARITYENV_PYTHONUSERBASE/lib/python3.8/site-packages:$REPO:${PYTHONPATH:-}"

CKPT="${CKPT:?CKPT is required}"
OUT="${OUT:-results/eval}"
DEVICE="${DEVICE:-cuda:0}"
EXPLORER_MAX="${EXPLORER_MAX:-3000}"

CMD="
cd $REPO
python -m src.evaluate_clustering \
  --ckpt '$CKPT' \
  ${CFG:+--cfg $CFG} \
  ${DATA:+--data_path $DATA} \
  --out_dir '$OUT' \
  --device $DEVICE \
  --explorer_max_events $EXPLORER_MAX
"

apptainer exec --nv \
  -B /nfs/cms/arqolmo/GPU_train/mlpf/extlib -B /nfs:/nfs -B /pnfs:/pnfs --pwd "$REPO" \
  "$IMG" bash -lc "$CMD"
