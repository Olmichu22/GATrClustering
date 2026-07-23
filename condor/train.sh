#!/bin/bash
set -euo pipefail

# Executable for the GATrClustering POC (train + evaluate) inside the Apptainer
# image. Single-GPU by design (the training loop in src/train_clustering.py is a
# plain single-device loop; see README for the multi-GPU note).

IMG="/nfs/cms/arqolmo/GPU_train/mlpf/gatr_v9.sif"
REPO="/nfs/cms/arqolmo/SDHCAL_Energy/GATrClustering"

export SINGULARITYENV_PYTHONUSERBASE="/nfs/cms/arqolmo/GPU_train/mlpf/extlib"
export SINGULARITYENV_PATH="$SINGULARITYENV_PYTHONUSERBASE/bin:$PATH"
export SINGULARITYENV_PYTHONPATH="$SINGULARITYENV_PYTHONUSERBASE/lib/python3.8/site-packages:$REPO:${PYTHONPATH:-}"

# W&B: pass the key from the submitting environment (do NOT hardcode secrets).
# Set it before submitting:  export WANDB_API_KEY=...   (or use WANDB_MODE=offline)
export SINGULARITYENV_WANDB_API_KEY="${WANDB_API_KEY:-}"
export SINGULARITYENV_WANDB_MODE="${WANDB_MODE:-online}"

CFG="${CFG:-configs/poc.yml}"
DATA="${DATA:-$REPO/data/E70GeV_2016.h5}"
OUT="${OUT:-results/poc_run1}"
EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-cuda:0}"

CMD="
cd $REPO
if [ -n \"\${WANDB_API_KEY:-}\" ]; then wandb login \"\$WANDB_API_KEY\" || true; fi
python -m src.train_clustering --cfg $CFG --data_path $DATA --out_dir $OUT --epochs $EPOCHS --device $DEVICE --wandb_mode \${WANDB_MODE:-online}
python -m src.evaluate_clustering --ckpt $OUT/last.ckpt --data_path $DATA --out_dir $OUT/eval --device $DEVICE
"

apptainer exec --nv \
  -B /nfs/cms/arqolmo/GPU_train/mlpf/extlib -B /nfs:/nfs --pwd "$REPO" \
  "$IMG" bash -lc "$CMD"
