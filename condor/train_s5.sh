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

# W&B: la clave NUNCA va en el repo. Se toma, por este orden, de $WANDB_API_KEY
# en el entorno de submit o del fichero $WANDB_KEY_FILE (por defecto
# $REPO/.wandb_key, gitignored y con permisos 600). Sin clave -> modo offline,
# que es preferible a que el run falle por login.
KEYFILE="${WANDB_KEY_FILE:-$REPO/.wandb_key}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$KEYFILE" ]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "$KEYFILE")"
fi
export SINGULARITYENV_WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[wandb] sin clave (ni \$WANDB_API_KEY ni $KEYFILE) -> offline" >&2
    export SINGULARITYENV_WANDB_MODE="offline"
else
    export SINGULARITYENV_WANDB_MODE="${WANDB_MODE:-online}"
fi

CFG="${CFG:-configs/sim_anchors_s5.yml}"
DATA="${DATA:-$REPO/data/E70GeV_2012.h5}"
OUT="${OUT:-results/sim_anchors_s5}"
EPOCHS="${EPOCHS:-500}"
DEVICE="${DEVICE:-cuda:0}"

CMD="
cd $REPO
if [ -n \"\${WANDB_API_KEY:-}\" ]; then wandb login \"\$WANDB_API_KEY\" || true; fi
python -m src.train_clustering --cfg $CFG --data_path $DATA --out_dir $OUT --epochs $EPOCHS --wandb_mode \${WANDB_MODE:-online} ${RESUME:+--resume $RESUME}
# Lightning writes checkpoints to \$OUT/<timestamp>/checkpoints/ (unique per run so
# nothing is overwritten). Pick the checkpoints dir of the run we just trained.
CKPT_DIR=\$(ls -dt $OUT/*/checkpoints 2>/dev/null | head -1)
if [ -z \"\$CKPT_DIR\" ]; then echo '[eval] no checkpoints dir found under $OUT' >&2; exit 1; fi
# Prefer the BEST checkpoint (highest val/heldout_anchor_acc) over last.ckpt: the
# model tends to collapse a cluster in late epochs, so last.ckpt can be far worse
# than the best saved by ModelCheckpoint (best-epoch<NNN>-acc<ACC>.ckpt).
EVAL_CKPT=\$(ls \$CKPT_DIR/best-*.ckpt 2>/dev/null | sed -E 's/.*acc([0-9.]+)\\.ckpt\$/\\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z \"\$EVAL_CKPT\" ]; then EVAL_CKPT=\"\$CKPT_DIR/last.ckpt\"; echo '[eval] no best-*.ckpt found, falling back to last.ckpt' >&2; fi
echo \"[eval] using checkpoint \$EVAL_CKPT\"
python -m src.evaluate_clustering --ckpt \"\$EVAL_CKPT\" --data_path $DATA --out_dir \"\$(dirname \"\$CKPT_DIR\")/eval\" --device $DEVICE
"

apptainer exec --nv \
  -B /nfs/cms/arqolmo/GPU_train/mlpf/extlib -B /nfs:/nfs -B /pnfs:/pnfs --pwd "$REPO" \
  "$IMG" bash -lc "$CMD"
