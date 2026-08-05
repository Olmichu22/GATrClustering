#!/bin/bash
set -euo pipefail

# Full-sample inference executable: runs a trained checkpoint over EVERY event of
# a dataset (no train/val split, no quality filter) and writes
#   $OUT/assignments_full.npz   cluster + particle_type + anchor + nhits + energy
#   $OUT/proj_<method>.png      2D projections of z (PCA + prototype plane)
#   $OUT/latent_explorer.npz    dump for src/latent_explorer_demo.py
#
# Unlike condor/eval.sh (which re-splits and evaluates the val fraction only)
# this predicts the whole training file.
#
# Env:
#   CKPT      checkpoint to load                        (required)
#   DATA      h5 to predict on                          (required)
#   OUT       output directory                          (required)
#   STATS     training stats yml (scaling reuse)        (default: the one in cfg)
#   DEVICE    cuda:0
#   EXPLORER_MAX  events kept for the explorer/pngs     (default 5000)
#   PROJECTIONS   comma list: pca,proto[,tsne,umap]     (default pca,proto)
#   ANCHORS   anchors yml giving the cluster -> particle names for the legends
#
#   REPO      checkout to run from (a git worktree is fine)
#
# REPO must be passed explicitly when running under Condor: the schedd SPOOLS the
# executable, so at run time this script lives in /var/lib/condor/spool/... and
# its own location says nothing about where the code is. Falling back to the
# script's directory is only for running it by hand.

IMG="/nfs/cms/arqolmo/GPU_train/mlpf/gatr_v9.sif"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -d "$REPO/src" ]; then
    echo "[infer] REPO=$REPO has no src/ -- pass REPO=<checkout> in the job environment" >&2
    exit 2
fi

export SINGULARITYENV_PYTHONUSERBASE="/nfs/cms/arqolmo/GPU_train/mlpf/extlib"
export SINGULARITYENV_PATH="$SINGULARITYENV_PYTHONUSERBASE/bin:$PATH"
export SINGULARITYENV_PYTHONPATH="$SINGULARITYENV_PYTHONUSERBASE/lib/python3.8/site-packages:$REPO:${PYTHONPATH:-}"

CKPT="${CKPT:?CKPT is required}"
DATA="${DATA:?DATA is required}"
OUT="${OUT:?OUT is required}"
DEVICE="${DEVICE:-cuda:0}"
EXPLORER_MAX="${EXPLORER_MAX:-5000}"
PROJECTIONS="${PROJECTIONS:-pca,proto}"

CMD="
cd $REPO
mkdir -p '$OUT'
python -m src.infer_eval \
  --ckpt '$CKPT' \
  --data_path '$DATA' \
  ${STATS:+--stats_path $STATS} \
  --out '$OUT/assignments_full.npz' \
  --explorer_out '$OUT/latent_explorer.npz' \
  --explorer_max_events $EXPLORER_MAX \
  --projections '$PROJECTIONS' \
  ${ANCHORS:+--anchors $ANCHORS} \
  --device $DEVICE
"

# Condor's stdout/stderr come back through the spool, which is not readable from
# the submit host here, so keep our own copy next to the results.
mkdir -p "$OUT"
apptainer exec --nv \
  -B /nfs/cms/arqolmo/GPU_train/mlpf/extlib -B /nfs:/nfs -B /pnfs:/pnfs --pwd "$REPO" \
  "$IMG" bash -lc "$CMD" 2>&1 | tee "$OUT/infer.log"
