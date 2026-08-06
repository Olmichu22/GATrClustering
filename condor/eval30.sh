#!/bin/bash
set -euo pipefail
# Inference of trained s9/s10 checkpoints on the 30 GeV eval set (Pi_30GeV,
# e+pi labeled by the filters). Runs src.infer_eval per run, dumping a compact
# npz (cluster, particle_type, energy, nhits, anchor, filter_status) for cross-tab.

IMG="/nfs/cms/arqolmo/GPU_train/mlpf/gatr_v9.sif"
REPO="/nfs/cms/arqolmo/SDHCAL_Energy/GATrClustering"
DATA="${DATA:-$REPO/data/eval30/Pi30_eval_clustering.h5}"
OUTDIR="${OUTDIR:-$REPO/results/eval30}"
DEVICE="${DEVICE:-cuda:0}"

export SINGULARITYENV_PYTHONUSERBASE="/nfs/cms/arqolmo/GPU_train/mlpf/extlib"
export SINGULARITYENV_PATH="$SINGULARITYENV_PYTHONUSERBASE/bin:$PATH"
export SINGULARITYENV_PYTHONPATH="$SINGULARITYENV_PYTHONUSERBASE/lib/python3.8/site-packages:$REPO:${PYTHONPATH:-}"
export SINGULARITYENV_WANDB_MODE="disabled"

# run_tag | best-ckpt | stats-file
RUNS=(
  "s9|results/sim_anchors_s9/20260801_210205_s9_dedicated_thrfix/checkpoints/best-epoch029-acc0.9902.ckpt|data/sim_anchors_s9_stats.yml"
  "s9b|results/sim_anchors_s9b/20260802_111209_s9b_dedicated_thrfix_seed123/checkpoints/best-epoch019-acc1.0000.ckpt|data/sim_anchors_s9b_stats.yml"
  "s9c|results/sim_anchors_s9c/20260802_114034_s9c_dedicated_thrfix_seed777/checkpoints/best-epoch029-acc0.9907.ckpt|data/sim_anchors_s9c_stats.yml"
  "s10|results/sim_anchors_s10/20260802_125417_s10_density_seed42/checkpoints/best-epoch024-acc0.9804.ckpt|data/sim_anchors_s10_stats.yml"
  "s10b|results/sim_anchors_s10b/20260802_132631_s10b_density_seed123/checkpoints/best-epoch019-acc1.0000.ckpt|data/sim_anchors_s10b_stats.yml"
  "s10c|results/sim_anchors_s10c/20260802_135427_s10c_density_seed777/checkpoints/best-epoch089-acc0.9907.ckpt|data/sim_anchors_s10c_stats.yml"
)

CMD="cd $REPO; mkdir -p $OUTDIR;"
for r in "${RUNS[@]}"; do
  IFS='|' read -r tag ckpt stats <<< "$r"
  CMD="$CMD echo '===== $tag =====';"
  CMD="$CMD python -m src.infer_eval --ckpt '$REPO/$ckpt' --data_path '$DATA' --stats_path '$REPO/$stats' --out '$OUTDIR/${tag}_30gev.npz' --device $DEVICE || echo '[eval30] $tag FAILED';"
done

apptainer exec --nv \
  -B /nfs/cms/arqolmo/GPU_train/mlpf/extlib -B /nfs:/nfs -B /pnfs:/pnfs --pwd "$REPO" \
  "$IMG" bash -lc "$CMD"
