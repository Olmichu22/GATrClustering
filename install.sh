#!/usr/bin/env bash
# install.sh — create the LOCAL virtualenv for the latent-space explorer.
#
# The explorer (src/latent_explorer_demo.py) is a Dash app that runs OUTSIDE the
# Apptainer/GATr container: it needs NO PyTorch, GATr nor GPU. This venv is only
# for viewing the latent_explorer.npz produced by evaluate_clustering.py.
#
# Usage:
#     bash install.sh          # one-off
#     source setup.sh          # activate it in your shell
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv_viewer"
PYTHON="${PYTHON:-python3}"

# A leaked PYTHONPATH/PYTHONHOME (key4hep/cvmfs) would pull a foreign python3.13
# site-packages into this venv and break numpy. Clear it before creating/using.
unset PYTHONPATH
unset PYTHONHOME

if [ ! -d "$VENV" ]; then
    echo "[install] creating venv -> $VENV"
    "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install dash plotly scipy numpy

echo
echo "[install] done."
echo "[install] activate with:  source setup.sh"
echo "[install] then run:        python src/latent_explorer_demo.py --data results/eval/latent_explorer.npz"
