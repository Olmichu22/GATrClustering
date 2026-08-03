#!/usr/bin/env bash
# install_labeler.sh — create/extend the local venv used by the anchor labeler.
#
# The labeler runs OUTSIDE the Apptainer/GATr container: no torch, no GATr, no
# GPU. It reuses the same .venv_viewer as the latent-space explorer, adding
# flask + h5py + pyyaml (plotly is only used to serve plotly.min.js locally, so
# the browser needs no CDN access).
#
#     bash install_labeler.sh
#     bash run_labeler.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${LABELER_VENV:-$HERE/.venv_viewer}"
PYTHON="${PYTHON:-python3}"

# A leaked PYTHONPATH/PYTHONHOME (key4hep/cvmfs) would pull a foreign
# site-packages into this venv and break numpy.
unset PYTHONPATH PYTHONHOME

if [ ! -d "$VENV" ]; then
    echo "[install] creating venv -> $VENV"
    "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install flask h5py numpy pyyaml plotly

echo
echo "[install] done -> $VENV"
echo "[install] start with:  bash run_labeler.sh"
