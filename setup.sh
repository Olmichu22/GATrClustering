# setup.sh — SOURCE this file to activate the explorer venv.
#
#     source setup.sh
#
# It only activates the local .venv_viewer created by install.sh (Dash app,
# no torch/GPU). Run `bash install.sh` first if the venv does not exist.

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_VENV="$_HERE/.venv_viewer"

if [ ! -d "$_VENV" ]; then
    echo "[setup] venv missing at $_VENV"
    echo "[setup] run:  bash install.sh"
    return 1 2>/dev/null || exit 1
fi

# Drop any inherited PYTHONPATH/PYTHONHOME (e.g. key4hep/cvmfs) so the venv's
# own numpy/scipy are used instead of a foreign python3.13 site-packages that
# would otherwise shadow them and crash the numpy C-extension import.
unset PYTHONPATH
unset PYTHONHOME

# shellcheck disable=SC1091
source "$_VENV/bin/activate"
echo "[setup] explorer venv active ($(python --version 2>&1))  PYTHONPATH cleared"
echo "[setup] run:  python src/latent_explorer_demo.py --data results/eval/latent_explorer.npz"
