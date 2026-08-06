#!/usr/bin/env bash
# run_labeler.sh — start the manual anchor labeler web app.
#
#     bash install.sh                 # once (creates .venv_viewer)
#     bash run_labeler.sh             # default config + port 8060
#     bash run_labeler.sh --config configs/labeler_other.yml --port 8070
#
# The app runs OUTSIDE the Apptainer/GATr container: it only needs
# flask + h5py + numpy + pyyaml + plotly (plotly is used only to serve
# plotly.min.js to the browser, no CDN required).
#
# Remote machine? Forward the port from your laptop:
#     ssh -L 8060:localhost:8060 <user>@<host>
# then open http://localhost:8060
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${LABELER_VENV:-$HERE/.venv_viewer}"

CONFIG="$HERE/configs/labeler_sdhcal.yml"
PORT=8060
HOST=127.0.0.1
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --port)   PORT="$2";   shift 2 ;;
        --host)   HOST="$2";   shift 2 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

# A leaked PYTHONPATH/PYTHONHOME (key4hep/cvmfs) shadows the venv's numpy.
unset PYTHONPATH PYTHONHOME

if [ ! -d "$VENV" ]; then
    echo "[labeler] venv missing at $VENV -> run: bash install.sh"
    exit 1
fi

exec "$VENV/bin/python" -m anchor_labeler.server \
    --config "$CONFIG" --host "$HOST" --port "$PORT" ${EXTRA[@]+"${EXTRA[@]}"}
