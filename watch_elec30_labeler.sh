#!/usr/bin/env bash
# watch_elec30_labeler.sh — arranca el labeler de electrones de 30 GeV en el
# puerto 8051 EN CUANTO el h5 exista y esté completo.
#
#     setsid nohup bash watch_elec30_labeler.sh > logs/labeler_elec30_watch.log 2>&1 &
#
# "Completo" = el tamaño no cambia entre dos comprobaciones separadas 60 s Y
# h5py lo abre y encuentra `offsets`. Sin eso arrancaría sobre un fichero a
# medio escribir. Si el pipeline lo deja con otro nombre, cambiar FILE aquí y
# `dataset.path` en configs/labeler_elec30_filtered.yml.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILE="${FILE:-/nfs/cms/arqolmo/SDHCAL_Energy/data/filtered/Elec30GeV_2012_thrswap_filtered.h5}"
CFG="${CFG:-$HERE/configs/labeler_elec30_filtered.yml}"
PORT="${PORT:-8051}"
unset PYTHONPATH PYTHONHOME
PY="$HERE/.venv_viewer/bin/python"

echo "[watch] esperando $FILE"
prev=-1
while true; do
    if [ -f "$FILE" ]; then
        cur=$(stat -c %s "$FILE" 2>/dev/null || echo 0)
        if [ "$cur" -gt 0 ] && [ "$cur" = "$prev" ]; then
            if "$PY" -c "
import h5py,sys
with h5py.File('$FILE','r') as f:
    assert 'offsets' in f, 'sin offsets'
    print('[watch] ok:', len(f['offsets'])-1, 'eventos')
" ; then
                echo "[watch] arrancando labeler en :$PORT"
                exec bash "$HERE/run_labeler.sh" --config "$CFG" --port "$PORT"
            else
                echo "[watch] el fichero aún no se puede leer, sigo esperando"
            fi
        fi
        prev=$cur
    fi
    sleep 60
done
