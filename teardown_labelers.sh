#!/usr/bin/env bash
# teardown_labelers.sh — cerrar las dos sesiones de etiquetado y dejar los
# anchors en rutas FIJAS que configs/manual_s14.yml ya referencia.
#
#     bash teardown_labelers.sh              # exporta, verifica y mata 8050+8052
#     bash teardown_labelers.sh --dry-run    # exporta y verifica, NO mata nada
#     bash teardown_labelers.sh --also-ebeam # además mata el 8051 (e-beam viejo)
#
# El orden importa y es el motivo de que esto sea un script y no tres curls:
# NO se mata ningún servidor hasta que la verificación pasa. Si algo falla, los
# labelers siguen vivos y se puede reexportar; matarlos primero y descubrir
# después que el h5 no estaba sería irrecuperable (la sesión json sobrevive,
# pero el export es lo que consume el entrenamiento).
#
# Qué escribe, con nombre fijo (sin timestamp) para que el config no cambie:
#   data/manual_s14/E70GeV_2012_filtered_anchors.h5   primario 833k + anchor_label
#   data/manual_s14/Elec70GeV_2012_electron_anchors.h5  solo electrones curados
# Los artefactos con timestamp (json/csv/yml y los subsets por clase, incluido
# el h5 de PIONES) quedan además en sessions/exports/ como copia auditable.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${LABELER_VENV:-$REPO/.venv_viewer}/bin/python"
PRIM_PORT=${PRIM_PORT:-8050}     # configs/labeler_filtered.yml   (833k, pi/mu/multip/penetrante/ruido)
ELEC_PORT=${ELEC_PORT:-8052}     # configs/labeler_elec_filtered.yml (electrones)
EBEAM_PORT=${EBEAM_PORT:-8051}   # e-beam dedicado, sesión vieja

OUT_DIR="$REPO/data/manual_s14"
PRIM_H5="$OUT_DIR/E70GeV_2012_filtered_anchors.h5"
ELEC_H5="$OUT_DIR/Elec70GeV_2012_electron_anchors.h5"

DRY_RUN=0; ALSO_EBEAM=0
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1 ;;
        --also-ebeam) ALSO_EBEAM=1 ;;
        *) echo "opción desconocida: $a" >&2; exit 2 ;;
    esac
done

unset PYTHONPATH PYTHONHOME
mkdir -p "$OUT_DIR"
[ -x "$PY" ] || { echo "[teardown] falta el venv: bash install.sh" >&2; exit 1; }

api() {  # api <port> <path> [json-body]
    local port="$1" path="$2" body="${3:-}"
    if [ -n "$body" ]; then
        curl -sS --max-time 1800 -H 'Content-Type: application/json' \
             -d "$body" "http://127.0.0.1:$port$path"
    else
        curl -sS --max-time 300 "http://127.0.0.1:$port$path"
    fi
}

alive() { curl -sS -o /dev/null --max-time 10 "http://127.0.0.1:$1/api/state"; }

echo "== 1. estado de las sesiones =========================================="
for p in "$PRIM_PORT" "$ELEC_PORT"; do
    alive "$p" || { echo "[teardown] el labeler del puerto $p no responde" >&2; exit 1; }
done
api "$PRIM_PORT" /api/state | "$PY" -c '
import json,sys; s=json.load(sys.stdin).get("summary",{})
print("  8050 primario :", json.dumps(s, ensure_ascii=False))'
api "$ELEC_PORT" /api/state | "$PY" -c '
import json,sys; s=json.load(sys.stdin).get("summary",{})
print("  8052 electrón :", json.dumps(s, ensure_ascii=False))'

echo "== 2. guardar y exportar ============================================="
api "$PRIM_PORT" /api/save > /dev/null || exit 1
api "$ELEC_PORT" /api/save > /dev/null || exit 1

# El primario ES el fichero de entrenamiento -> write_h5 (copia de 566 MB con
# anchor_label relleno; tarda ~1 min). write_subsets además, para tener un h5
# explícito por clase: es donde queda el "h5 de piones" que hay que comprobar.
echo "  exportando primario (566 MB, paciencia)…"
api "$PRIM_PORT" /api/export \
    "{\"write_h5\": true, \"h5_out\": \"$PRIM_H5\", \"write_subsets\": true}" \
    > "$OUT_DIR/.export_primary.json" || exit 1

# El fichero de electrones es FUENTE de anchors -> solo subsets por clase.
echo "  exportando electrones…"
api "$ELEC_PORT" /api/export '{"write_subsets": true}' \
    > "$OUT_DIR/.export_elec.json" || exit 1

for f in "$OUT_DIR/.export_primary.json" "$OUT_DIR/.export_elec.json"; do
    "$PY" - "$f" <<'EOF' || exit 1
import json,sys
d=json.load(open(sys.argv[1]))
if "error" in d:
    print("[teardown] el export falló:", d["error"], file=sys.stderr); sys.exit(1)
print(f"  {sys.argv[1]}: {d['n_anchors']} anchors  {d['per_class']}")
EOF
done

# Ruta fija para el subset de electrones (el export la nombra con timestamp).
"$PY" - "$OUT_DIR/.export_elec.json" "$ELEC_H5" <<'EOF' || exit 1
import json,shutil,sys
d=json.load(open(sys.argv[1])); subs=d.get("subsets") or {}
if "electron" not in subs:
    print("[teardown] el export de electrones no escribió subset 'electron';",
          "clases presentes:", list(subs), file=sys.stderr); sys.exit(1)
shutil.copyfile(subs["electron"], sys.argv[2])
print("  electrón ->", sys.argv[2], "(copia de", subs["electron"] + ")")
EOF

echo "== 3. verificación ==================================================="
"$PY" - "$PRIM_H5" "$ELEC_H5" "$OUT_DIR/.export_primary.json" "$REPO/sessions/filtered_2012.json" <<'EOF'
import collections, json, sys
import h5py, numpy as np

prim_h5, elec_h5, prim_export, session = sys.argv[1:5]
ok = True

# --- lo que dice la sesión (la fuente de verdad) ---
st = json.load(open(session))
kept = collections.Counter(v["label"] for v in st["decisions"].values()
                           if v.get("label") is not None)
names = {0: "electron", 1: "pion", 2: "muon", 3: "multip", 4: "penetrante", 5: "ruido"}
print("  sesión primaria :", {names.get(k, k): v for k, v in sorted(kept.items())})

# --- primario: anchor_label relleno y coincidente ---
with h5py.File(prim_h5, "r") as f:
    if "anchor_label" not in f:
        print("  FALLO: el primario exportado no tiene anchor_label"); ok = False
    else:
        a = f["anchor_label"][:]
        got = collections.Counter(int(v) for v in a[a >= 0])
        print("  primario h5     :", {names.get(k, k): v for k, v in sorted(got.items())},
              f"  ({len(f['offsets'])-1} eventos)")
        for lab in sorted(kept):
            if got.get(lab, 0) != kept[lab]:
                print(f"  FALLO: clase {names.get(lab,lab)}: sesión {kept[lab]} vs h5 {got.get(lab,0)}")
                ok = False
        if got.get(1, 0) == 0:
            print("  FALLO: CERO piones en el primario exportado"); ok = False

# --- el h5 de PIONES por separado (lo que pediste comprobar) ---
subs = (json.load(open(prim_export)).get("subsets") or {})
pion = subs.get("pion")
if not pion:
    print("  FALLO: no se escribió el subset de piones"); ok = False
else:
    with h5py.File(pion, "r") as f:
        n = len(f["offsets"]) - 1
        lab = np.unique(f["anchor_label"][:])
        nh = f["nHits_total"][:] if "nHits_total" in f else np.diff(f["offsets"][:])
        print(f"  h5 de piones    : {pion}")
        print(f"                    {n} eventos, anchor_label={lab.tolist()}, "
              f"nHits p10/50/90 = {np.percentile(nh,[10,50,90]).round(0).tolist()}")
        if n != kept.get(1, -1) or lab.tolist() != [1]:
            print("  FALLO: el subset de piones no cuadra con la sesión"); ok = False

# --- electrones ---
try:
    with h5py.File(elec_h5, "r") as f:
        n = len(f["offsets"]) - 1
        lab = np.unique(f["anchor_label"][:])
        nh = f["nHits_total"][:] if "nHits_total" in f else np.diff(f["offsets"][:])
        print(f"  h5 de electrones: {n} eventos, anchor_label={lab.tolist()}, "
              f"nHits p10/50/90 = {np.percentile(nh,[10,50,90]).round(0).tolist()}")
        if n == 0 or lab.tolist() != [0]:
            print("  FALLO: subset de electrones vacío o mal etiquetado"); ok = False
        if n < 30:
            print(f"  AVISO: solo {n} electrones; s14 los usa como única fuente "
                  "de la clase 0 (s13 usó 113)")
except OSError as exc:
    print("  FALLO: no se pudo leer el h5 de electrones:", exc); ok = False

print("  ->", "TODO OK" if ok else "HAY FALLOS")
sys.exit(0 if ok else 1)
EOF
VERIFY=$?

if [ "$VERIFY" -ne 0 ]; then
    echo "[teardown] verificación FALLIDA: los labelers siguen vivos, nada se ha matado." >&2
    echo "[teardown] revisa arriba, corrige y vuelve a lanzar este script." >&2
    exit 1
fi

echo "== 4. apagado ========================================================"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "  --dry-run: no se mata nada. Los labelers siguen en $PRIM_PORT/$ELEC_PORT."
    exit 0
fi
PORTS=("$PRIM_PORT" "$ELEC_PORT")
[ "$ALSO_EBEAM" -eq 1 ] && PORTS+=("$EBEAM_PORT")
for p in "${PORTS[@]}"; do
    pid=$(ss -ltnp 2>/dev/null | awk -v P=":$p" '$4 ~ P {print $0}' \
          | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
    if [ -n "$pid" ]; then kill "$pid" && echo "  puerto $p: pid $pid terminado"
    else echo "  puerto $p: no había nadie escuchando"; fi
done
[ "$ALSO_EBEAM" -eq 0 ] && echo "  (el 8051 e-beam sigue vivo; --also-ebeam para matarlo también)"

echo
echo "Listo. Siguiente paso:"
echo "  condor_submit condor/manual_s14.sub"
