#!/bin/bash
# Inferencia stage 4 v2 — 3 seeds — timeout=4000 steps
#
# Reutiliza los modelos ya entrenados: run002_s*_stage4v2_final
# Sobreescribe los CSVs anteriores con los nuevos resultados.
#
# Uso: bash run_infer_stage4v2_seeds.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"

echo "======================================================="
echo " Inferencia stage 4 v2 — 3 seeds (timeout=4000)"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_infer() {
    local KEY="$1"
    local LABEL="$2"
    echo ""
    echo "------- $LABEL — $(date '+%H:%M:%S') -------"
    echo "$KEY" > "$STAGE_FILE"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    local CODE=$?
    if [ $CODE -ne 0 ]; then
        echo "[ERROR] Webots terminó con código $CODE en $KEY. Abortando."
        exit 1
    fi
    echo "[OK] $LABEL — $(date '+%H:%M:%S')"
}

run_infer "infer_r2_s42_s4v2"  "Inferencia s42  (2800 ep, timeout=4000)"
run_infer "infer_r2_s123_s4v2" "Inferencia s123 (2800 ep, timeout=4000)"
run_infer "infer_r2_s524_s4v2" "Inferencia s524 (2800 ep, timeout=4000)"

echo ""
echo "======================================================="
echo " Inferencia completa — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs en inferencia_sthwp/resultados/"
echo "======================================================="
