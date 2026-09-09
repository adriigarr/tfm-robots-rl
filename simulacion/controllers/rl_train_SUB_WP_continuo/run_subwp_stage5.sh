#!/bin/bash
# SUB-WP Stage 5 — Return puro (descarga → espera) — 3 seeds
#
# ~2M steps × 3 seeds  (~2-3h c/u)
# Base: subwp_s*_wp75_stage4_final.zip
# Salidas: pruebas/subwp_s*_wp75_stage5_final.zip
#
# Uso: bash run_subwp_stage5.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"

echo "======================================================="
echo " SUB-WP Stage 5 — Return puro — 3 seeds (2M steps c/u)"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_webots() {
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

run_webots "5_s42"  "Stage 5 — seed=42  (2M steps, return puro)"
run_webots "5_s123" "Stage 5 — seed=123 (2M steps, return puro)"
run_webots "5_s524" "Stage 5 — seed=524 (2M steps, return puro)"

echo ""
echo "======================================================="
echo " Stage 5 completo — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos:"
echo "   pruebas/subwp_s42_wp75_stage5_final.zip"
echo "   pruebas/subwp_s123_wp75_stage5_final.zip"
echo "   pruebas/subwp_s524_wp75_stage5_final.zip"
echo "======================================================="
