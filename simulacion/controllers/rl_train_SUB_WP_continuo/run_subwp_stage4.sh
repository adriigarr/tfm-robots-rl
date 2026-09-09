#!/bin/bash
# Stage 4 SUB-WP — 3 seeds secuenciales (~5-6h cada uno)
# Base: subwp_s*_stage3_final.zip
# Salidas: pruebas/subwp_s{42,123,524}_stage4_final.zip

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"

echo "======================================================="
echo " SUB-WP Stage 4 — Ciclo completo — 3 seeds (4M steps c/u)"
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

run_webots "4_s42"  "Stage 4 — seed=42  (4M steps, ciclo completo)"
run_webots "4_s123" "Stage 4 — seed=123 (4M steps, ciclo completo)"
run_webots "4_s524" "Stage 4 — seed=524 (4M steps, ciclo completo)"

echo ""
echo "======================================================="
echo " Stage 4 completo — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos:"
echo "   pruebas/subwp_s42_stage4_final.zip"
echo "   pruebas/subwp_s123_stage4_final.zip"
echo "   pruebas/subwp_s524_stage4_final.zip"
echo "======================================================="
