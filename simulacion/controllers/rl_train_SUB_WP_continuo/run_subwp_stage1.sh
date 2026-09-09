#!/bin/bash
# Stage 1 SUB-WP — 3 seeds secuenciales (~45min cada uno)
# Salidas: pruebas/subwp_s{42,123,524}_stage1_final.zip

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"

echo "======================================================="
echo " SUB-WP Stage 1 — 3 seeds"
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

run_webots "1_s42"  "Stage 1 — seed=42  (500k steps)"
run_webots "1_s123" "Stage 1 — seed=123 (500k steps)"
run_webots "1_s524" "Stage 1 — seed=524 (500k steps)"

echo ""
echo "======================================================="
echo " Stage 1 completo — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos:"
echo "   pruebas/subwp_s42_stage1_final.zip"
echo "   pruebas/subwp_s123_stage1_final.zip"
echo "   pruebas/subwp_s524_stage1_final.zip"
echo "======================================================="
