#!/bin/bash
# Stage 3 SUB-WP — 3 seeds secuenciales (~5-6h cada uno)
# Base: subwp_s*_stage2_final.zip
# Headings: inferencia_subwp/resultados/arrival_headings_stage2.json
# Salidas: pruebas/subwp_s{42,123,524}_stage3_final.zip

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"
HEADINGS="$CONTROLLER_DIR/inferencia_subwp/resultados/arrival_headings_stage2.json"

if [ ! -f "$HEADINGS" ]; then
    echo "[ERROR] arrival_headings_stage2.json no encontrado. Ejecuta primero run_subwp_infer_s1_s2.sh."
    exit 1
fi

echo "======================================================="
echo " SUB-WP Stage 3 — Exit puro — 3 seeds (4M steps c/u)"
echo " Headings: $HEADINGS"
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

run_webots "3_s42"  "Stage 3 — seed=42  (4M steps, exit puro)"
run_webots "3_s123" "Stage 3 — seed=123 (4M steps, exit puro)"
run_webots "3_s524" "Stage 3 — seed=524 (4M steps, exit puro)"

echo ""
echo "======================================================="
echo " Stage 3 completo — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos:"
echo "   pruebas/subwp_s42_stage3_final.zip"
echo "   pruebas/subwp_s123_stage3_final.zip"
echo "   pruebas/subwp_s524_stage3_final.zip"
echo "======================================================="
