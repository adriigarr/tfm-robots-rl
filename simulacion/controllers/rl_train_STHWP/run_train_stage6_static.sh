#!/bin/bash
# Entrenamiento STH-WP stage 6 estático — ciclo completo sin peatones.
#
# Punto de partida: run002_s{42,524}_stage4v2_final
# Salida:           run002_s{42,524}_stage6_final
# World: warehouse_1_static.wbt (sin peatones)
# Steps: 1M por seed  |  LR: 1e-4  |  max_steps_ep: 7000
# Uso: bash run_train_stage6_static.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1_static.wbt"

echo "======================================================="
echo " STH-WP — Entrenamiento STAGE 6 estático"
echo " Ciclo completo: approach → exit → return"
echo " World: warehouse_1_static.wbt (sin peatones)"
echo " Base: stage4v2_final  |  LR=1e-4  |  1M steps"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_train() {
    local KEY="$1"; local LABEL="$2"
    echo ""
    echo "--- $LABEL — $(date '+%Y-%m-%d %H:%M:%S') ---"
    echo "$KEY" > "$STAGE_FILE"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    [ $? -ne 0 ] && echo "[ERROR] $KEY falló" && exit 1
    echo "[OK] $LABEL"
}

run_train "6_s42"  "Stage 6 — seed=42  (1M steps)"
run_train "6_s123" "Stage 6 — seed=123 (1M steps)"
run_train "6_s524" "Stage 6 — seed=524 (1M steps)"

echo ""
echo "======================================================="
echo " PIPELINE COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos guardados:"
echo "   $CONTROLLER_DIR/pruebas/run003_s42_stage6_final.zip"
echo "   $CONTROLLER_DIR/pruebas/run003_s123_stage6_final.zip"
echo "   $CONTROLLER_DIR/pruebas/run003_s524_stage6_final.zip"
echo "======================================================="
