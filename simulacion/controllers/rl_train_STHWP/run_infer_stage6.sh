#!/bin/bash
# Inferencia STH-WP stage 6 estático — ciclo completo, modelo unificado.
# 100 ep/goal × 28 goals × 3 seeds = 8400 ciclos totales.
# World: warehouse_1_static.wbt (sin peatones).
# Uso: bash run_infer_stage6.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1_static.wbt"

echo "======================================================="
echo " STH-WP — Inferencia STAGE 6 (ciclo completo estático)"
echo " Modelo unificado: approach → exit → return"
echo " 100 ep/goal × 28 goals × 3 seeds"
echo " World: warehouse_1_static.wbt"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_infer() {
    local KEY="$1"; local LABEL="$2"
    echo ""
    echo "--- $LABEL — $(date '+%Y-%m-%d %H:%M:%S') ---"
    echo "$KEY" > "$STAGE_FILE"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    [ $? -ne 0 ] && echo "[ERROR] $KEY falló" && exit 1
    echo "[OK] $LABEL"
}

run_infer "infer_r3_s42_s6"  "Inferencia stage6 — seed=42"
run_infer "infer_r3_s123_s6" "Inferencia stage6 — seed=123"
run_infer "infer_r3_s524_s6" "Inferencia stage6 — seed=524"

echo ""
echo "======================================================="
echo " INFERENCIA COMPLETA — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs en: $CONTROLLER_DIR/inferencia_sthwp/resultados/"
echo "   infer_run003_s42_stage6.csv"
echo "   infer_run003_s123_stage6.csv"
echo "   infer_run003_s524_stage6.csv"
echo "======================================================="
