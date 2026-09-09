#!/bin/bash
# Inferencia con obstáculos dinámicos (peatones) — STH-WP y SUB-WP.
#
# Lanza 4 runs secuenciales:
#   1. STH-WP seed=42   (100 ep × 28 goals = 2800 ciclos)
#   2. STH-WP seed=524  (100 ep × 28 goals = 2800 ciclos)
#   3. SUB-WP seed=123  (100 ep × 28 goals = 2800 ciclos)
#   4. SUB-WP seed=524  (100 ep × 28 goals = 2800 ciclos)
#
# El world file warehouse_1.wbt ya tiene los peatones PEDESTRIAN_1 y
# PEDESTRIAN_2 activos — no se necesita modificación adicional.
#
# Salidas:
#   rl_train_STHWP/inferencia_sthwp/resultados/infer_dinamico_s42.csv
#   rl_train_STHWP/inferencia_sthwp/resultados/infer_dinamico_s524.csv
#   rl_train_SUB_WP_continuo/inferencia_subwp/resultados/infer_dinamico_s123.csv
#   rl_train_SUB_WP_continuo/inferencia_subwp/resultados/infer_dinamico_s524.csv
#
# Duración estimada: ~4 × 2h = ~8h
# Uso: bash run_infer_dinamico.sh

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD_STHWP="$CONTROLLERS_DIR/../worlds/warehouse_1.wbt"
WORLD_SUBWP="$CONTROLLERS_DIR/../worlds/warehouse_1_subwp.wbt"

echo "======================================================="
echo " Inferencia DINÁMICA — STH-WP vs SUB-WP"
echo " Obstáculos: 2 peatones activos en warehouse_1.wbt"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_sthwp() {
    local KEY="$1"; local LABEL="$2"
    echo ""; echo "--- [STH-WP] $LABEL — $(date '+%H:%M:%S') ---"
    echo "$KEY" > "$STHWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_STHWP"
    [ $? -ne 0 ] && echo "[ERROR] $KEY falló" && exit 1
    echo "[OK] $LABEL"
}

run_subwp() {
    local KEY="$1"; local LABEL="$2"
    echo ""; echo "--- [SUB-WP] $LABEL — $(date '+%H:%M:%S') ---"
    echo "$KEY" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_SUBWP"
    [ $? -ne 0 ] && echo "[ERROR] $KEY falló" && exit 1
    echo "[OK] $LABEL"
}

run_sthwp "infer_din_s42"  "STH-WP seed=42  — 2800 ciclos"
run_sthwp "infer_din_s524" "STH-WP seed=524 — 2800 ciclos"
run_subwp "infer_din_s123" "SUB-WP seed=123 — 2800 ciclos"
run_subwp "infer_din_s524" "SUB-WP seed=524 — 2800 ciclos"

echo ""
echo "======================================================="
echo " PIPELINE COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " STH-WP resultados:"
echo "   $STHWP_DIR/inferencia_sthwp/resultados/infer_dinamico_s42.csv"
echo "   $STHWP_DIR/inferencia_sthwp/resultados/infer_dinamico_s524.csv"
echo " SUB-WP resultados:"
echo "   $SUBWP_DIR/inferencia_subwp/resultados/infer_dinamico_s123.csv"
echo "   $SUBWP_DIR/inferencia_subwp/resultados/infer_dinamico_s524.csv"
echo "======================================================="
