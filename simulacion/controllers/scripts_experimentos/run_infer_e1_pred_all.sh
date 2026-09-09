#!/bin/bash
# Inferencia E1_pred — STHWP + SUB-WP — 3 seeds cada uno — secuencial
set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"
WORLD_STHWP="$WORLDS_DIR/warehouse_1_1ped.wbt"
WORLD_SUBWP="$WORLDS_DIR/warehouse_1_subwp_1ped.wbt"

run_sthwp() {
    local key="$1" label="$2"
    echo ""; echo "--- $label ---"
    echo "$key" > "$STHWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_STHWP"
    echo "[OK] $label"
}

run_subwp() {
    local key="$1" label="$2"
    echo ""; echo "--- $label ---"
    echo "$key" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_SUBWP"
    echo "[OK] $label"
}

echo "======================================================="
echo " INFERENCIA E1_pred — STHWP + SUB-WP — 6 seeds total"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

echo ""
echo "=== STHWP E1_pred (3 seeds) ==="
run_sthwp "infer_e1_pred_s42"  "STHWP E1_pred s42"
run_sthwp "infer_e1_pred_s123" "STHWP E1_pred s123"
run_sthwp "infer_e1_pred_s524" "STHWP E1_pred s524"

echo ""
echo "=== SUB-WP E1_pred (3 seeds) ==="
run_subwp "infer_e1_pred_s42"  "SUBWP E1_pred s42"
run_subwp "infer_e1_pred_s123" "SUBWP E1_pred s123"
run_subwp "infer_e1_pred_s524" "SUBWP E1_pred s524"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " STHWP CSVs: $STHWP_DIR/experimentos/resultados/"
echo " SUBWP CSVs: $SUBWP_DIR/experimentos/resultados/"
echo "======================================================="
