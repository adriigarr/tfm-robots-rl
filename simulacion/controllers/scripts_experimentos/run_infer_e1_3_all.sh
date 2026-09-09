#!/bin/bash
# Inferencia E1.3 — STHWP + SUB-WP — 3 seeds cada uno — secuencial
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
echo " INFERENCIA E1.3 — STHWP + SUB-WP — 6 seeds total"
echo " 28 goals × 100 episodios × 6 seeds = 168.000 eps"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

echo ""
echo "=== STHWP E1.3 (3 seeds) — 52 dims, pred_horizon=[1.0,2.0] ==="
run_sthwp "infer_e1_3_s42"  "STHWP E1.3 s42"
run_sthwp "infer_e1_3_s123" "STHWP E1.3 s123"
run_sthwp "infer_e1_3_s524" "STHWP E1.3 s524"

echo ""
echo "=== SUB-WP E1.3 (3 seeds) — 48 dims, sin pred_horizon ==="
run_subwp "infer_e1_3_s42"  "SUBWP E1.3 s42"
run_subwp "infer_e1_3_s123" "SUBWP E1.3 s123"
run_subwp "infer_e1_3_s524" "SUBWP E1.3 s524"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " STHWP CSVs: $STHWP_DIR/experimentos/resultados/"
echo " SUBWP CSVs: $SUBWP_DIR/experimentos/resultados/"
echo "======================================================="
