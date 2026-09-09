#!/bin/bash
# Inferencia E1 — STHWP — 1 peatón — 3 seeds
set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
WORLD="$WORLDS_DIR/warehouse_1_1ped.wbt"

run_infer() {
    local key="$1" label="$2"
    echo ""
    echo "--- $label ---"
    echo "$key" > "$STHWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    echo "[OK] $label"
}

echo "======================================================="
echo " INFERENCIA E1 — STHWP — 1 peatón"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_infer "infer_e1_1ped_s42"  "STHWP E1 inferencia s42"
run_infer "infer_e1_1ped_s123" "STHWP E1 inferencia s123"
run_infer "infer_e1_1ped_s524" "STHWP E1 inferencia s524"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs en $STHWP_DIR/experimentos/resultados/"
echo "======================================================="
