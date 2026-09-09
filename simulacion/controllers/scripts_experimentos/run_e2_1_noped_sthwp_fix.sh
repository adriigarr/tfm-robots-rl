#!/bin/bash
# Fix: re-ejecuta solo STHWP noped con warehouse_1_static.wbt (sin peatones)
# warehouse_1.wbt tenía 2 peatones moviéndose → resultado inválido (0.5%)
set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
WORLD_STHWP_NOPED="$WORLDS_DIR/warehouse_1_static.wbt"

run_sthwp_noped() {
    local key="$1" label="$2"
    echo ""; echo "--- $label ---"
    echo "$key" > "$STHWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_STHWP_NOPED"
    echo "[OK] $label"
}

echo "======================================================="
echo " STHWP noped FIX — mundo: warehouse_1_static.wbt"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_sthwp_noped "infer_e2_1_noped_s42"  "STHWP E2.1 noped s42  (static world)"
run_sthwp_noped "infer_e2_1_noped_s123" "STHWP E2.1 noped s123 (static world)"
run_sthwp_noped "infer_e2_1_noped_s524" "STHWP E2.1 noped s524 (static world)"

echo ""
echo "======================================================="
echo " COMPLETADO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Ahora analiza: /opt/anaconda3/envs/base_rl/bin/python analisis_estadistico.py"
echo "======================================================="
