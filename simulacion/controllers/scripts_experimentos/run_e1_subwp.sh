#!/bin/bash
# Experimento E1 — SUB-WP — 1 peatón (PEDESTRIAN_1)
# Lanza training de 3 seeds secuencialmente con world warehouse_1_subwp_1ped.wbt
#
# Base: subwp_s{42,123,524}_wp75_r2_stage6_din_v3_final (dropoff correcto -11,0)
# Salida: pruebas/subwp_e1_1ped_s{42,123,524}_final.zip

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"
WORLD_1PED="$WORLDS_DIR/warehouse_1_subwp_1ped.wbt"

run_train() {
    local key="$1" label="$2"
    echo ""
    echo "--- $label ---"
    echo "$key" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_1PED"
    echo "[OK] $label"
}

echo "======================================================="
echo " EXPERIMENTO E1 — SUB-WP — 1 peatón (PEDESTRIAN_1)"
echo " World: warehouse_1_subwp_1ped.wbt"
echo " 3M steps × 3 seeds = 9M steps total"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_train "e1_1ped_s42"  "E1 SUB-WP s42  — 3M steps desde r2 din_v3"
run_train "e1_1ped_s123" "E1 SUB-WP s123 — 3M steps desde r2 din_v3"
run_train "e1_1ped_s524" "E1 SUB-WP s524 — 3M steps desde r2 din_v3"

echo ""
echo "======================================================="
echo " TRAINING COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos en $SUBWP_DIR/pruebas/subwp_e1_1ped_s*.zip"
echo "======================================================="
