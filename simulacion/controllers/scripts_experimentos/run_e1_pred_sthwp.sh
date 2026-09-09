#!/bin/bash
# Entrenamiento E1_pred STHWP — obs predictiva P1 (48→52 dims) — 3 seeds secuencial
set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
WORLD="$WORLDS_DIR/warehouse_1_1ped.wbt"

run() {
    local key="$1" label="$2"
    echo ""; echo "--- $label ---"
    echo "$key" > "$STHWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    echo "[OK] $label"
}

echo "======================================================="
echo " ENTRENAMIENTO E1_pred STHWP — 3 seeds (3M steps cada uno)"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run "e1_pred_s42"  "STHWP E1_pred s42"
run "e1_pred_s123" "STHWP E1_pred s123"
run "e1_pred_s524" "STHWP E1_pred s524"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos: $STHWP_DIR/pruebas/sthwp_e1_pred_s*_final.zip"
echo "======================================================="
