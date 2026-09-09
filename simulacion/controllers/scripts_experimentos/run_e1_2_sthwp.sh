#!/bin/bash
# E1.2 STHWP — sesgo goals 24-28 | 1.5M steps × 3 seeds
set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
WORLD="$WORLDS_DIR/warehouse_1_1ped.wbt"

run_train() {
    local key="$1" label="$2"
    echo ""; echo "--- $label ---"
    echo "$key" > "$STHWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    echo "[OK] $label"
}

echo "======================================================="
echo " E1.2 STHWP — sesgo 70% goals 24-28 | 1.5M × 3 seeds"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_train "e1_2_s42"  "E1.2 STHWP s42  — desde e1_final"
run_train "e1_2_s123" "E1.2 STHWP s123 — desde e1_final"
run_train "e1_2_s524" "E1.2 STHWP s524 — desde ckpt 6.5M"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos en $STHWP_DIR/pruebas/sthwp_e1_2_s*.zip"
echo "======================================================="
