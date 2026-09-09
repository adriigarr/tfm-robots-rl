#!/bin/bash
# E1.2 SUB-WP — sesgo goals 19-22 + 24-28 | 2M steps × 3 seeds
set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"
WORLD="$WORLDS_DIR/warehouse_1_subwp_1ped.wbt"

run_train() {
    local key="$1" label="$2"
    echo ""; echo "--- $label ---"
    echo "$key" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    echo "[OK] $label"
}

echo "======================================================="
echo " E1.2 SUB-WP — sesgo 70% goals 19-22+24-28 | 2M × 3 seeds"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_train "e1_2_s42"  "E1.2 SUBWP s42  — desde e1_final"
run_train "e1_2_s123" "E1.2 SUBWP s123 — desde e1_final"
run_train "e1_2_s524" "E1.2 SUBWP s524 — desde e1_final"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos en $SUBWP_DIR/pruebas/subwp_e1_2_s*.zip"
echo "======================================================="
