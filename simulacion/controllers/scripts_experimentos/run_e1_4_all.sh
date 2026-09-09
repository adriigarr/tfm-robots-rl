#!/bin/bash
# Experimento E1.4 — STHWP + SUB-WP — patience reward + proximidad reforzada
# STHWP: base E1.3 (52 dims) | SUBWP: base E1.3 (48 dims) + replan agresivo
# 2M steps × 6 seeds = 12M steps total
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
echo " EXPERIMENTO E1.4 — STHWP + SUB-WP"
echo " Patience reward + penalización proximidad reforzada"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

echo ""
echo "=== STHWP E1.4 (3 seeds) — base E1.3, 52 dims ==="
run_sthwp "e1_4_s42"  "STHWP E1.4 s42"
run_sthwp "e1_4_s123" "STHWP E1.4 s123"
run_sthwp "e1_4_s524" "STHWP E1.4 s524"

echo ""
echo "=== SUB-WP E1.4 (3 seeds) — base E1.3, 48 dims + replan agresivo ==="
run_subwp "e1_4_s42"  "SUBWP E1.4 s42"
run_subwp "e1_4_s123" "SUBWP E1.4 s123"
run_subwp "e1_4_s524" "SUBWP E1.4 s524"

echo ""
echo "======================================================="
echo " TRAINING COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " STHWP: rl_train_STHWP/pruebas/sthwp_e1_4_s*.zip"
echo " SUBWP: rl_train_SUB_WP_continuo/pruebas/subwp_e1_4_s*.zip"
echo "======================================================="
