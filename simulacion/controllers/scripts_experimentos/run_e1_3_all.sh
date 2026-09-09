#!/bin/bash
# Experimento E1.3 — STHWP + SUB-WP — trayectoria P1 extendida x∈[-4,4]
# Lanza training de 3 seeds × 2 sistemas = 6 runs secuenciales
#
# STHWP: fine-tune desde E1_pred (52 dims, pred_horizon=[1.0,2.0]), lr=1e-5, 2M steps
# SUBWP: fine-tune desde E1.2   (48 dims, sin pred_horizon),         lr=1e-5, 2M steps

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
echo " EXPERIMENTO E1.3 — STHWP + SUB-WP"
echo " P1 trayectoria x∈[-4,4] (antes -2,4)"
echo " 2M steps × 6 seeds = 12M steps total"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

echo ""
echo "=== STHWP E1.3 (3 seeds) — base: E1_pred 52 dims ==="
run_sthwp "e1_3_s42"  "STHWP E1.3 s42  — 2M steps desde E1_pred"
run_sthwp "e1_3_s123" "STHWP E1.3 s123 — 2M steps desde E1_pred"
run_sthwp "e1_3_s524" "STHWP E1.3 s524 — 2M steps desde E1_pred"

echo ""
echo "=== SUB-WP E1.3 (3 seeds) — base: E1.2 48 dims ==="
run_subwp "e1_3_s42"  "SUBWP E1.3 s42  — 2M steps desde E1.2"
run_subwp "e1_3_s123" "SUBWP E1.3 s123 — 2M steps desde E1.2"
run_subwp "e1_3_s524" "SUBWP E1.3 s524 — 2M steps desde E1.2"

echo ""
echo "======================================================="
echo " TRAINING COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " STHWP: $STHWP_DIR/pruebas/sthwp_e1_3_s*.zip"
echo " SUBWP: $SUBWP_DIR/pruebas/subwp_e1_3_s*.zip"
echo "======================================================="
