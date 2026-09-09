#!/bin/bash
# Corrección SUBWP s123: E2.0b (desde stage4) → E2.1 s123 → inferencia s123
# Causa: subwp_s123_wp75_r2_stage5_final tenía olvido catastrófico del approach.
# Fix: E2.0b carga desde stage4_r2 (domina approach+exit) para obtener stage6 válido.
set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"
WORLD_SUBWP="$WORLDS_DIR/warehouse_1_subwp_1ped.wbt"

run_subwp() {
    local key="$1" label="$2"
    echo ""; echo "--- $label ---"
    echo "$key" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_SUBWP"
    echo "[OK] $label"
}

echo "======================================================="
echo " SUBWP s123 — Corrección E2.0b + E2.1 + inferencia"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_subwp "e2_0b_s123"   "SUBWP E2.0b s123 — stage6 sin peatón (desde stage4)"
run_subwp "e2_1_s123"    "SUBWP E2.1  s123 — entrenamiento con peatón (6M)"
run_subwp "infer_e2_1_s123" "SUBWP E2.1  s123 — inferencia"

echo ""
echo "======================================================="
echo " COMPLETADO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSV: rl_train_SUB_WP_continuo/experimentos/resultados/subwp_infer_e2_1_s123.csv"
echo "======================================================="
