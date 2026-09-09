#!/bin/bash
# E1.5 — Pendiente completo: SUBWP training × 3 + toda la inferencia
# STHWP training ya completado. Ningún seed SUBWP tiene modelo guardado.
# Orden: SUBWP s42 → s123 → s524 → STHWP infer × 3 → SUBWP infer × 3
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
echo " E1.5 — PENDIENTE COMPLETO"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

echo ""
echo "========== TRAINING SUBWP E1.5 (4M × 3 seeds) =========="
run_subwp "e1_5_s42"  "SUBWP E1.5 s42  — training"
run_subwp "e1_5_s123" "SUBWP E1.5 s123 — training"
run_subwp "e1_5_s524" "SUBWP E1.5 s524 — training"

echo ""
echo "========== INFERENCIA STHWP E1.5 =========="
run_sthwp "infer_e1_5_s42"  "STHWP E1.5 s42  — inferencia"
run_sthwp "infer_e1_5_s123" "STHWP E1.5 s123 — inferencia"
run_sthwp "infer_e1_5_s524" "STHWP E1.5 s524 — inferencia"

echo ""
echo "========== INFERENCIA SUBWP E1.5 =========="
run_subwp "infer_e1_5_s42"  "SUBWP E1.5 s42  — inferencia"
run_subwp "infer_e1_5_s123" "SUBWP E1.5 s123 — inferencia"
run_subwp "infer_e1_5_s524" "SUBWP E1.5 s524 — inferencia"

echo ""
echo "======================================================="
echo " TODO COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs STHWP: rl_train_STHWP/experimentos/resultados/"
echo " CSVs SUBWP: rl_train_SUB_WP_continuo/experimentos/resultados/"
echo "======================================================="
