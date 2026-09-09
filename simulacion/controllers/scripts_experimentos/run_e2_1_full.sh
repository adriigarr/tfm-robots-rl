#!/bin/bash
# E2.0 + E2.1 — Obs realista (solo LIDAR 5m, sin supervisor), 40 dims
# E2.0: SUBWP stage6 sin peatón (2M steps × 3 seeds) — genera base equivalente a STHWP stage6_final
# E2.1: STHWP desde stage6_final + SUBWP desde E2.0 (6M steps × 3 seeds)
# Orden: SUBWP E2.0 × 3 → STHWP E2.1 × 3 → SUBWP E2.1 × 3 → inferencias × 6
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
echo " E2.0 + E2.1 — Obs realista: solo LIDAR 5m (sin supervisor)"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

echo ""
echo "========== PRE-TRAINING SUBWP E2.0 (2M × 3 seeds) =========="
run_subwp "e2_0_s42"  "SUBWP E2.0 s42  — stage6 sin peatón"
run_subwp "e2_0_s123" "SUBWP E2.0 s123 — stage6 sin peatón"
run_subwp "e2_0_s524" "SUBWP E2.0 s524 — stage6 sin peatón"

echo ""
echo "========== TRAINING STHWP E2.1 (6M × 3 seeds) =========="
run_sthwp "e2_1_s42"  "STHWP E2.1 s42  — training"
run_sthwp "e2_1_s123" "STHWP E2.1 s123 — training"
run_sthwp "e2_1_s524" "STHWP E2.1 s524 — training"

echo ""
echo "========== TRAINING SUBWP E2.1 (6M × 3 seeds) =========="
run_subwp "e2_1_s42"  "SUBWP E2.1 s42  — training"
run_subwp "e2_1_s123" "SUBWP E2.1 s123 — training"
run_subwp "e2_1_s524" "SUBWP E2.1 s524 — training"

echo ""
echo "========== INFERENCIA STHWP E2.1 =========="
run_sthwp "infer_e2_1_s42"  "STHWP E2.1 s42  — inferencia"
run_sthwp "infer_e2_1_s123" "STHWP E2.1 s123 — inferencia"
run_sthwp "infer_e2_1_s524" "STHWP E2.1 s524 — inferencia"

echo ""
echo "========== INFERENCIA SUBWP E2.1 =========="
run_subwp "infer_e2_1_s42"  "SUBWP E2.1 s42  — inferencia"
run_subwp "infer_e2_1_s123" "SUBWP E2.1 s123 — inferencia"
run_subwp "infer_e2_1_s524" "SUBWP E2.1 s524 — inferencia"

echo ""
echo "======================================================="
echo " TODO COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs STHWP: rl_train_STHWP/experimentos/resultados/"
echo " CSVs SUBWP: rl_train_SUB_WP_continuo/experimentos/resultados/"
echo "======================================================="
