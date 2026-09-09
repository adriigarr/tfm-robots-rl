#!/bin/bash
# E2.2b — Fix LIDAR_MIN_DYNAMIC 1.5→2.5m: entrenamientos + inferencias
# STHWP × 3 seeds → SUBWP × 3 seeds → inferencias STHWP × 3 → inferencias SUBWP × 3
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
echo " E2.2b — Replanning LIDAR (LIDAR_MIN_DYNAMIC=2.5m)"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

# ── STHWP — entrenamiento (3 × 6M steps ≈ 18–21h) ────────────────────────────
run_sthwp "e2_2b_s42"  "STHWP E2.2b s42  — entrenamiento 6M"
run_sthwp "e2_2b_s123" "STHWP E2.2b s123 — entrenamiento 6M"
run_sthwp "e2_2b_s524" "STHWP E2.2b s524 — entrenamiento 6M"

# ── SUBWP — entrenamiento (3 × 6M steps ≈ 18–21h) ────────────────────────────
run_subwp "e2_2b_s42"  "SUBWP E2.2b s42  — entrenamiento 6M"
run_subwp "e2_2b_s123" "SUBWP E2.2b s123 — entrenamiento 6M"
run_subwp "e2_2b_s524" "SUBWP E2.2b s524 — entrenamiento 6M"

# ── STHWP — inferencia (3 × 2800 ep) ─────────────────────────────────────────
run_sthwp "infer_e2_2b_s42"  "STHWP E2.2b s42  — inferencia"
run_sthwp "infer_e2_2b_s123" "STHWP E2.2b s123 — inferencia"
run_sthwp "infer_e2_2b_s524" "STHWP E2.2b s524 — inferencia"

# ── SUBWP — inferencia (3 × 2800 ep) ─────────────────────────────────────────
run_subwp "infer_e2_2b_s42"  "SUBWP E2.2b s42  — inferencia"
run_subwp "infer_e2_2b_s123" "SUBWP E2.2b s123 — inferencia"
run_subwp "infer_e2_2b_s524" "SUBWP E2.2b s524 — inferencia"

echo ""
echo "======================================================="
echo " COMPLETADO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs STHWP: rl_train_STHWP/experimentos/resultados/sthwp_infer_e2_2b_s{42,123,524}.csv"
echo " CSVs SUBWP: rl_train_SUB_WP_continuo/experimentos/resultados/subwp_infer_e2_2b_s{42,123,524}.csv"
echo "======================================================="
