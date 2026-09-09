#!/bin/bash
# Ablación E2.1 sin peatón: modelos E2.1 en mundo sin obstáculo dinámico
# STHWP × 3 seeds → SUBWP × 3 seeds
# Worlds: warehouse_1.wbt (STHWP) / warehouse_1_subwp_noped.wbt (SUBWP)
set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"
WORLD_STHWP_NOPED="$WORLDS_DIR/warehouse_1_static.wbt"
WORLD_SUBWP_NOPED="$WORLDS_DIR/warehouse_1_subwp_noped.wbt"

run_sthwp_noped() {
    local key="$1" label="$2"
    echo ""; echo "--- $label ---"
    echo "$key" > "$STHWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_STHWP_NOPED"
    echo "[OK] $label"
}

run_subwp_noped() {
    local key="$1" label="$2"
    echo ""; echo "--- $label ---"
    echo "$key" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_SUBWP_NOPED"
    echo "[OK] $label"
}

echo "======================================================="
echo " Ablación E2.1 sin peatón"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

# ── STHWP — inferencia sin peatón (3 × 2800 ep ≈ 3-4h total) ─────────────────
run_sthwp_noped "infer_e2_1_noped_s42"  "STHWP E2.1 noped s42"
run_sthwp_noped "infer_e2_1_noped_s123" "STHWP E2.1 noped s123"
run_sthwp_noped "infer_e2_1_noped_s524" "STHWP E2.1 noped s524"

# ── SUBWP — inferencia sin peatón (3 × 2800 ep ≈ 3-4h total) ─────────────────
run_subwp_noped "infer_e2_1_noped_s42"  "SUBWP E2.1 noped s42"
run_subwp_noped "infer_e2_1_noped_s123" "SUBWP E2.1 noped s123"
run_subwp_noped "infer_e2_1_noped_s524" "SUBWP E2.1 noped s524"

echo ""
echo "======================================================="
echo " COMPLETADO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs STHWP: rl_train_STHWP/experimentos/resultados/sthwp_infer_e2_1_noped_s{42,123,524}.csv"
echo " CSVs SUBWP: rl_train_SUB_WP_continuo/experimentos/resultados/subwp_infer_e2_1_noped_s{42,123,524}.csv"
echo ""
echo " Para analizar:"
echo "   /opt/anaconda3/envs/base_rl/bin/python analisis_estadistico.py"
echo "======================================================="
