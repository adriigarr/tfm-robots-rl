#!/bin/bash
# Inferencia v4 — Replanning integrado en entorno | 6 variantes
#
# Matriz:
#   STH-WP  s42, s123, s524  — modelo v4_final, peatones dinámicos
#   SUB-WP  s42, s123, s524  — modelo v4_final, peatones dinámicos
#
# 100 ep × 28 goals × 6 variantes = 16800 episodios totales

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

run_sthwp() {
    local key="$1" label="$2"
    echo ""
    echo "--- [STH-WP v4] $label ---"
    echo "$key" > "$STHWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLDS_DIR/warehouse_1.wbt"
    echo "[OK] $label"
}

run_subwp() {
    local key="$1" label="$2"
    echo ""
    echo "--- [SUB-WP v4] $label ---"
    echo "$key" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLDS_DIR/warehouse_1_subwp.wbt"
    echo "[OK] $label"
}

echo "======================================================="
echo " INFERENCIA v4 — Replanning integrado en entorno"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

# ── STH-WP v4 ─────────────────────────────────────────────────────────────
run_sthwp "infer_r3_s42_s6din_v4"  "STH-WP s42  v4_final (base v3: 35.7%)"
run_sthwp "infer_r3_s123_s6din_v4" "STH-WP s123 v4_final (base v3: 38.0%)"
run_sthwp "infer_r3_s524_s6din_v4" "STH-WP s524 v4_final (base v3: 45.4%)"

# ── SUB-WP v4 ─────────────────────────────────────────────────────────────
run_subwp "infer_din_v4_s42"  "SUB-WP s42  v4_final (base v3: 39.8%)"
run_subwp "infer_din_v4_s123" "SUB-WP s123 v4_final (base v3: 38.3%)"
run_subwp "infer_din_v4_s524" "SUB-WP s524 v4_final (base v3: ~38%)"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs en:"
echo "   $STHWP_DIR/inferencia_sthwp/resultados/infer_run003_s*_stage6_din_v4.csv"
echo "   $SUBWP_DIR/inferencia_subwp/resultados/infer_din_v4_s*.csv"
echo "======================================================="
