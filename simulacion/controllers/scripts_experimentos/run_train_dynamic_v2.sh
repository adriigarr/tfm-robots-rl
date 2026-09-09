#!/bin/bash
# Entrenamiento dinámico v2 — Opción 1 + Opción A
# STH-WP: run003_sXX_stage6_final (40d) → stage6_din_v2_final (48d)
# SUB-WP: subwp_sXX_wp75_stage5_final (40d) → stage6_din_v2_final (48d)
# 1M steps × 3 seeds × 2 métodos = 6M steps totales
# Uso: bash run_train_dynamic_v2.sh

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"

STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

WORLD_STHWP="$WORLDS_DIR/warehouse_1.wbt"
WORLD_SUBWP="$WORLDS_DIR/warehouse_1_subwp.wbt"

echo "======================================================="
echo " ENTRENAMIENTO DINÁMICO v2 — Opción 1 + Opción A"
echo " STH-WP (s42, s123, s524) + SUB-WP (s42, s123, s524)"
echo " 1M steps × 6 seeds = 6M steps totales"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_sthwp() {
    local KEY="$1"; local LABEL="$2"
    echo ""
    echo "--- [STH-WP] $LABEL — $(date '+%Y-%m-%d %H:%M:%S') ---"
    echo "$KEY" > "$STHWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_STHWP"
    [ $? -ne 0 ] && echo "[ERROR] STH-WP $KEY falló" && exit 1
    echo "[OK] STH-WP $LABEL"
}

run_subwp() {
    local KEY="$1"; local LABEL="$2"
    echo ""
    echo "--- [SUB-WP] $LABEL — $(date '+%Y-%m-%d %H:%M:%S') ---"
    echo "$KEY" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_SUBWP"
    [ $? -ne 0 ] && echo "[ERROR] SUB-WP $KEY falló" && exit 1
    echo "[OK] SUB-WP $LABEL"
}

# ── STH-WP ────────────────────────────────────────────────────────────────────
run_sthwp "6_din_v2_s42"  "Stage6 din v2 — seed=42"
run_sthwp "6_din_v2_s123" "Stage6 din v2 — seed=123"
run_sthwp "6_din_v2_s524" "Stage6 din v2 — seed=524"

# ── SUB-WP ────────────────────────────────────────────────────────────────────
run_subwp "6_din_v2_s42"  "Stage6 din v2 — seed=42"
run_subwp "6_din_v2_s123" "Stage6 din v2 — seed=123"
run_subwp "6_din_v2_s524" "Stage6 din v2 — seed=524"

echo ""
echo "======================================================="
echo " ENTRENAMIENTO COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos STH-WP: pruebas/run003_sXX_stage6_din_v2_final.zip"
echo " Modelos SUB-WP: pruebas/subwp_sXX_wp75_stage6_din_v2_final.zip"
echo "======================================================="
