#!/bin/bash
# Entrenamiento dinámico con peatones — STH-WP y SUB-WP, 3 seeds cada uno.
#
# STH-WP: parte de run003_sXX_stage6_final   → run003_sXX_stage6_din_final
# SUB-WP: parte de subwp_sXX_wp75_stage5_final → subwp_sXX_wp75_stage6_din_final
#
# World STH-WP: warehouse_1.wbt         (controller "rl_train_STHWP")
# World SUB-WP: warehouse_1_subwp.wbt   (controller "rl_train_SUB_WP_continuo")
# Steps: 1M por seed | LR: 5e-5 | ent_coef: 0.01
# Uso: bash run_train_dynamic.sh

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"

STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

WORLD_STHWP="$WORLDS_DIR/warehouse_1.wbt"
WORLD_SUBWP="$WORLDS_DIR/warehouse_1_subwp.wbt"

echo "======================================================="
echo " ENTRENAMIENTO DINÁMICO CON PEATONES"
echo " STH-WP (s42, s123, s524) + SUB-WP (s42, s123, s524)"
echo " World STH-WP: warehouse_1.wbt"
echo " World SUB-WP: warehouse_1_subwp.wbt"
echo " LR=5e-5 | ent_coef=0.01 | 1M steps/seed"
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
run_sthwp "6_din_s42"  "Stage 6 din — seed=42  (1M steps)"
run_sthwp "6_din_s123" "Stage 6 din — seed=123 (1M steps)"
run_sthwp "6_din_s524" "Stage 6 din — seed=524 (1M steps)"

# ── SUB-WP ────────────────────────────────────────────────────────────────────
run_subwp "6_din_s42"  "Stage 6 din — seed=42  (1M steps)"
run_subwp "6_din_s123" "Stage 6 din — seed=123 (1M steps)"
run_subwp "6_din_s524" "Stage 6 din — seed=524 (1M steps)"

echo ""
echo "======================================================="
echo " PIPELINE COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos STH-WP:"
echo "   $STHWP_DIR/pruebas/run003_s42_stage6_din_final.zip"
echo "   $STHWP_DIR/pruebas/run003_s123_stage6_din_final.zip"
echo "   $STHWP_DIR/pruebas/run003_s524_stage6_din_final.zip"
echo " Modelos SUB-WP:"
echo "   $SUBWP_DIR/pruebas/subwp_s42_wp75_stage6_din_final.zip"
echo "   $SUBWP_DIR/pruebas/subwp_s123_wp75_stage6_din_final.zip"
echo "   $SUBWP_DIR/pruebas/subwp_s524_wp75_stage6_din_final.zip"
echo "======================================================="
