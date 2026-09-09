#!/bin/bash
# Inferencia dinámica v2_cont — modelos continuados 3M steps
# STH-WP: s42=final, s123=ckpt_2501472 (pico), s524=ckpt_2501472 (pico)
# SUB-WP: s42/s123/s524 = din_v2_cont_final
# 100 ep/goal × 28 goals × 3 seeds × 2 métodos = 16800 ciclos totales
# Uso: bash run_infer_dynamic_v2_cont.sh

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"

STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

WORLD_STHWP="$WORLDS_DIR/warehouse_1.wbt"
WORLD_SUBWP="$WORLDS_DIR/warehouse_1_subwp.wbt"

echo "======================================================="
echo " INFERENCIA DINÁMICA v2_cont"
echo " STH-WP s42(final) | s123(ckpt 2501472) | s524(ckpt 2501472)"
echo " SUB-WP s42/s123/s524 = din_v2_cont_final"
echo " 100 ep/goal × 28 goals × 6 seeds = 16800 ciclos"
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
run_sthwp "infer_r3_s42_s6din_v2_cont"  "v2_cont final — seed=42"
run_sthwp "infer_r3_s123_s6din_v2_cont" "v2_cont ckpt 2.5M — seed=123"
run_sthwp "infer_r3_s524_s6din_v2_cont" "v2_cont ckpt 2.5M — seed=524"

# ── SUB-WP ────────────────────────────────────────────────────────────────────
run_subwp "infer_din_v2_cont_s42"  "v2_cont final — seed=42"
run_subwp "infer_din_v2_cont_s123" "v2_cont final — seed=123"
run_subwp "infer_din_v2_cont_s524" "v2_cont final — seed=524"

echo ""
echo "======================================================="
echo " INFERENCIA COMPLETA — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs STH-WP:"
echo "   $STHWP_DIR/inferencia_sthwp/resultados/infer_run003_s{42,123,524}_stage6_din_v2_cont.csv"
echo " CSVs SUB-WP:"
echo "   $SUBWP_DIR/inferencia_subwp/resultados/infer_din_v2_cont_s{42,123,524}.csv"
echo "======================================================="
