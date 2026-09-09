#!/bin/bash
# Ablación peatones fijos — v2_cont models con trayectoria fija de peatones
#
# Propósito: aislar el efecto de la randomización de posición inicial.
# Los modelos son los mismos que en run_infer_dynamic_v2_cont.sh, pero
# los peatones siempre arrancan desde su posición por defecto del world file:
#   PEDESTRIAN_1: x=-2, y=0.3  →  oscila hasta x=4
#   PEDESTRIAN_2: x=-9.5, y=-5 →  oscila hasta y=-0.5
#
# Seeds incluidas: STH-WP s123+s524 (las seeds funcionales), SUB-WP s42+s123+s524
# (s42 STH-WP omitida — su modelo no aprendió nada relevante)
#
# Uso: bash run_infer_ablation_fixedped.sh

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"

STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

WORLD_STHWP="$WORLDS_DIR/warehouse_1.wbt"
WORLD_SUBWP="$WORLDS_DIR/warehouse_1_subwp.wbt"

echo "======================================================="
echo " ABLACIÓN — peatones en posición FIJA (trayectoria fija)"
echo " STH-WP s123 (ckpt 2.5M) + s524 (ckpt 2.5M)"
echo " SUB-WP s42 + s123 + s524 (final)"
echo " Comparar con run_infer_dynamic_v2_cont.sh (pos. aleatoria)"
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

# ── STH-WP (s123 y s524 — checkpoint pico 2.5M) ───────────────────────────────
run_sthwp "infer_r3_s123_s6din_v2_cont_fixedped" "Ablación ped fijo — s123 (ckpt 2.5M)"
run_sthwp "infer_r3_s524_s6din_v2_cont_fixedped" "Ablación ped fijo — s524 (ckpt 2.5M)"

# ── SUB-WP (s42, s123, s524 — final) ─────────────────────────────────────────
run_subwp "infer_din_v2_cont_fixedped_s42"  "Ablación ped fijo — s42 (final)"
run_subwp "infer_din_v2_cont_fixedped_s123" "Ablación ped fijo — s123 (final)"
run_subwp "infer_din_v2_cont_fixedped_s524" "Ablación ped fijo — s524 (final)"

echo ""
echo "======================================================="
echo " ABLACIÓN COMPLETA — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs STH-WP:"
echo "   $STHWP_DIR/inferencia_sthwp/resultados/infer_run003_s{123,524}_stage6_din_v2_cont_fixedped.csv"
echo " CSVs SUB-WP:"
echo "   $SUBWP_DIR/inferencia_subwp/resultados/infer_din_v2_cont_fixedped_s{42,123,524}.csv"
echo ""
echo " Comparar con (pos. aleatoria):"
echo "   STH-WP s123: 40.4%  s524: 27.0%"
echo "   SUB-WP s42: 42.4%  s123: 26.1%  s524: 30.0%"
echo "======================================================="
