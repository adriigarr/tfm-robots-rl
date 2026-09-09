#!/bin/bash
# Continuación entrenamiento dinámico v2 — 3M steps adicionales por seed
# STH-WP: run003_sXX_stage6_din_v2_final  → run003_sXX_stage6_din_v2_cont_final
# SUB-WP: subwp_sXX_wp75_stage6_din_v2_final → subwp_sXX_wp75_stage6_din_v2_cont_final
# Uso: bash run_train_dynamic_v2_cont.sh

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"

STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

WORLD_STHWP="$WORLDS_DIR/warehouse_1.wbt"
WORLD_SUBWP="$WORLDS_DIR/warehouse_1_subwp.wbt"

echo "======================================================="
echo " ENTRENAMIENTO DIN v2 CONT — 3M steps adicionales"
echo " STH-WP (s42, s123, s524) + SUB-WP (s42, s123, s524)"
echo " Base: din_v2_final  →  Salida: din_v2_cont_final"
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
run_sthwp "6_din_v2_cont_s42"  "Cont. din v2 — seed=42"
run_sthwp "6_din_v2_cont_s123" "Cont. din v2 — seed=123"
run_sthwp "6_din_v2_cont_s524" "Cont. din v2 — seed=524"

# ── SUB-WP ────────────────────────────────────────────────────────────────────
run_subwp "6_din_v2_cont_s42"  "Cont. din v2 — seed=42"
run_subwp "6_din_v2_cont_s123" "Cont. din v2 — seed=123"
run_subwp "6_din_v2_cont_s524" "Cont. din v2 — seed=524"

echo ""
echo "======================================================="
echo " ENTRENAMIENTO COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos STH-WP:"
echo "   $STHWP_DIR/pruebas/run003_s{42,123,524}_stage6_din_v2_cont_final.zip"
echo " Modelos SUB-WP:"
echo "   $SUBWP_DIR/pruebas/subwp_s{42,123,524}_wp75_stage6_din_v2_cont_final.zip"
echo "======================================================="
