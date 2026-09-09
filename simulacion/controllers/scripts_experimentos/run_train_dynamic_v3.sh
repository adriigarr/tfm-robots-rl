#!/bin/bash
# Entrenamiento dinámico v3 — recompensa B1 activa
#
# Novedad respecto a v2_cont:
#   - webots_env.py penaliza proximidad específica al peatón (B1): −0.8·exp(−2·d) si d < 2m
#   - ent_coef reducido a 0.005 para mitigar colapso post-pico
#
# Puntos de partida:
#   STH-WP s42:   run003_s42_stage6_din_v2_cont_final        (modelo final v2_cont)
#   STH-WP s123:  checkpoint v2_cont paso 2501472 (pico)      (modelo pico v2_cont)
#   STH-WP s524:  checkpoint v2_cont paso 2501472 (pico)      (modelo pico v2_cont)
#   SUB-WP s42:   subwp_s42_wp75_stage6_din_v2_cont_final
#   SUB-WP s123:  subwp_s123_wp75_stage6_din_v2_cont_final
#   SUB-WP s524:  subwp_s524_wp75_stage6_din_v2_cont_final
#
# Duración: 2M steps adicionales por seed (paso 4M → 6M en TensorBoard)
#
# Uso: bash run_train_dynamic_v3.sh

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"

STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

WORLD_STHWP="$WORLDS_DIR/warehouse_1.wbt"
WORLD_SUBWP="$WORLDS_DIR/warehouse_1_subwp.wbt"

echo "======================================================="
echo " ENTRENAMIENTO DINÁMICO v3 — recompensa B1"
echo " STH-WP s42 + s123 (ckpt pico) + s524 (ckpt pico)"
echo " SUB-WP s42 + s123 + s524"
echo " 2M steps adicionales por seed (4M → 6M total)"
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
run_sthwp "6_din_v3_s42"  "v3 s42  (desde v2_cont final)"
run_sthwp "6_din_v3_s123" "v3 s123 (desde ckpt pico 2.5M)"
run_sthwp "6_din_v3_s524" "v3 s524 (desde ckpt pico 2.5M)"

# ── SUB-WP ────────────────────────────────────────────────────────────────────
run_subwp "6_din_v3_s42"  "v3 s42  (desde v2_cont final)"
run_subwp "6_din_v3_s123" "v3 s123 (desde v2_cont final)"
run_subwp "6_din_v3_s524" "v3 s524 (desde v2_cont final)"

echo ""
echo "======================================================="
echo " ENTRENAMIENTO v3 COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos guardados en:"
echo "   STH-WP: rl_train_STHWP/pruebas/run003_s{42,123,524}_stage6_din_v3_final.zip"
echo "   SUB-WP: rl_train_SUB_WP_continuo/pruebas/subwp_s{42,123,524}_wp75_stage6_din_v3_final.zip"
echo " Siguiente paso: bash run_infer_dynamic_v3.sh"
echo "======================================================="
