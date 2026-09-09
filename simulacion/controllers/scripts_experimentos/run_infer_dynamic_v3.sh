#!/bin/bash
# Inferencia dinámica v3 — recompensa B1, política determinista, peatones aleatorios
#
# Checkpoints seleccionados por pico de tasa_exito en entrenamiento:
#   STH-WP s42:  ckpt 4,101,792 (pico 25% → final colapsa a 12.5%)
#   STH-WP s123: ckpt 2,601,472 (pico 33.6% → final colapsa a 9%)
#   STH-WP s524: ckpt 2,801,472 (pico 38.9% → final 27.4% estable)
#   SUB-WP s42:  final          (sin colapso, curva estable)
#   SUB-WP s123: final          (sin colapso, curva estable)
#   SUB-WP s524: final          (sin colapso, creciente al final)
#
# Comparar resultados con: run_infer_dynamic_v2_cont.sh
#   STH-WP s123=40.4%  s524=27.0%
#   SUB-WP s42=42.4%   s123=26.1%  s524=30.0%
#
# Uso: bash run_infer_dynamic_v3.sh

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"

STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

WORLD_STHWP="$WORLDS_DIR/warehouse_1.wbt"
WORLD_SUBWP="$WORLDS_DIR/warehouse_1_subwp.wbt"

echo "======================================================="
echo " INFERENCIA DINÁMICA v3 — recompensa B1"
echo " Política determinista | peatones en posición aleatoria"
echo " STH-WP s42(ckpt4.1M) + s123(ckpt2.6M) + s524(ckpt2.8M)"
echo " SUB-WP s42 + s123 + s524 (modelos finales)"
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
run_sthwp "infer_r3_s42_s6din_v3"  "v3 s42  (ckpt 4.1M)"
run_sthwp "infer_r3_s123_s6din_v3" "v3 s123 (ckpt 2.6M)"
run_sthwp "infer_r3_s524_s6din_v3" "v3 s524 (ckpt 2.8M)"

# ── SUB-WP ────────────────────────────────────────────────────────────────────
run_subwp "infer_din_v3_s42"  "v3 s42  (final)"
run_subwp "infer_din_v3_s123" "v3 s123 (final)"
run_subwp "infer_din_v3_s524" "v3 s524 (final)"

echo ""
echo "======================================================="
echo " INFERENCIA v3 COMPLETA — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs STH-WP: rl_train_STHWP/inferencia_sthwp/resultados/"
echo "   infer_run003_s{42,123,524}_stage6_din_v3.csv"
echo " CSVs SUB-WP: rl_train_SUB_WP_continuo/inferencia_subwp/resultados/"
echo "   infer_din_v3_s{42,123,524}.csv"
echo "======================================================="
