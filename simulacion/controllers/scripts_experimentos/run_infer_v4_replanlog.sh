#!/bin/bash
# Inferencia v4 SUB-WP con logging de replanning por fase — 3 seeds
#
# Mide cuántos replans se disparan en cada fase (approach/exit/retorno)
# y en qué fase se producen las colisiones, para diagnosticar el cuello
# de botella en la fase de retorno identificado en §27.

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

run_subwp() {
    local key="$1" label="$2"
    echo ""
    echo "--- [SUB-WP v4 replanlog] $label ---"
    echo "$key" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLDS_DIR/warehouse_1_subwp.wbt"
    echo "[OK] $label"
}

echo "======================================================="
echo " INFERENCIA v4 SUB-WP — logging replanning por fase"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_subwp "infer_din_v4_s42_replanlog"  "s42  — 2800 episodios"
run_subwp "infer_din_v4_s123_replanlog" "s123 — 2800 episodios"
run_subwp "infer_din_v4_s524_replanlog" "s524 — 2800 episodios"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Resultados en:"
echo "   $SUBWP_DIR/inferencia_subwp/resultados/infer_din_v4_s*_replanlog.csv"
echo "   $SUBWP_DIR/inferencia_subwp/resultados/infer_din_v4_s*_replanlog_summary.txt"
echo "======================================================="
