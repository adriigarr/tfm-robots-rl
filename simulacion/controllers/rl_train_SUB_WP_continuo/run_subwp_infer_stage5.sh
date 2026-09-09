#!/bin/bash
# Inferencia SUB-WP stage 5 — Return puro — 3 seeds
#
# 300 ep × 3 seeds (~5 min c/u)
# Salidas: inferencia_subwp/resultados/infer_subwp_s*_wp75_stage5.csv
#
# Uso: bash run_subwp_infer_stage5.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"
RESULTS_DIR="$CONTROLLER_DIR/inferencia_subwp/resultados"

echo "======================================================="
echo " Inferencia SUB-WP Stage 5 — Return puro — 3 seeds"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

mkdir -p "$RESULTS_DIR"

run_stage() {
    local KEY="$1"; local LABEL="$2"
    echo ""; echo "--- $LABEL — $(date '+%H:%M:%S') ---"
    echo "$KEY" > "$STAGE_FILE"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    [ $? -ne 0 ] && echo "[ERROR] $KEY falló" && exit 1
    echo " OK: $LABEL"
}

run_stage "infer_5_s42"  "Stage 5 — seed=42  (300 ep, return puro)"
run_stage "infer_5_s123" "Stage 5 — seed=123 (300 ep, return puro)"
run_stage "infer_5_s524" "Stage 5 — seed=524 (300 ep, return puro)"

echo ""
echo "======================================================="
echo " Inferencia stage 5 completa — $(date '+%Y-%m-%d %H:%M:%S')"
echo "   $RESULTS_DIR/infer_subwp_s42_wp75_stage5.csv"
echo "   $RESULTS_DIR/infer_subwp_s123_wp75_stage5.csv"
echo "   $RESULTS_DIR/infer_subwp_s524_wp75_stage5.csv"
echo "======================================================="
