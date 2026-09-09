#!/bin/bash
# Inferencia Stage 1 + Stage 2 — SUB-WP — 6 runs secuenciales
#
# Stage 1: 500 ep × goal_01 × 3 seeds
# Stage 2: 2800 ep × 28 goals × 3 seeds (s42 además genera arrival_headings_stage2.json)
#
# Uso: bash run_subwp_infer_s1_s2.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"
RESULTS_DIR="$CONTROLLER_DIR/inferencia_subwp/resultados"

echo "======================================================="
echo " Inferencia SUB-WP — Stage 1 + Stage 2 (3 seeds)"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

mkdir -p "$RESULTS_DIR"

run_webots() {
    local KEY="$1"
    local LABEL="$2"
    echo ""
    echo "------- $LABEL — $(date '+%H:%M:%S') -------"
    echo "$KEY" > "$STAGE_FILE"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    local CODE=$?
    if [ $CODE -ne 0 ]; then
        echo "[ERROR] Webots terminó con código $CODE en $KEY. Abortando."
        exit 1
    fi
    echo "[OK] $LABEL — $(date '+%H:%M:%S')"
}

echo ""
echo "━━━ INFERENCIA STAGE 1 (goal_01, 500 ep × 3 seeds) ━━━━━"

run_webots "infer_1_s42"  "Infer stage 1 — seed=42  (500 ep, goal_01)"
run_webots "infer_1_s123" "Infer stage 1 — seed=123 (500 ep, goal_01)"
run_webots "infer_1_s524" "Infer stage 1 — seed=524 (500 ep, goal_01)"

echo ""
echo "━━━ INFERENCIA STAGE 2 (28 goals, 100 ep × 3 seeds) ━━━━"
echo "    seed=42 genera arrival_headings_stage2.json"

run_webots "infer_2_s42"  "Infer stage 2 — seed=42  (2800 ep, genera headings)"
run_webots "infer_2_s123" "Infer stage 2 — seed=123 (2800 ep)"
run_webots "infer_2_s524" "Infer stage 2 — seed=524 (2800 ep)"

if [ ! -f "$RESULTS_DIR/arrival_headings_stage2.json" ]; then
    echo "[ERROR] arrival_headings_stage2.json no generado. Revisar infer_stage2_s42.py."
    exit 1
fi
echo ""
echo "[OK] arrival_headings_stage2.json listo para stage 3."

echo ""
echo "======================================================="
echo " Inferencia completa — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs en: $RESULTS_DIR/"
echo "======================================================="
