#!/bin/bash
# Inferencia stage 3 v2 — 3 seeds secuenciales (s42, s123, s524)
#
# Evalúa los modelos run002_s*_stage3v2_final con política determinista.
# 100 episodios por goal × 28 goals = 2800 ep por seed.
#
# Salidas:
#   inferencia_sthwp/resultados/infer_run002_s42_stage3v2.csv
#   inferencia_sthwp/resultados/infer_run002_s123_stage3v2.csv
#   inferencia_sthwp/resultados/infer_run002_s524_stage3v2.csv
#
# Duración estimada: ~3 × 40min = ~2h
#
# Uso: bash run_infer_stage3v2_seeds.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"
RESULTS_DIR="$CONTROLLER_DIR/inferencia_sthwp/resultados"

echo "======================================================="
echo " Inferencia stage 3 v2 — 3 seeds"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

mkdir -p "$RESULTS_DIR"

run_infer() {
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
    echo "[OK] $LABEL completado — $(date '+%H:%M:%S')"
}

run_infer "infer_r2_s42_s3v2"  "Inferencia s42  stage3v2 (2800 ep)"
run_infer "infer_r2_s123_s3v2" "Inferencia s123 stage3v2 (2800 ep)"
run_infer "infer_r2_s524_s3v2" "Inferencia s524 stage3v2 (2800 ep)"

echo ""
echo "======================================================="
echo " Inferencia stage 3 v2 completa — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs en:"
echo "   $RESULTS_DIR/infer_run002_s42_stage3v2.csv"
echo "   $RESULTS_DIR/infer_run002_s123_stage3v2.csv"
echo "   $RESULTS_DIR/infer_run002_s524_stage3v2.csv"
echo "======================================================="
