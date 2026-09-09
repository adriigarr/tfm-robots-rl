#!/bin/bash
# Inferencia ciclo completo — 3 seeds secuenciales.
#
# Encadena stage4v2 (approach+exit) con stage5 (retorno) en un único
# proceso de Webots por seed. 100 ep × 28 goals = 2800 ciclos por seed.
#
# Modelos usados:
#   run002_s42_stage4v2_final  +  run002_s42_stage5_final
#   run002_s123_stage4v2_final +  run002_s123_stage5_final
#   run002_s524_stage4v2_final +  run002_s524_stage5_final
#
# Salidas CSV:
#   inferencia_sthwp/resultados/infer_run002_s42_ciclo_completo.csv
#   inferencia_sthwp/resultados/infer_run002_s123_ciclo_completo.csv
#   inferencia_sthwp/resultados/infer_run002_s524_ciclo_completo.csv
#
# Duración estimada: ~3 × 2h = ~6h
# Uso: bash run_infer_ciclo_completo.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"
RESULTS_DIR="$CONTROLLER_DIR/inferencia_sthwp/resultados"

echo "======================================================="
echo " Inferencia CICLO COMPLETO — STH-WP (3 seeds)"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo " ap+exit: run002_s*_stage4v2_final"
echo " retorno: run002_s*_stage5_final"
echo "======================================================="

mkdir -p "$RESULTS_DIR"

run_webots() {
    local KEY="$1"
    local LABEL="$2"
    echo ""
    echo "------- $LABEL"
    echo "        $(date '+%Y-%m-%d %H:%M:%S') -------"
    echo "$KEY" > "$STAGE_FILE"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    local CODE=$?
    if [ $CODE -ne 0 ]; then
        echo ""
        echo "[ERROR] Webots terminó con código $CODE en: $KEY"
        echo "        Abortando pipeline."
        exit 1
    fi
    echo "[OK] $LABEL — $(date '+%H:%M:%S')"
}

run_webots "infer_r2_s42_cc"  "Inferencia ciclo completo — seed=42  (2800 ciclos)"
run_webots "infer_r2_s123_cc" "Inferencia ciclo completo — seed=123 (2800 ciclos)"
run_webots "infer_r2_s524_cc" "Inferencia ciclo completo — seed=524 (2800 ciclos)"

echo ""
echo "======================================================="
echo " PIPELINE COMPLETO"
echo " Fin: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " CSVs:"
echo "   $RESULTS_DIR/infer_run002_s42_ciclo_completo.csv"
echo "   $RESULTS_DIR/infer_run002_s123_ciclo_completo.csv"
echo "   $RESULTS_DIR/infer_run002_s524_ciclo_completo.csv"
echo "======================================================="
