#!/bin/bash
# Inferencia SUB-WP — Ciclo completo (approach + exit + return) — 3 seeds
#
# Modelo: subwp_s*_wp75_stage5_final  (conoce las 3 fases)
# Stage=6: encadena approach → exit → return en un único episodio
# 100 ep × 28 goals × 3 seeds = 8400 episodios totales
#
# Tiempo estimado: ~45-60 min por seed (~2-2.5h total)
# Salidas: inferencia_subwp/resultados/infer_subwp_s*_wp75_ciclo.csv
#
# Uso: bash run_subwp_infer_ciclo.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"
RESULTS_DIR="$CONTROLLER_DIR/inferencia_subwp/resultados"

echo "======================================================="
echo " Inferencia SUB-WP — Ciclo completo — 3 seeds"
echo " approach → exit → return (stage5_final, det=True)"
echo " 100 ep × 28 goals × 3 seeds = 8400 episodios"
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

run_stage "infer_ciclo_s42"  "Ciclo completo — seed=42  (2800 ep)"
run_stage "infer_ciclo_s123" "Ciclo completo — seed=123 (2800 ep)"
run_stage "infer_ciclo_s524" "Ciclo completo — seed=524 (2800 ep)"

echo ""
echo "======================================================="
echo " Inferencia ciclo completo — $(date '+%Y-%m-%d %H:%M:%S')"
echo "   $RESULTS_DIR/infer_subwp_s42_wp75_ciclo.csv"
echo "   $RESULTS_DIR/infer_subwp_s123_wp75_ciclo.csv"
echo "   $RESULTS_DIR/infer_subwp_s524_wp75_ciclo.csv"
echo "======================================================="
