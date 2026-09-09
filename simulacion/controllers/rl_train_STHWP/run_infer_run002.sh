#!/bin/bash
# Lanza inferencia completa de run002 seed=524 — stages 1, 2, 3 y 4 en serie.
# Cada stage abre Webots, ejecuta la inferencia, guarda el CSV y cierra Webots.
#
# Uso: bash run_infer_run002.sh
# Resultados: inferencia_sthwp/resultados/infer_run002_s524_stage{1..4}.csv
#
# Duración estimada (a ~1000 fps):
#   Stage 1:  ~2 min  (100 ep, goal único, ~200 steps/ep)
#   Stage 2: ~40 min  (2800 ep, 28 goals × 100 ep approach, ~850 steps/ep)
#   Stage 3: ~55 min  (2800 ep, 28 goals × 100 ep exit, ~1200 steps/ep)
#   Stage 4: ~60 min  (2800 ep, 28 goals × 100 ciclo completo ap+exit, ~1800 steps/ep)
#   TOTAL:   ~2h 40min

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"
RESULTS_DIR="$CONTROLLER_DIR/inferencia_sthwp/resultados"

echo "======================================================="
echo " Inferencia run002 s524 — stages 1 / 2 / 3 / 4"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

mkdir -p "$RESULTS_DIR"

run_stage() {
    local KEY="$1"
    local LABEL="$2"
    echo ""
    echo "-------------------------------------------------------"
    echo " Lanzando: $LABEL"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "-------------------------------------------------------"
    echo "$KEY" > "$STAGE_FILE"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    local EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Webots terminó con código $EXIT_CODE en $LABEL"
        exit 1
    fi
    echo " ✓ $LABEL completado — $(date '+%Y-%m-%d %H:%M:%S')"
}

run_stage "infer_r2_s524_s1" "Inferencia stage 1 (100 ep, goal_01)"
run_stage "infer_r2_s524_s2" "Inferencia stage 2 (2800 ep, 28 goals × 100 approach)"
run_stage "infer_r2_s524_s3" "Inferencia stage 3 (2800 ep, 28 goals × 100 exit)"
run_stage "infer_r2_s524_s4" "Inferencia stage 4 (2800 ep, 28 goals × 100 ciclo completo encadenado)"

echo ""
echo "======================================================="
echo " Inferencia completa — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs guardados en:"
echo "   $RESULTS_DIR/infer_run002_s524_stage1.csv"
echo "   $RESULTS_DIR/infer_run002_s524_stage2.csv"
echo "   $RESULTS_DIR/infer_run002_s524_stage3.csv"
echo "   $RESULTS_DIR/infer_run002_s524_stage4.csv"
echo "======================================================="
