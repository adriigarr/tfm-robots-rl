#!/bin/bash
# Lanza el stage 4 (ciclo completo) con seed=524.
# Solo un seed — el que pasó el criterio de stage 3.
# Webots cierra automáticamente al terminar.
#
# Uso: bash run_stage4.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"

echo "======================================================="
echo " Stage 4 — ciclo completo — seed 524"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

echo "4" > "$STAGE_FILE"
echo "[run_stage4] current_stage.txt = 4"

"$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Webots terminó con código $EXIT_CODE. Revisa los logs."
    exit 1
fi

echo ""
echo "======================================================="
echo " Stage 4 completado — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelo guardado en pruebas/run002_s524_stage4_final.zip"
echo "======================================================="
