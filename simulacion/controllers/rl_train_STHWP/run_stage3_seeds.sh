#!/bin/bash
# Lanza los 3 seeds del stage 3 secuencialmente.
# Cada seed escribe en current_stage.txt y arranca Webots en modo fast.
# Al terminar cada entrenamiento, Webots cierra solo (simulationQuit).
#
# Uso: bash run_stage3_seeds.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"

SEEDS=("3_s42" "3_s123" "3_s524")

echo "======================================================="
echo " Stage 3 — 3 seeds — $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

for SEED_TAG in "${SEEDS[@]}"; do
    echo ""
    echo "------- Iniciando $SEED_TAG — $(date '+%H:%M:%S') -------"

    echo "$SEED_TAG" > "$STAGE_FILE"
    echo "[run_stage3] current_stage.txt = $SEED_TAG"

    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Webots terminó con código $EXIT_CODE en $SEED_TAG. Abortando."
        exit 1
    fi

    echo "[OK] $SEED_TAG completado — $(date '+%H:%M:%S')"
done

echo ""
echo "======================================================="
echo " Stage 3 completo — $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="
