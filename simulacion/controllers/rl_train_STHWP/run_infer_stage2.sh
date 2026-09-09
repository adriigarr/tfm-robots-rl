#!/bin/bash
# Lanza la inferencia de stage 2 para los 3 seeds secuencialmente.
# 28 goals × 10 episodios = 280 episodios por seed (840 en total).
#
# Uso: bash run_infer_stage2.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"

SEEDS=("infer_2_s42" "infer_2_s123" "infer_2_s524")

echo "======================================================="
echo " Inferencia Stage 2 — 3 seeds — $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

for SEED_TAG in "${SEEDS[@]}"; do
    echo ""
    echo "------- Iniciando $SEED_TAG — $(date '+%H:%M:%S') -------"

    echo "$SEED_TAG" > "$STAGE_FILE"
    echo "[run_infer_stage2] current_stage.txt = $SEED_TAG"

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
echo " Inferencia Stage 2 completa — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Resultados en: inferencia_sthwp/resultados/"
echo "======================================================="
