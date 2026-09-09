#!/bin/bash
# Stage 3 v2 — 3 seeds secuenciales (s42, s123, s524)
#
# Cambios vs stage3 original (i5):
#   - Headings de teleport: reales medidos en inferencia stage 2
#   - Reward: marcha atrás hacia subgoal premiada (vel_sign * cos(angulo_rel))
#   - ENT_COEF=0.01 para exploración de maniobras de retroceso
#
# Salidas:
#   pruebas/run002_s42_stage3v2_final.zip
#   pruebas/run002_s123_stage3v2_final.zip
#   pruebas/run002_s524_stage3v2_final.zip
#
# Duración estimada: ~3 × 4–5h = 12–15h
#
# Uso: bash run_stage3v2_seeds.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"

SEEDS=("3v2_s42" "3v2_s123" "3v2_s524")

echo "======================================================="
echo " Stage 3 v2 — 3 seeds — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Headings reales + reward marcha atrás"
echo "======================================================="

for SEED_TAG in "${SEEDS[@]}"; do
    echo ""
    echo "------- Iniciando $SEED_TAG — $(date '+%H:%M:%S') -------"

    echo "$SEED_TAG" > "$STAGE_FILE"

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
echo " Stage 3 v2 completo — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos guardados en pruebas/run002_s*_stage3v2_final.zip"
echo "======================================================="
