#!/bin/bash
# Ejecuta los 3 seeds de stage 1 de forma secuencial.
# Webots se cierra automáticamente al terminar cada seed.
# Uso: bash run_stage1_seeds.sh

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLD="/Users/adrigarcia/tfm-robots-rl/simulacion/worlds/warehouse_1.wbt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"

SEEDS=("1_s42" "1_s123" "1_s524")

echo "======================================================"
echo " Stage 1 — entrenamiento secuencial de 3 seeds"
echo " Inicio: $(date)"
echo "======================================================"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "[$(date +%H:%M:%S)] Lanzando seed: $seed"
    echo "$seed" > "$STAGE_FILE"

    "$WEBOTS" --mode=fast --no-rendering "$WORLD"
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Webots terminó con código $EXIT_CODE en seed $seed. Abortando."
        exit 1
    fi

    echo "[$(date +%H:%M:%S)] Seed $seed completado."
done

echo ""
echo "======================================================"
echo " Los 3 seeds de stage 1 han terminado."
echo " Fin: $(date)"
echo " Revisa TensorBoard antes de lanzar stage 2."
echo "======================================================"
