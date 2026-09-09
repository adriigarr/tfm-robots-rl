#!/bin/bash
# SUB-WP Stage 5 v3 — Return puro (ret_path[2] → espera) — 3 seeds
#
# VERSIÓN DEFINITIVA: corrección de heading_to_webots_rotation.
# v1: spawn en zona_descarga (0.0,10.5) → bumper toca wall3 en step 1.
# v2: spawn en ret_path[2]=(0.0,7.25) → mismo fallo, causa diferente.
# v3: bug real — heading_to_webots_rotation usaba eje Y (pitch) en lugar de Z (yaw).
#     Para θ≈-108° (heading suroeste), el pitch de 108° hundía el bumper 0.29m
#     bajo el suelo → bumper disparaba en step 1 en ambas posiciones de spawn.
#     Fix: [0,1,0,-θ] → [0,0,1,-θ] en heading_utils.py.
#
# Hiperparámetros: lr=3e-4, ent_coef=0.02
# TensorBoard: stage5_subwp_s*_wp75_v3_0/  (carpetas nuevas, sin solapamiento)
#
# ~2M steps × 3 seeds  (~2-3h c/u)
# Base: subwp_s*_wp75_stage4_final.zip
# Salidas: pruebas/subwp_s*_wp75_stage5_final.zip
#
# Uso: bash run_subwp_stage5_v3.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"

echo "======================================================="
echo " SUB-WP Stage 5 v3 — Return puro — 3 seeds (2M steps c/u)"
echo " Spawn: ret_path[2]=(0.0,7.25), heading eje Z (yaw correcto)"
echo " Fix: heading_to_webots_rotation [0,1,0,-θ] → [0,0,1,-θ]"
echo " lr=3e-4, ent_coef=0.02"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

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

run_webots "5_s42"  "Stage 5 v3 — seed=42  (2M steps, return puro)"
run_webots "5_s123" "Stage 5 v3 — seed=123 (2M steps, return puro)"
run_webots "5_s524" "Stage 5 v3 — seed=524 (2M steps, return puro)"

echo ""
echo "======================================================="
echo " Stage 5 v3 completo — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos:"
echo "   pruebas/subwp_s42_wp75_stage5_final.zip"
echo "   pruebas/subwp_s123_wp75_stage5_final.zip"
echo "   pruebas/subwp_s524_wp75_stage5_final.zip"
echo " TensorBoard:"
echo "   tensorboard_logs/stage5_subwp_s42_wp75_v3_0/"
echo "   tensorboard_logs/stage5_subwp_s123_wp75_v3_0/"
echo "   tensorboard_logs/stage5_subwp_s524_wp75_v3_0/"
echo "======================================================="
