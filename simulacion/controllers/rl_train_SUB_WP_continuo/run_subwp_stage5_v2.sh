#!/bin/bash
# SUB-WP Stage 5 v2 — Return puro (ret_path[2] → espera) — 3 seeds
#
# VERSIÓN CORREGIDA: spawn en (0.0, 7.25) = ret_path[2], 4.65m de wall3.
# La v1 fallaba con ep_len=1 por colisión bumper vs pared norte al spawnar
# en zona_descarga (0.0, 10.5) con heading [0,1,0,π/2].
#
# Hiperparámetros: lr=3e-4, ent_coef=0.02
# TensorBoard: stage5_subwp_s*_wp75_v2_0/  (carpetas nuevas, sin solapamiento)
#
# ~2M steps × 3 seeds  (~2-3h c/u)
# Base: subwp_s*_wp75_stage4_final.zip
# Salidas: pruebas/subwp_s*_wp75_stage5_final.zip
#
# Uso: bash run_subwp_stage5_v2.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"

echo "======================================================="
echo " SUB-WP Stage 5 v2 — Return puro — 3 seeds (2M steps c/u)"
echo " Spawn: ret_path[2]=(0.0,7.25), 4.65m de wall3"
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

run_webots "5_s42"  "Stage 5 v2 — seed=42  (2M steps, return puro)"
run_webots "5_s123" "Stage 5 v2 — seed=123 (2M steps, return puro)"
run_webots "5_s524" "Stage 5 v2 — seed=524 (2M steps, return puro)"

echo ""
echo "======================================================="
echo " Stage 5 v2 completo — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos:"
echo "   pruebas/subwp_s42_wp75_stage5_final.zip"
echo "   pruebas/subwp_s123_wp75_stage5_final.zip"
echo "   pruebas/subwp_s524_wp75_stage5_final.zip"
echo " TensorBoard:"
echo "   tensorboard_logs/stage5_subwp_s42_wp75_v2_0/"
echo "   tensorboard_logs/stage5_subwp_s123_wp75_v2_0/"
echo "   tensorboard_logs/stage5_subwp_s524_wp75_v2_0/"
echo "======================================================="
