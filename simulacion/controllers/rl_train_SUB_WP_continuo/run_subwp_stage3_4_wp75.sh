#!/bin/bash
# SUB-WP WP_STEP_EXIT=0.75m — Reentrenamiento stages 3 y 4 — 3 seeds
#
# Stage 3 (exit puro):  4M steps × 3 seeds  (~5-6h c/u)  → subwp_s*_wp75_stage3_final.zip
# Stage 4 (ciclo comp): 4M steps × 3 seeds  (~5-6h c/u)  → subwp_s*_wp75_stage4_final.zip
#
# Duración estimada total: ~30-36h
#
# Prerrequisitos:
#   - pruebas/subwp_s{42,123,524}_stage2_final.zip  (ya existentes)
#   - inferencia_subwp/resultados/arrival_headings_stage2.json (ya existente)
#
# Uso: bash run_subwp_stage3_4_wp75.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"
HEADINGS="$CONTROLLER_DIR/inferencia_subwp/resultados/arrival_headings_stage2.json"

if [ ! -f "$HEADINGS" ]; then
    echo "[ERROR] arrival_headings_stage2.json no encontrado."
    echo "        Ejecuta primero la inferencia de stage 2."
    exit 1
fi

echo "======================================================="
echo " SUB-WP WP_STEP_EXIT=0.75m — Stages 3 + 4 — 3 seeds"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Duración estimada: ~30-36h"
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

# ─── STAGE 3 — Exit puro con WP_STEP_EXIT=0.75m ──────────────────────────
echo ""
echo "━━━ STAGE 3 — Exit puro (4M steps × 3 seeds) ━━━━━━━━━━━━━━━━━━━━━━━━"

run_webots "3_s42"  "Stage 3 wp75 — seed=42  (4M steps, exit puro)"
run_webots "3_s123" "Stage 3 wp75 — seed=123 (4M steps, exit puro)"
run_webots "3_s524" "Stage 3 wp75 — seed=524 (4M steps, exit puro)"

echo ""
echo "======================================================="
echo " Stage 3 completo — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos:"
echo "   pruebas/subwp_s42_wp75_stage3_final.zip"
echo "   pruebas/subwp_s123_wp75_stage3_final.zip"
echo "   pruebas/subwp_s524_wp75_stage3_final.zip"
echo "======================================================="

# ─── STAGE 4 — Ciclo completo con WP_STEP_EXIT=0.75m ─────────────────────
echo ""
echo "━━━ STAGE 4 — Ciclo completo (4M steps × 3 seeds) ━━━━━━━━━━━━━━━━━━━"

run_webots "4_s42"  "Stage 4 wp75 — seed=42  (4M steps, ciclo completo)"
run_webots "4_s123" "Stage 4 wp75 — seed=123 (4M steps, ciclo completo)"
run_webots "4_s524" "Stage 4 wp75 — seed=524 (4M steps, ciclo completo)"

echo ""
echo "======================================================="
echo " Stages 3 + 4 completos — $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " Stage 3 (exit puro, wp75):"
echo "   pruebas/subwp_s42_wp75_stage3_final.zip"
echo "   pruebas/subwp_s123_wp75_stage3_final.zip"
echo "   pruebas/subwp_s524_wp75_stage3_final.zip"
echo ""
echo " Stage 4 (ciclo completo, wp75):"
echo "   pruebas/subwp_s42_wp75_stage4_final.zip"
echo "   pruebas/subwp_s123_wp75_stage4_final.zip"
echo "   pruebas/subwp_s524_wp75_stage4_final.zip"
echo ""
echo " Siguiente paso: bash run_subwp_infer_wp75.sh"
echo "======================================================="
