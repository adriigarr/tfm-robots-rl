#!/bin/bash
# Stage 4 v2 — Entrenamiento (3 seeds) + Inferencia (3 seeds) secuenciales.
#
# Base: run002_s*_stage3v2_final (headings reales + reward marcha atrás)
# Tarea: ciclo completo approach→exit encadenado (stage=4)
#
# Orden de ejecución:
#   1. Entrenamiento s42  → pruebas/run002_s42_stage4v2_final.zip
#   2. Entrenamiento s123 → pruebas/run002_s123_stage4v2_final.zip
#   3. Entrenamiento s524 → pruebas/run002_s524_stage4v2_final.zip
#   4. Inferencia s42     → inferencia_sthwp/resultados/infer_run002_s42_stage4v2.csv
#   5. Inferencia s123    → inferencia_sthwp/resultados/infer_run002_s123_stage4v2.csv
#   6. Inferencia s524    → inferencia_sthwp/resultados/infer_run002_s524_stage4v2.csv
#
# Duración estimada:
#   Entrenamiento: ~3 × 5-6h = 15-18h
#   Inferencia:    ~3 × 45min = ~2.5h
#   Total:         ~18-20h
#
# Uso: bash run_stage4v2_full.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"
RESULTS_DIR="$CONTROLLER_DIR/inferencia_sthwp/resultados"

echo "======================================================="
echo " Stage 4 v2 — Entrenamiento + Inferencia (3 seeds)"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Base:   run002_s*_stage3v2_final"
echo "======================================================="

mkdir -p "$RESULTS_DIR"

run_webots() {
    local KEY="$1"
    local LABEL="$2"
    echo ""
    echo "------- $LABEL"
    echo "        $(date '+%Y-%m-%d %H:%M:%S') -------"
    echo "$KEY" > "$STAGE_FILE"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    local CODE=$?
    if [ $CODE -ne 0 ]; then
        echo ""
        echo "[ERROR] Webots terminó con código $CODE en: $KEY"
        echo "        Abortando pipeline."
        exit 1
    fi
    echo "[OK] $LABEL — $(date '+%H:%M:%S')"
}

# ── Fase 1: Entrenamientos ────────────────────────────────────────────────────
echo ""
echo "━━━ FASE 1: ENTRENAMIENTO (3 seeds × 4M steps) ━━━━━━━━━"

run_webots "4v2_s42"  "Entrenamiento stage 4 v2 — seed=42  (4M steps)"
run_webots "4v2_s123" "Entrenamiento stage 4 v2 — seed=123 (4M steps)"
run_webots "4v2_s524" "Entrenamiento stage 4 v2 — seed=524 (4M steps)"

echo ""
echo "━━━ FASE 1 COMPLETA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Modelos guardados:"
echo "    pruebas/run002_s42_stage4v2_final.zip"
echo "    pruebas/run002_s123_stage4v2_final.zip"
echo "    pruebas/run002_s524_stage4v2_final.zip"

# ── Fase 2: Inferencia ────────────────────────────────────────────────────────
echo ""
echo "━━━ FASE 2: INFERENCIA (3 seeds × 2800 ep) ━━━━━━━━━━━━━"

run_webots "infer_r2_s42_s4v2"  "Inferencia stage 4 v2 — seed=42  (2800 ep)"
run_webots "infer_r2_s123_s4v2" "Inferencia stage 4 v2 — seed=123 (2800 ep)"
run_webots "infer_r2_s524_s4v2" "Inferencia stage 4 v2 — seed=524 (2800 ep)"

echo ""
echo "======================================================="
echo " PIPELINE COMPLETO"
echo " Fin: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " CSVs de inferencia:"
echo "   $RESULTS_DIR/infer_run002_s42_stage4v2.csv"
echo "   $RESULTS_DIR/infer_run002_s123_stage4v2.csv"
echo "   $RESULTS_DIR/infer_run002_s524_stage4v2.csv"
echo "======================================================="
