#!/bin/bash
# Stage 5 — Entrenamiento (3 seeds) + Inferencia (3 seeds) secuenciales.
#
# Base: run002_s*_stage4v2_final (ciclo approach+exit consolidado)
# Tarea: retorno puro descarga→zona_espera (stage=5)
#
# Fix clave: heading_utils.py usa eje Z [0,0,1,-θ] (antes Y [0,1,0,-θ]).
# Para θ≈-108° (heading SW del retorno) el eje Y producía pitch de 108°
# → el bumper penetraba el suelo 0.29m → colisión en step 1.
#
# Orden de ejecución:
#   1. Entrenamiento s42  → pruebas/run002_s42_stage5_final.zip
#   2. Entrenamiento s123 → pruebas/run002_s123_stage5_final.zip
#   3. Entrenamiento s524 → pruebas/run002_s524_stage5_final.zip
#   4. Inferencia s42     → inferencia_sthwp/resultados/infer_run002_s42_stage5.csv
#   5. Inferencia s123    → inferencia_sthwp/resultados/infer_run002_s123_stage5.csv
#   6. Inferencia s524    → inferencia_sthwp/resultados/infer_run002_s524_stage5.csv
#
# Duración estimada:
#   Entrenamiento: ~3 × 2-3h = 6-9h
#   Inferencia:    ~3 × 30min = ~1.5h
#   Total:         ~8-11h
#
# Uso: bash run_stage5_seeds.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"
RESULTS_DIR="$CONTROLLER_DIR/inferencia_sthwp/resultados"

echo "======================================================="
echo " Stage 5 STH-WP — Entrenamiento + Inferencia (3 seeds)"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Base:   run002_s*_stage4v2_final"
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
echo "━━━ FASE 1: ENTRENAMIENTO (3 seeds × 2M steps) ━━━━━━━━━"

run_webots "5_s42"  "Entrenamiento stage 5 — seed=42  (2M steps)"
run_webots "5_s123" "Entrenamiento stage 5 — seed=123 (2M steps)"
run_webots "5_s524" "Entrenamiento stage 5 — seed=524 (2M steps)"

echo ""
echo "━━━ FASE 1 COMPLETA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Modelos guardados:"
echo "    pruebas/run002_s42_stage5_final.zip"
echo "    pruebas/run002_s123_stage5_final.zip"
echo "    pruebas/run002_s524_stage5_final.zip"

# ── Fase 2: Inferencia ────────────────────────────────────────────────────────
echo ""
echo "━━━ FASE 2: INFERENCIA RETORNO (3 seeds × 300 ep) ━━━━━━━"

run_webots "infer_r2_s42_s5"  "Inferencia stage 5 — seed=42  (300 ep)"
run_webots "infer_r2_s123_s5" "Inferencia stage 5 — seed=123 (300 ep)"
run_webots "infer_r2_s524_s5" "Inferencia stage 5 — seed=524 (300 ep)"

echo ""
echo "======================================================="
echo " PIPELINE COMPLETO"
echo " Fin: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " CSVs de inferencia:"
echo "   $RESULTS_DIR/infer_run002_s42_stage5.csv"
echo "   $RESULTS_DIR/infer_run002_s123_stage5.csv"
echo "   $RESULTS_DIR/infer_run002_s524_stage5.csv"
echo "======================================================="
