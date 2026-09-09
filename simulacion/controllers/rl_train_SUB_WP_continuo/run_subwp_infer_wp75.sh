#!/bin/bash
# Inferencia SUB-WP wp75 — solo stages 3 y 4 — 3 seeds
#
# Stages 1 y 2 no se reinferencian (approach no cambió con WP_STEP_EXIT=0.75m)
#
#   Stage 3: 2800 ep × 28 goals × 3 seeds  (~30 min c/u, exit puro wp75)
#   Stage 4: 2800 ep × 28 goals × 3 seeds  (~60 min c/u, ciclo completo wp75)
#
# Duración estimada: ~4-5 horas
#
# Salidas (no sobreescriben nada anterior):
#   inferencia_subwp/resultados/infer_subwp_s*_wp75_stage3.csv
#   inferencia_subwp/resultados/infer_subwp_s*_wp75_stage4.csv
#
# Uso: bash run_subwp_infer_wp75.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"
RESULTS_DIR="$CONTROLLER_DIR/inferencia_subwp/resultados"

echo "======================================================="
echo " Inferencia SUB-WP wp75 — Stages 3 y 4 — 3 seeds"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Resultados: $RESULTS_DIR/"
echo "======================================================="

mkdir -p "$RESULTS_DIR"

run_stage() {
    local KEY="$1"
    local LABEL="$2"
    echo ""
    echo "-------------------------------------------------------"
    echo " Lanzando: $LABEL"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "-------------------------------------------------------"
    echo "$KEY" > "$STAGE_FILE"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD"
    local EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Webots terminó con código $EXIT_CODE en: $LABEL"
        exit 1
    fi
    echo " OK: $LABEL — $(date '+%H:%M:%S')"
}

# ─── STAGE 3 — Exit puro wp75 (2800 ep × 3 seeds) ────────────────────────
echo ""
echo "━━━ STAGE 3 wp75 — Exit puro (2800 ep × 3 seeds) ━━━━━━━━━━━━━━━━━━━━"

run_stage "infer_3_s42"  "Stage 3 wp75 — seed=42  (2800 ep, exit puro)"
run_stage "infer_3_s123" "Stage 3 wp75 — seed=123 (2800 ep, exit puro)"
run_stage "infer_3_s524" "Stage 3 wp75 — seed=524 (2800 ep, exit puro)"

# ─── STAGE 4 — Ciclo completo wp75 (2800 ep × 3 seeds) ───────────────────
echo ""
echo "━━━ STAGE 4 wp75 — Ciclo completo (2800 ep × 3 seeds) ━━━━━━━━━━━━━━━"

run_stage "infer_4_s42"  "Stage 4 wp75 — seed=42  (2800 ep, ciclo completo)"
run_stage "infer_4_s123" "Stage 4 wp75 — seed=123 (2800 ep, ciclo completo)"
run_stage "infer_4_s524" "Stage 4 wp75 — seed=524 (2800 ep, ciclo completo)"

# ─── RESUMEN ──────────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo " Inferencia wp75 completa — $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " Stage 3 (exit puro, wp75):"
echo "   $RESULTS_DIR/infer_subwp_s42_wp75_stage3.csv"
echo "   $RESULTS_DIR/infer_subwp_s123_wp75_stage3.csv"
echo "   $RESULTS_DIR/infer_subwp_s524_wp75_stage3.csv"
echo ""
echo " Stage 4 (ciclo completo, wp75):"
echo "   $RESULTS_DIR/infer_subwp_s42_wp75_stage4.csv"
echo "   $RESULTS_DIR/infer_subwp_s123_wp75_stage4.csv"
echo "   $RESULTS_DIR/infer_subwp_s524_wp75_stage4.csv"
echo "======================================================="
