#!/bin/bash
# Inferencia completa SUB-WP — Stages 1, 2, 3 y 4 — 3 seeds
#
# Lanza 12 ejecuciones de Webots en serie:
#   Stage 1:  100 ep × goal_01  × 3 seeds  (~1 min c/u)
#   Stage 2: 2800 ep × 28 goals × 3 seeds  (~40 min c/u)
#   Stage 3: 2800 ep × 28 goals × 3 seeds  (~30 min c/u, exit puro)
#   Stage 4: 2800 ep × 28 goals × 3 seeds  (~60 min c/u, ciclo completo)
#
# Duración estimada total: ~6-7 horas
#
# Uso: bash run_subwp_infer_all.sh
# Resultados: inferencia_subwp/resultados/infer_subwp_s*_stage*.csv

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"
RESULTS_DIR="$CONTROLLER_DIR/inferencia_subwp/resultados"

echo "======================================================="
echo " Inferencia SUB-WP — Stages 1 / 2 / 3 / 4 — 3 seeds"
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

# ─── STAGE 1 — Approach goal_01 (500 ep × 3 seeds) ───────────────────────────
echo ""
echo "━━━ STAGE 1 — Approach goal_01 (500 ep × 3 seeds) ━━━━━━━━━━━━━━━━━━━━━━"

run_stage "infer_1_s42"  "Stage 1 — seed=42  (500 ep, goal_01)"
run_stage "infer_1_s123" "Stage 1 — seed=123 (500 ep, goal_01)"
run_stage "infer_1_s524" "Stage 1 — seed=524 (500 ep, goal_01)"

# ─── STAGE 2 — Approach 28 goals (2800 ep × 3 seeds) ─────────────────────────
echo ""
echo "━━━ STAGE 2 — Approach 28 goals (2800 ep × 3 seeds) ━━━━━━━━━━━━━━━━━━━━"

run_stage "infer_2_s42"  "Stage 2 — seed=42  (2800 ep, approach, genera headings)"
run_stage "infer_2_s123" "Stage 2 — seed=123 (2800 ep, approach)"
run_stage "infer_2_s524" "Stage 2 — seed=524 (2800 ep, approach)"

if [ ! -f "$RESULTS_DIR/arrival_headings_stage2.json" ]; then
    echo "[WARN] arrival_headings_stage2.json no encontrado. Continuando de todos modos."
fi

# ─── STAGE 3 — Exit puro (2800 ep × 3 seeds) ─────────────────────────────────
echo ""
echo "━━━ STAGE 3 — Exit puro (2800 ep × 3 seeds) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_stage "infer_3_s42"  "Stage 3 — seed=42  (2800 ep, exit puro)"
run_stage "infer_3_s123" "Stage 3 — seed=123 (2800 ep, exit puro)"
run_stage "infer_3_s524" "Stage 3 — seed=524 (2800 ep, exit puro)"

# ─── STAGE 4 — Ciclo completo (2800 ep × 3 seeds) ────────────────────────────
echo ""
echo "━━━ STAGE 4 — Ciclo completo (2800 ep × 3 seeds) ━━━━━━━━━━━━━━━━━━━━━━━"

run_stage "infer_4_s42"  "Stage 4 — seed=42  (2800 ep, ciclo completo)"
run_stage "infer_4_s123" "Stage 4 — seed=123 (2800 ep, ciclo completo)"
run_stage "infer_4_s524" "Stage 4 — seed=524 (2800 ep, ciclo completo)"

# ─── RESUMEN ──────────────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo " Inferencia completa — $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " Stage 1:"
echo "   $RESULTS_DIR/infer_subwp_s42_stage1.csv"
echo "   $RESULTS_DIR/infer_subwp_s123_stage1.csv"
echo "   $RESULTS_DIR/infer_subwp_s524_stage1.csv"
echo ""
echo " Stage 2:"
echo "   $RESULTS_DIR/infer_subwp_s42_stage2.csv"
echo "   $RESULTS_DIR/infer_subwp_s123_stage2.csv"
echo "   $RESULTS_DIR/infer_subwp_s524_stage2.csv"
echo ""
echo " Stage 3:"
echo "   $RESULTS_DIR/infer_subwp_s42_stage3.csv"
echo "   $RESULTS_DIR/infer_subwp_s123_stage3.csv"
echo "   $RESULTS_DIR/infer_subwp_s524_stage3.csv"
echo ""
echo " Stage 4:"
echo "   $RESULTS_DIR/infer_subwp_s42_stage4.csv"
echo "   $RESULTS_DIR/infer_subwp_s123_stage4.csv"
echo "   $RESULTS_DIR/infer_subwp_s524_stage4.csv"
echo "======================================================="
