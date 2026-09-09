#!/bin/bash
# Pipeline completo SUB-WP — 3 seeds × 4 stages + inferencia.
#
# Orden de ejecución:
#   Stage 1: s42 → s123 → s524   (3 × ~500k steps)
#   Stage 2: s42 → s123 → s524   (3 × ~4M steps)
#   Inferencia 2: s42 (mide headings) → s123 → s524
#   Stage 3: s42 → s123 → s524   (3 × ~4M steps)
#   Stage 4: s42 → s123 → s524   (3 × ~4M steps)
#   Inferencia 4: s42 → s123 → s524
#
# Duración estimada:
#   Stage 1:  3 × ~45min  = ~2.5h
#   Stage 2:  3 × ~5-6h   = ~16h
#   Infer 2:  3 × ~45min  = ~2.5h
#   Stage 3:  3 × ~5-6h   = ~16h
#   Stage 4:  3 × ~5-6h   = ~16h
#   Infer 4:  3 × ~50min  = ~2.5h
#   TOTAL:    ~55h (≈ 2-3 días)
#
# Uso: bash run_subwp_full.sh

set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
WORLD="$CONTROLLER_DIR/../../worlds/warehouse_1.wbt"
RESULTS_DIR="$CONTROLLER_DIR/inferencia_subwp/resultados"

echo "======================================================="
echo " Pipeline SUB-WP completo — 3 seeds"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
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

# ── Stage 1 ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━ STAGE 1: APPROACH goal_00 (3 seeds × 500k steps) ━━━"

run_webots "1_s42"  "Stage 1 — seed=42  (500k steps)"
run_webots "1_s123" "Stage 1 — seed=123 (500k steps)"
run_webots "1_s524" "Stage 1 — seed=524 (500k steps)"

echo ""
echo "━━━ STAGE 1 COMPLETO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Stage 2 ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━ STAGE 2: APPROACH todos los goals (3 seeds × 4M steps) ━━━"

run_webots "2_s42"  "Stage 2 — seed=42  (4M steps)"
run_webots "2_s123" "Stage 2 — seed=123 (4M steps)"
run_webots "2_s524" "Stage 2 — seed=524 (4M steps)"

echo ""
echo "━━━ STAGE 2 COMPLETO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Inferencia Stage 2 (heading measurement) ─────────────────────────────────
echo ""
echo "━━━ INFERENCIA STAGE 2 (3 seeds × 2800 ep) ━━━━━━━━━━━━━"
echo "    seed=42 genera arrival_headings_stage2.json"

run_webots "infer_2_s42"  "Inferencia stage 2 — seed=42  (2800 ep, genera headings)"
run_webots "infer_2_s123" "Inferencia stage 2 — seed=123 (2800 ep)"
run_webots "infer_2_s524" "Inferencia stage 2 — seed=524 (2800 ep)"

# Verificar que el JSON de headings se generó
if [ ! -f "$RESULTS_DIR/arrival_headings_stage2.json" ]; then
    echo "[ERROR] arrival_headings_stage2.json no se generó. Abortando."
    exit 1
fi
echo "[OK] arrival_headings_stage2.json disponible para stage 3."

# ── Stage 3 ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━ STAGE 3: EXIT PURO (3 seeds × 4M steps) ━━━━━━━━━━━━"

run_webots "3_s42"  "Stage 3 — seed=42  (4M steps)"
run_webots "3_s123" "Stage 3 — seed=123 (4M steps)"
run_webots "3_s524" "Stage 3 — seed=524 (4M steps)"

echo ""
echo "━━━ STAGE 3 COMPLETO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Stage 4 ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━ STAGE 4: CICLO COMPLETO (3 seeds × 4M steps) ━━━━━━━"

run_webots "4_s42"  "Stage 4 — seed=42  (4M steps)"
run_webots "4_s123" "Stage 4 — seed=123 (4M steps)"
run_webots "4_s524" "Stage 4 — seed=524 (4M steps)"

echo ""
echo "━━━ STAGE 4 COMPLETO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Inferencia Stage 4 ────────────────────────────────────────────────────────
echo ""
echo "━━━ INFERENCIA STAGE 4 (3 seeds × 2800 ep) ━━━━━━━━━━━━━"

run_webots "infer_4_s42"  "Inferencia stage 4 — seed=42  (2800 ep)"
run_webots "infer_4_s123" "Inferencia stage 4 — seed=123 (2800 ep)"
run_webots "infer_4_s524" "Inferencia stage 4 — seed=524 (2800 ep)"

echo ""
echo "======================================================="
echo " PIPELINE SUB-WP COMPLETO"
echo " Fin: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " Modelos finales:"
echo "   pruebas/subwp_s42_stage4_final.zip"
echo "   pruebas/subwp_s123_stage4_final.zip"
echo "   pruebas/subwp_s524_stage4_final.zip"
echo ""
echo " CSVs de inferencia stage 4:"
echo "   $RESULTS_DIR/infer_subwp_s42_stage4.csv"
echo "   $RESULTS_DIR/infer_subwp_s123_stage4.csv"
echo "   $RESULTS_DIR/infer_subwp_s524_stage4.csv"
echo "======================================================="
