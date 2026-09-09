#!/bin/bash
# Re-inferencia r2 stages 3-5 — world SIN peatones (fix consistencia con entrenamiento)
#
# Relanza únicamente stages 3/4/5 (9 ejecuciones) con warehouse_1_subwp_noped.wbt.
# Las inferencias de stage 6 (din_v3/v4) ya son correctas y no se repiten.

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"
WORLD_NOPED="$WORLDS_DIR/warehouse_1_subwp_noped.wbt"

run_noped() {
    local key="$1" label="$2"
    echo ""
    echo "--- $label ---"
    echo "$key" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_NOPED"
    echo "[OK] $label"
}

echo "======================================================="
echo " RE-INFERENCIA r2 stages 3-5 (sin peatones)"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

# ── Stage 3: exit puro ────────────────────────────────────────────
run_noped "infer_3_r2_s42"  "Stage 3 r2 s42  — exit puro (100ep×28goals)"
run_noped "infer_3_r2_s123" "Stage 3 r2 s123 — exit puro (100ep×28goals)"
run_noped "infer_3_r2_s524" "Stage 3 r2 s524 — exit puro (100ep×28goals)"

# ── Stage 4: approach + exit ──────────────────────────────────────
run_noped "infer_4_r2_s42"  "Stage 4 r2 s42  — approach+exit (100ep×28goals)"
run_noped "infer_4_r2_s123" "Stage 4 r2 s123 — approach+exit (100ep×28goals)"
run_noped "infer_4_r2_s524" "Stage 4 r2 s524 — approach+exit (100ep×28goals)"

# ── Stage 5: return puro ──────────────────────────────────────────
run_noped "infer_5_r2_s42"  "Stage 5 r2 s42  — return puro (300ep)"
run_noped "infer_5_r2_s123" "Stage 5 r2 s123 — return puro (300ep)"
run_noped "infer_5_r2_s524" "Stage 5 r2 s524 — return puro (300ep)"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs en $SUBWP_DIR/inferencia_subwp/resultados/"
echo "======================================================="
