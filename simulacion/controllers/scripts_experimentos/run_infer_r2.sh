#!/bin/bash
# Inferencia r2 — SUB-WP con dropoff corregido (-11.0, 0.0)
#
# 5 etapas × 3 seeds = 15 ejecuciones secuenciales:
#   stage3_r2  (exit puro,       100ep × 28goals)
#   stage4_r2  (approach+exit,   100ep × 28goals)
#   stage5_r2  (return puro,     300ep)
#   din_v3_r2  (ciclo+peatones,  100ep × 28goals)
#   din_v4_r2  (ciclo+peat+repl, 100ep × 28goals)
#
# Stages 3-5: world sin peatones (igual que entrenamiento r2)
# Stages 6:   world con peatones activos

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"
WORLD_NOPED="$WORLDS_DIR/warehouse_1_subwp_noped.wbt"
WORLD_PED="$WORLDS_DIR/warehouse_1_subwp.wbt"

run_noped() {
    local key="$1" label="$2"
    echo ""
    echo "--- $label ---"
    echo "$key" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_NOPED"
    echo "[OK] $label"
}

run_ped() {
    local key="$1" label="$2"
    echo ""
    echo "--- $label ---"
    echo "$key" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLD_PED"
    echo "[OK] $label"
}

echo "======================================================="
echo " INFERENCIA r2 — SUB-WP dropoff corregido"
echo " Stages 3-5: sin peatones (warehouse_1_subwp_noped.wbt)"
echo " Stages 6:   con peatones (warehouse_1_subwp.wbt)"
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

# ── Stage 6 din v3: ciclo completo + peatones ─────────────────────
run_ped "infer_din_v3_r2_s42"  "Stage 6 din v3 r2 s42  — ciclo+peatones (100ep×28goals)"
run_ped "infer_din_v3_r2_s123" "Stage 6 din v3 r2 s123 — ciclo+peatones (100ep×28goals)"
run_ped "infer_din_v3_r2_s524" "Stage 6 din v3 r2 s524 — ciclo+peatones (100ep×28goals)"

# ── Stage 6 din v4: ciclo completo + peatones + replanning ────────
run_ped "infer_din_v4_r2_s42"  "Stage 6 din v4 r2 s42  — ciclo+peat+repl (100ep×28goals)"
run_ped "infer_din_v4_r2_s123" "Stage 6 din v4 r2 s123 — ciclo+peat+repl (100ep×28goals)"
run_ped "infer_din_v4_r2_s524" "Stage 6 din v4 r2 s524 — ciclo+peat+repl (100ep×28goals)"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs en $SUBWP_DIR/inferencia_subwp/resultados/"
echo "======================================================="
