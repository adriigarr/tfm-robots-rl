#!/usr/bin/env bash
# run_infer_r2.sh — Inferencia de todos los modelos r2 (dropoff corregido)
# Stages: 3 (exit), 4 (approach+exit), 5 (return), din_v3, din_v4
# Seeds: s42, s123, s524  →  15 ejecuciones en total
# Uso: bash run_infer_r2.sh
set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"

run_stage() {
    local key="$1"
    echo ""
    echo "============================================================"
    echo "  Inferencia r2: $key"
    echo "============================================================"
    echo "$key" > "$STAGE_FILE"
    # El dispatcher de Webots leerá current_stage.txt y lanzará el script
    # correspondiente. Aquí lo ejecutamos directamente para CLI:
    python3 "$CONTROLLER_DIR/rl_train_SUB_WP_continuo.py" || true
}

# ── Stage 3: exit puro ────────────────────────────────────────────
run_stage "infer_3_r2_s42"
run_stage "infer_3_r2_s123"
run_stage "infer_3_r2_s524"

# ── Stage 4: approach + exit ──────────────────────────────────────
run_stage "infer_4_r2_s42"
run_stage "infer_4_r2_s123"
run_stage "infer_4_r2_s524"

# ── Stage 5: return puro ──────────────────────────────────────────
run_stage "infer_5_r2_s42"
run_stage "infer_5_r2_s123"
run_stage "infer_5_r2_s524"

# ── Stage 6 din v3: ciclo completo + peatones ─────────────────────
run_stage "infer_din_v3_r2_s42"
run_stage "infer_din_v3_r2_s123"
run_stage "infer_din_v3_r2_s524"

# ── Stage 6 din v4: ciclo completo + peatones + replanning ────────
run_stage "infer_din_v4_r2_s42"
run_stage "infer_din_v4_r2_s123"
run_stage "infer_din_v4_r2_s524"

echo ""
echo "============================================================"
echo "  Inferencia r2 completada (15 ejecuciones)"
echo "  Resultados en: $CONTROLLER_DIR/inferencia_subwp/resultados/"
echo "============================================================"
