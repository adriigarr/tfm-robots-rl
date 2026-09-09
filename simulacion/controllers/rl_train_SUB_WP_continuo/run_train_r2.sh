#!/usr/bin/env bash
# run_train_r2.sh — Entrenamiento r2 completo (dropoff corregido: -11.0, 0.0)
# 5 etapas × 3 seeds = 15 ejecuciones secuenciales
# Pasos totales por seed: 4M + 4M + 2M + 2M + 2M = 14M (~36-48h total)
# Uso: bash run_train_r2.sh
set -euo pipefail

CONTROLLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_FILE="$CONTROLLER_DIR/current_stage.txt"

run_stage() {
    local key="$1"
    echo ""
    echo "============================================================"
    echo "  Entrenamiento r2: $key"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo "$key" > "$STAGE_FILE"
    python3 "$CONTROLLER_DIR/rl_train_SUB_WP_continuo.py"
}

# ── Stage 3: exit puro (4M steps, desde stage2_final) ────────────
run_stage "3_r2_s42"
run_stage "3_r2_s123"
run_stage "3_r2_s524"

# ── Stage 4: approach+exit (4M steps, desde stage3_r2_final) ─────
run_stage "4_r2_s42"
run_stage "4_r2_s123"
run_stage "4_r2_s524"

# ── Stage 5: return puro (2M steps, desde stage4_r2_final) ───────
run_stage "5_r2_s42"
run_stage "5_r2_s123"
run_stage "5_r2_s524"

# ── Stage 6 din v3: peatones + B1 (2M steps, desde stage5_r2_final) ──
run_stage "6_din_v3_r2_s42"
run_stage "6_din_v3_r2_s123"
run_stage "6_din_v3_r2_s524"

# ── Stage 6 din v4: replanning (2M steps, desde stage6_din_v3_r2_final) ──
run_stage "6_din_v4_r2_s42"
run_stage "6_din_v4_r2_s123"
run_stage "6_din_v4_r2_s524"

echo ""
echo "============================================================"
echo "  Entrenamiento r2 COMPLETADO"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Modelos en: $CONTROLLER_DIR/pruebas/"
echo "============================================================"
