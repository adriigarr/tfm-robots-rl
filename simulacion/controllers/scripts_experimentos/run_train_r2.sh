#!/bin/bash
# Reentrenamiento SUB-WP r2 — dropoff corregido (-11.0, 0.0)
#
# Cadena completa por seed:
#   stage3_r2 (4M) → stage4_r2 (4M) → stage5_r2 (2M)
#   → stage6_din_v3_r2 (2M) → stage6_din_v4_r2 (2M)
#
# Total por seed: ~14M steps
# Total global:   ~42M steps (3 seeds secuenciales)
# Tiempo estimado: ~36–48h

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"
WORLD_NOPED="$WORLDS_DIR/warehouse_1_subwp_noped.wbt"   # stages 3-5: sin peatones físicos
WORLD_PED="$WORLDS_DIR/warehouse_1_subwp.wbt"            # stages 6: peatones activos

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
echo " REENTRENAMIENTO SUB-WP r2 — dropoff corregido"
echo " Stages 3-5: sin peatones (warehouse_1_subwp_noped.wbt)"
echo " Stages 6:   con peatones (warehouse_1_subwp.wbt)"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

# ── seed 42 — stages 3/4/5 ya completados, arranca en stage 6 ────────────
run_ped   "6_din_v3_r2_s42" "Stage 6 v3 r2 s42 — peatones+B1 (2M steps)"
run_ped   "6_din_v4_r2_s42" "Stage 6 v4 r2 s42 — replanning  (2M steps)"

# ── seed 123 ──────────────────────────────────────────────────────────────
run_noped "3_r2_s123"        "Stage 3 r2 s123 — exit puro      (4M steps, sin peatones)"
run_noped "4_r2_s123"        "Stage 4 r2 s123 — approach+exit  (4M steps, sin peatones)"
run_noped "5_r2_s123"        "Stage 5 r2 s123 — return puro    (2M steps, sin peatones)"
run_ped   "6_din_v3_r2_s123" "Stage 6 v3 r2 s123 — peatones+B1 (2M steps)"
run_ped   "6_din_v4_r2_s123" "Stage 6 v4 r2 s123 — replanning  (2M steps)"

# ── seed 524 ──────────────────────────────────────────────────────────────
run_noped "3_r2_s524"        "Stage 3 r2 s524 — exit puro      (4M steps, sin peatones)"
run_noped "4_r2_s524"        "Stage 4 r2 s524 — approach+exit  (4M steps, sin peatones)"
run_noped "5_r2_s524"        "Stage 5 r2 s524 — return puro    (2M steps, sin peatones)"
run_ped   "6_din_v3_r2_s524" "Stage 6 v3 r2 s524 — peatones+B1 (2M steps)"
run_ped   "6_din_v4_r2_s524" "Stage 6 v4 r2 s524 — replanning  (2M steps)"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos finales en $SUBWP_DIR/pruebas/"
echo "   subwp_s{42,123,524}_wp75_r2_stage6_din_v3_final.zip"
echo "   subwp_s{42,123,524}_wp75_r2_stage6_din_v4_final.zip"
echo "======================================================="
