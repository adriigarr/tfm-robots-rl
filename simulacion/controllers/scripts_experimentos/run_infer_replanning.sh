#!/bin/bash
# Inferencia con replanificación A* dinámica — 12 variantes (3 seeds × 2 modelos × 2 arquitecturas)
#
# Matriz completa:
#   STH-WP  v1 (ped_obs=False) + peatones congelados + replanning  → s42, s123, s524
#   STH-WP  v3 (ped_obs=True)  + peatones dinámicos  + replanning  → s42, s123, s524
#   SUB-WP  v1 (ped_obs=False) + peatones congelados + replanning  → s42, s123, s524
#   SUB-WP  v3 (ped_obs=True)  + peatones dinámicos  + replanning  → s42, s123, s524
#
# Pregunta clave:
#   v1+replan ≈ 100%  → replanning solo basta
#   v3+replan > base  → replanning mejora el mejor modelo actual

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

run_sthwp() {
    local key="$1" label="$2"
    echo ""
    echo "--- [STH-WP] $label ---"
    echo "$key" > "$STHWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLDS_DIR/warehouse_1.wbt"
    echo "[OK] $label"
}

run_subwp() {
    local key="$1" label="$2"
    echo ""
    echo "--- [SUB-WP] $label ---"
    echo "$key" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLDS_DIR/warehouse_1_subwp.wbt"
    echo "[OK] $label"
}

echo "======================================================="
echo " INFERENCIA REPLANNING — 12 variantes"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

# ── STH-WP v1 + peatones congelados + replanning ──────────────────────────
run_sthwp "infer_r3_s42_s6_static_replanning"  "v1 s42  + frozen peds + replan  (baseline 0.7%)"
run_sthwp "infer_r3_s123_s6_static_replanning" "v1 s123 + frozen peds + replan  (baseline 0.7%)"
run_sthwp "infer_r3_s524_s6_static_replanning" "v1 s524 + frozen peds + replan  (baseline 0.7%)"

# ── STH-WP v3 + peatones dinámicos + replanning ───────────────────────────
run_sthwp "infer_r3_s42_s6din_v3_replanning"  "v3 s42  + dynamic peds + replan (baseline 35.7%)"
run_sthwp "infer_r3_s123_s6din_v3_replanning" "v3 s123 + dynamic peds + replan (baseline 38.0%)"
run_sthwp "infer_r3_s524_s6din_v3_replanning" "v3 s524 + dynamic peds + replan (baseline 45.4%)"

# ── SUB-WP v1 + peatones congelados + replanning ──────────────────────────
run_subwp "infer_subwp_s42_s5_static_replanning"  "v1 s42  + frozen peds + replan  (baseline 15.5%)"
run_subwp "infer_subwp_s123_s5_static_replanning" "v1 s123 + frozen peds + replan"
run_subwp "infer_subwp_s524_s5_static_replanning" "v1 s524 + frozen peds + replan"

# ── SUB-WP v3 + peatones dinámicos + replanning ───────────────────────────
run_subwp "infer_din_v3_s42_replanning"  "v3 s42  + dynamic peds + replan (baseline 39.8%)"
run_subwp "infer_din_v3_s123_replanning" "v3 s123 + dynamic peds + replan (baseline 38.3%)"
run_subwp "infer_din_v3_s524_replanning" "v3 s524 + dynamic peds + replan"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs en:"
echo "   $STHWP_DIR/inferencia_sthwp/resultados/"
echo "   $SUBWP_DIR/inferencia_subwp/resultados/"
echo "======================================================="
