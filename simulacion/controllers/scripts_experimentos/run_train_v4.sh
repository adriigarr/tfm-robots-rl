#!/bin/bash
# Entrenamiento v4 — Replanning integrado en entorno | 6 variantes
#
# Matriz:
#   STH-WP  s42, s123, s524  — carga v3_final, 2M steps, guarda v4_final
#   SUB-WP  s42, s123, s524  — carga v3_final, 2M steps, guarda v4_final
#
# Novedad v4: webots_env.py ejecuta replanificación A* mid-episode cuando
# un peatón intercepta el segmento robot→subgoal (dist_perp < 0.6 m).

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

run_sthwp() {
    local key="$1" label="$2"
    echo ""
    echo "--- [STH-WP v4] $label ---"
    echo "$key" > "$STHWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLDS_DIR/warehouse_1.wbt"
    echo "[OK] $label"
}

run_subwp() {
    local key="$1" label="$2"
    echo ""
    echo "--- [SUB-WP v4] $label ---"
    echo "$key" > "$SUBWP_DIR/current_stage.txt"
    "$WEBOTS" --mode=fast --no-rendering --minimize "$WORLDS_DIR/warehouse_1_subwp.wbt"
    echo "[OK] $label"
}

echo "======================================================="
echo " ENTRENAMIENTO v4 — Replanning integrado en entorno"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

# ── STH-WP v4 ─────────────────────────────────────────────────────────────
run_sthwp "6_din_v4_s42"  "STH-WP s42  (desde v3_final, 2M steps)"
run_sthwp "6_din_v4_s123" "STH-WP s123 (desde v3_final, 2M steps)"
run_sthwp "6_din_v4_s524" "STH-WP s524 (desde v3_final, 2M steps)"

# ── SUB-WP v4 ─────────────────────────────────────────────────────────────
run_subwp "6_din_v4_s42"  "SUB-WP s42  (desde v3_final, 2M steps)"
run_subwp "6_din_v4_s123" "SUB-WP s123 (desde v3_final, 2M steps)"
run_subwp "6_din_v4_s524" "SUB-WP s524 (desde v3_final, 2M steps)"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Modelos guardados en:"
echo "   $STHWP_DIR/pruebas/*_stage6_din_v4_final.zip"
echo "   $SUBWP_DIR/pruebas/*_stage6_din_v4_final.zip"
echo "======================================================="
