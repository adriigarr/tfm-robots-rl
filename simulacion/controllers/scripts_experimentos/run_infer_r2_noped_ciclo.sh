#!/bin/bash
# Inferencia r2 — Ciclo completo SIN peatones físicos (din_v4_r2 × 3 seeds)
#
# Usa warehouse_1_subwp_noped.wbt con el modelo din_v4_r2 (48-dim, ped_obs=True).
# Los peatones existen como nodos pero no tienen física ni controlador activo.
# Propósito: aislar la capacidad del ciclo completo del efecto de bloqueo físico
# de los peatones → compara con §30 (din_v4 con peatones activos, 10.5% media).

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
echo " INFERENCIA r2 — Ciclo completo SIN peatones físicos"
echo " Modelo: din_v4_r2_final (48-dim, replanning)"
echo " World:  warehouse_1_subwp_noped.wbt"
echo " 3 seeds × 100ep × 28goals = 8.400 ciclos total"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

run_noped "infer_din_v4_r2_noped_s42"  "Din v4 r2 NOPED s42  — ciclo completo (100ep×28goals)"
run_noped "infer_din_v4_r2_noped_s123" "Din v4 r2 NOPED s123 — ciclo completo (100ep×28goals)"
run_noped "infer_din_v4_r2_noped_s524" "Din v4 r2 NOPED s524 — ciclo completo (100ep×28goals)"

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs en $SUBWP_DIR/inferencia_subwp/resultados/"
echo "   infer_din_v4_r2_noped_s{42,123,524}.csv"
echo "======================================================="
