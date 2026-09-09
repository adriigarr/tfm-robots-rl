#!/bin/bash
# Diagnóstico: modelo v1 estático (ped_obs=False) con peatones físicamente presentes pero congelados
#
# Pregunta: ¿los peatones como obstáculos físicos estáticos bloquean la ruta A*,
# o el modelo v1 (ped_obs=False) sigue siendo ~100% aunque estén en el mundo?
#
# Interpretación:
#   ~100% → los peatones no bloquean la ruta; la caída al 45% en v3 es por ped_obs=True
#   ~45%  → los peatones físicamente están en el camino A*; problema geométrico
#   intermedio → impacto mixto
#
# Cadena de diagnóstico completa:
#   v1 sin peatones:              ~100%
#   v1 ped_obs=False, ped. fijos: ???%   ← ESTE SCRIPT
#   v3 ped_obs=True,  ped. fijos:  45.4% (STH) / 38.0% (SUB)
#   v3 ped_obs=True,  ped. móviles: 45.4% (STH) / 39.8% (SUB)
#
# Uso: bash run_infer_diagnostic_pedpresent.sh

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"

STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

echo "======================================================="
echo " DIAGNÓSTICO — modelo v1 (ped_obs=False) + peatones presentes"
echo " STH-WP s524 stage6_final  →  dinámico v3: 45.4%"
echo " SUB-WP s42  stage5_final  →  dinámico v3: 39.8%"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

echo ""
echo "--- [STH-WP] s524 v1 ped_obs=False + peatones estáticos ---"
echo "infer_r3_s524_s6_pedpresent" > "$STHWP_DIR/current_stage.txt"
"$WEBOTS" --mode=fast --no-rendering --minimize "$WORLDS_DIR/warehouse_1.wbt"
echo "[OK] STH-WP s524"

echo ""
echo "--- [SUB-WP] s42 v1 ped_obs=False + peatones estáticos ---"
echo "infer_subwp_s42_s5_pedpresent" > "$SUBWP_DIR/current_stage.txt"
"$WEBOTS" --mode=fast --no-rendering --minimize "$WORLDS_DIR/warehouse_1_subwp.wbt"
echo "[OK] SUB-WP s42"

echo ""
echo "======================================================="
echo " DIAGNÓSTICO COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs:"
echo "   $STHWP_DIR/inferencia_sthwp/resultados/infer_run003_s524_stage6_static_pedpresent.csv"
echo "   $SUBWP_DIR/inferencia_subwp/resultados/infer_subwp_s42_stage5_pedpresent.csv"
echo "======================================================="
