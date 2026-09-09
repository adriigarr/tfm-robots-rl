#!/bin/bash
# Inferencia diagnóstica — peatones completamente estáticos (congelados en cada step)
#
# Modelos: STH-WP s524 v3 (ckpt 2.8M) y SUB-WP s42 v3 (final)
#
# Pregunta: ¿cuánto del 55% de fallos viene del MOVIMIENTO del peatón
# vs de la reformulación de política por B1?
#
# Interpretación esperada:
#   ~100% → toda la caída es por movimiento del peatón (buen diagnóstico)
#   ~45%  → el modelo B1 ha alterado la política base (problema de reward)
#   valor intermedio → impacto mixto
#
# Uso: bash run_infer_diagnostic_staticped.sh

set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"

STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

echo "======================================================="
echo " DIAGNÓSTICO — peatones completamente estáticos"
echo " STH-WP s524 v3 (ckpt 2.8M)  →  dinámico: 45.4%"
echo " SUB-WP s42  v3 (final)       →  dinámico: 39.8%"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

echo ""
echo "--- [STH-WP] s524 v3 peatones estáticos ---"
echo "infer_r3_s524_s6din_v3_staticped" > "$STHWP_DIR/current_stage.txt"
"$WEBOTS" --mode=fast --no-rendering --minimize "$WORLDS_DIR/warehouse_1.wbt"
echo "[OK] STH-WP s524 staticped"

echo ""
echo "--- [SUB-WP] s42 v3 peatones estáticos ---"
echo "infer_din_v3_s42_staticped" > "$SUBWP_DIR/current_stage.txt"
"$WEBOTS" --mode=fast --no-rendering --minimize "$WORLDS_DIR/warehouse_1_subwp.wbt"
echo "[OK] SUB-WP s42 staticped"

echo ""
echo "======================================================="
echo " DIAGNÓSTICO COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs:"
echo "   $STHWP_DIR/inferencia_sthwp/resultados/infer_run003_s524_stage6_din_v3_staticped.csv"
echo "   $SUBWP_DIR/inferencia_subwp/resultados/infer_din_v3_s42_staticped.csv"
echo "======================================================="
