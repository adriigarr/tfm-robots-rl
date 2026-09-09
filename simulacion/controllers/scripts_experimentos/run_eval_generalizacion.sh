#!/bin/bash
# Evaluación de generalización — Fase 1 (sin peatón) — WH02-05
# Modelo E2.1 (mejor de la serie), 3 seeds, STHWP + SUBWP
# 4 almacenes x 2 métodos x 3 seeds = 24 lanzamientos, 1 episodio por goal.
#
# max_steps por almacén: calibrado a partir de la longitud real del recorrido
# A* completo (estantería -> descarga -> espera) del goal más exigente de cada
# almacén, manteniendo el mismo margen de seguridad relativo (~2x) que se usó
# originalmente para calibrar WH01 (7000 STHWP / 6000 SUBWP sobre 56.6m).
# Sin esto, WH04 y WH05 truncan todos los episodios antes de completar el
# ciclo de 3 tramos (confirmado: reward alto y pasos=límite exacto en el
# intento anterior, no es un fallo real de la política).
#
# FORCE=1 ./run_eval_generalizacion.sh para reevaluar combinaciones cuyo CSV
# ya existe (por defecto se saltan, para no repetir WH02/WH03 STHWP que ya
# corrieron correctamente con el max_steps antiguo).
set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
OUT_DIR="$CONTROLLERS_DIR/../../almacenes/resultados_evaluacion"

STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

FORCE="${FORCE:-0}"

mkdir -p "$OUT_DIR"

WAREHOUSES=(02 03 04 05)
SEEDS=(42 123 524)

# bash 3.2 (el que trae macOS por defecto) no soporta arrays asociativos
# (declare -A), así que se resuelve el max_steps por almacén con un case.
max_steps_sthwp() {
    case "$1" in
        02) echo 11600 ;;
        03) echo 13300 ;;
        04) echo 22400 ;;
        05) echo 36600 ;;
    esac
}

max_steps_subwp() {
    case "$1" in
        02) echo 9900 ;;
        03) echo 11400 ;;
        04) echo 19200 ;;
        05) echo 31400 ;;
    esac
}

run_eval() {
    local metodo="$1" wh="$2" seed="$3" stage_dir="$4" world="$5" max_steps="$6"
    local out_csv="$OUT_DIR/${metodo}_wh${wh}_s${seed}.csv"

    if [[ -f "$out_csv" && "$FORCE" != "1" ]]; then
        echo "--- $metodo WH$wh seed=$seed: ya existe, se salta ($out_csv) ---"
        return
    fi

    echo ""
    echo "--- $metodo WH$wh seed=$seed (max_steps=$max_steps) ---"
    echo "eval_gen" > "$stage_dir/current_stage.txt"
    EVAL_WH="$wh" EVAL_SEED="$seed" EVAL_OUT_CSV="$out_csv" EVAL_MAX_STEPS="$max_steps" \
        "$WEBOTS" --mode=fast --no-rendering --minimize "$world"
    echo "[OK] $metodo WH$wh seed=$seed -> $out_csv"
}

echo "======================================================="
echo " EVALUACIÓN DE GENERALIZACIÓN — Fase 1 (sin peatón)"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

for wh in "${WAREHOUSES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_eval "sthwp" "$wh" "$seed" "$STHWP_DIR" "$WORLDS_DIR/warehouse_$wh.wbt" "$(max_steps_sthwp "$wh")"
    done
done

for wh in "${WAREHOUSES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_eval "subwp" "$wh" "$seed" "$SUBWP_DIR" "$WORLDS_DIR/warehouse_${wh}_subwp.wbt" "$(max_steps_subwp "$wh")"
    done
done

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs en $OUT_DIR"
echo "======================================================="
