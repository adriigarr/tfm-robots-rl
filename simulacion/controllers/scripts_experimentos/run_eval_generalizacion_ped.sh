#!/bin/bash
# Evaluación de generalización — Fase 2 (con peatón) — WH02-05
# Modelo E2.1 (mejor de la serie), 3 seeds, STHWP + SUBWP, 10 repeticiones/goal.
# 4 almacenes x 2 métodos x 3 seeds = 24 lanzamientos, 10 episodios por goal
# (1360 goals x 10 x 3 x 2 = 81.600 episodios en total).
#
# Por qué 10 repeticiones y no 1 (a diferencia de la fase 1): el peatón se
# aleatoriza de verdad en cada reset() cuando ped_obs=True
# (_randomize_pedestrians() en webots_env.py usa self.np_random.uniform sobre
# su corredor), así que cada repetición de un mismo goal es un episodio
# genuinamente distinto — a diferencia de la fase sin peatón, donde el
# simulador era determinista y repetir no aportaba nada.
#
# max_steps: se reutilizan los mismos valores calibrados en la fase 1 (el
# peatón no cambia el tamaño del almacén ni el recorrido A*, y el baseline
# real de WH01 (E2.1) usaba el mismo max_steps con y sin peatón).
#
# Fix de rango del peatón: _ped_ranges en webots_env.py está hardcodeado con
# las coordenadas del corredor de WH01 — no vale para WH02-05. El fix vive
# en los propios scripts *_eval_generalizacion.py (tabla PED_RANGES leída de
# los --trajectory reales de cada warehouse_ped_0{2..5}.wbt), no aquí.
#
# FORCE=1 ./run_eval_generalizacion_ped.sh para reevaluar combinaciones cuyo
# CSV ya existe (por defecto se saltan).
set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
OUT_DIR="$CONTROLLERS_DIR/../../almacenes/resultados_evaluacion"

STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

FORCE="${FORCE:-0}"
N_REPS="${N_REPS:-10}"

mkdir -p "$OUT_DIR"

WAREHOUSES=(02 03 04 05)
SEEDS=(42 123 524)

# bash 3.2 (el que trae macOS por defecto) no soporta arrays asociativos
# (declare -A), así que se resuelve el max_steps por almacén con un case.
# Mismos valores que la fase 1 (ver run_eval_generalizacion.sh).
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
    local out_csv="$OUT_DIR/${metodo}_ped_wh${wh}_s${seed}.csv"

    if [[ -f "$out_csv" && "$FORCE" != "1" ]]; then
        echo "--- $metodo WH$wh seed=$seed (ped): ya existe, se salta ($out_csv) ---"
        return
    fi

    echo ""
    echo "--- $metodo WH$wh seed=$seed CON PEATÓN (max_steps=$max_steps, $N_REPS reps/goal) ---"
    echo "eval_gen" > "$stage_dir/current_stage.txt"
    EVAL_WH="$wh" EVAL_SEED="$seed" EVAL_OUT_CSV="$out_csv" EVAL_MAX_STEPS="$max_steps" \
    EVAL_PED="1" EVAL_N_REPS="$N_REPS" \
        "$WEBOTS" --mode=fast --no-rendering --minimize "$world"
    echo "[OK] $metodo WH$wh seed=$seed (ped) -> $out_csv"
}

echo "======================================================="
echo " EVALUACIÓN DE GENERALIZACIÓN — Fase 2 (con peatón)"
echo " $N_REPS repeticiones por goal"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

for wh in "${WAREHOUSES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_eval "sthwp" "$wh" "$seed" "$STHWP_DIR" "$WORLDS_DIR/warehouse_ped_$wh.wbt" "$(max_steps_sthwp "$wh")"
    done
done

for wh in "${WAREHOUSES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_eval "subwp" "$wh" "$seed" "$SUBWP_DIR" "$WORLDS_DIR/warehouse_ped_${wh}_subwp.wbt" "$(max_steps_subwp "$wh")"
    done
done

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs en $OUT_DIR"
echo "======================================================="
