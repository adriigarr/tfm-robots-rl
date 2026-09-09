#!/bin/bash
# Evaluación E2.1-R0 — mismos modelos E2.1 de fase 2 (con peatón), pero con la
# replanificación dinámica LIDAR desactivada (env.enable_dynamic_replanning=False).
# Se compara después, de forma AGREGADA (no episodio a episodio — ver
# E2_1_R0_README.md), con la evaluación ya existente E2.1-R1 (replanning activo,
# resultados_evaluacion/*_ped_wh*_s*.csv).
#
# No reentrena ni modifica ningún modelo. No toca los CSV de E2.1-R1.
#
# Salida: almacenes/resultados_evaluacion/fase2_e2_1_r0_sin_replanificacion/
#         {sthwp|subwp}_ped_e2_1_r0_wh{02..05}_s{42|123|524}.csv
#
# NOTA IMPORTANTE sobre "argumentos CLI": Webots lanza el controller
# directamente (no pasa argv), así que en este proyecto los parámetros viajan
# por variables de entorno leídas por el script Python (mismo patrón que
# run_eval_generalizacion_ped.sh). Este runner expone esos parámetros como
# variables de entorno de bash, no como flags --tipo-cli:
#
#   ARCH         — "sthwp" | "subwp" | "both" (por defecto "both")
#   WAREHOUSES   — lista separada por espacios, p.ej. "02 03" (por defecto "02 03 04 05")
#   SEEDS        — lista separada por espacios, p.ej. "42" (por defecto "42 123 524")
#   GOAL         — un único goal_id, p.ej. "goal_003" (opcional, por defecto todos)
#   N_REPS       — repeticiones por goal (por defecto 10)
#   FORCE        — "1" para re-ejecutar combinaciones/episodios ya presentes (por defecto 0)
#   SMOKE_TEST   — "1" para ejecutar solo 1 goal x 1 rep por combinación (por defecto 0)
#
# Ejemplos:
#   ./run_eval_generalizacion_e2_1_r0.sh                       # protocolo completo
#   ARCH=sthwp ./run_eval_generalizacion_e2_1_r0.sh            # solo STH-WP, todo lo demás
#   ARCH=subwp WAREHOUSES="02" SEEDS="42" ./run_eval_generalizacion_e2_1_r0.sh
#   SMOKE_TEST=1 ARCH=sthwp WAREHOUSES="02" SEEDS="42" ./run_eval_generalizacion_e2_1_r0.sh
#   ./run_eval_generalizacion_e2_1_r0.sh                       # reanudar: se salta lo ya hecho
#
# Reanudación: cada script Python ya comprueba qué (goal_id, rep) están
# presentes en el CSV de salida y los salta — basta con relanzar el mismo
# comando tras una interrupción. FORCE=1 ignora ese filtro y repite todo.
set -euo pipefail

CONTROLLERS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLDS_DIR="$CONTROLLERS_DIR/../worlds"
WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
OUT_DIR="$CONTROLLERS_DIR/../../almacenes/resultados_evaluacion/fase2_e2_1_r0_sin_replanificacion"

STHWP_DIR="$CONTROLLERS_DIR/rl_train_STHWP"
SUBWP_DIR="$CONTROLLERS_DIR/rl_train_SUB_WP_continuo"

ARCH="${ARCH:-both}"
WAREHOUSES=(${WAREHOUSES:-02 03 04 05})
SEEDS=(${SEEDS:-42 123 524})
GOAL="${GOAL:-}"
N_REPS="${N_REPS:-10}"
FORCE="${FORCE:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"

mkdir -p "$OUT_DIR"

# Mismos max_steps calibrados que usa la evaluación E2.1-R1 (fase 2 con
# peatón) — no cambia el tamaño del almacén ni el recorrido A*, así que debe
# ser idéntico para que R0 y R1 sean comparables.
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
    local out_csv="$OUT_DIR/${metodo}_ped_e2_1_r0_wh${wh}_s${seed}.csv"

    echo ""
    echo "--- $metodo WH$wh seed=$seed | E2.1-R0 (sin replanning, max_steps=$max_steps, $N_REPS reps/goal) ---"
    echo "eval_gen_r0" > "$stage_dir/current_stage.txt"
    EVAL_WH="$wh" EVAL_SEED="$seed" EVAL_OUT_CSV="$out_csv" EVAL_MAX_STEPS="$max_steps" \
    EVAL_N_REPS="$N_REPS" EVAL_GOAL="$GOAL" EVAL_FORCE="$FORCE" EVAL_SMOKE_TEST="$SMOKE_TEST" \
        "$WEBOTS" --mode=fast --no-rendering --minimize "$world"
    echo "[OK] $metodo WH$wh seed=$seed (E2.1-R0) -> $out_csv"
}

echo "======================================================="
echo " EVALUACIÓN E2.1-R0 — replanificación dinámica desactivada"
[[ "$SMOKE_TEST" == "1" ]] && echo " *** SMOKE-TEST: 1 goal x 1 rep por combinación ***"
echo " ARCH=$ARCH  WAREHOUSES=(${WAREHOUSES[*]})  SEEDS=(${SEEDS[*]})  N_REPS=$N_REPS"
echo " Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

if [[ "$ARCH" == "sthwp" || "$ARCH" == "both" ]]; then
    for wh in "${WAREHOUSES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_eval "sthwp" "$wh" "$seed" "$STHWP_DIR" "$WORLDS_DIR/warehouse_ped_$wh.wbt" "$(max_steps_sthwp "$wh")"
        done
    done
fi

if [[ "$ARCH" == "subwp" || "$ARCH" == "both" ]]; then
    for wh in "${WAREHOUSES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_eval "subwp" "$wh" "$seed" "$SUBWP_DIR" "$WORLDS_DIR/warehouse_ped_${wh}_subwp.wbt" "$(max_steps_subwp "$wh")"
        done
    done
fi

echo ""
echo "======================================================="
echo " COMPLETO — $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSVs en $OUT_DIR"
echo "======================================================="
