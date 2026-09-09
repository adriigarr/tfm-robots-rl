# E2.1-R0 — evaluación de generalización sin replanificación dinámica

## Qué representa

`E2.1-R0` es la misma política E2.1 (los mismos 6 checkpoints finales:
`sthwp_e2_1_s{42,123,524}_final`, `subwp_e2_1_s{42,123,524}_final`) y el
mismo protocolo de evaluación con peatón que ya existe (`E2.1-R1`,
`resultados_evaluacion/*_ped_wh*_s*.csv`), pero con **la replanificación
dinámica LIDAR desactivada**: el robot ya no recalcula su ruta A* a mitad de
tramo cuando el LIDAR detecta que el peatón corta el camino hacia el
siguiente subgoal/waypoint.

## Diferencia con E2.1-R1

| | E2.1-R1 (ya existe) | E2.1-R0 (nuevo) |
|---|---|---|
| Modelo | E2.1 | E2.1 (idéntico, mismos pesos) |
| Peatón activo | Sí | Sí |
| Replanificación LIDAR mid-episodio | **Activa** | **Desactivada** |
| Planificación A* inicial por tramo | Activa | Activa (sin cambios) |
| CSV | `resultados_evaluacion/*_ped_wh*_s*.csv` | `resultados_evaluacion/fase2_e2_1_r0_sin_replanificacion/*_ped_e2_1_r0_wh*_s*.csv` |

## Qué permanece activo en E2.1-R0

- El peatón y su movimiento, `ped_obs=True`, las 36 lecturas LIDAR, la
  observación completa de 40 dims, la política PPO tal cual se entrenó.
- El *shield* de reducción de velocidad ante obstáculos próximos (no es
  replanificación, es un recorte de velocidad lineal en `step()`).
- La detección de colisiones y las condiciones de éxito/truncamiento.
- La planificación A* inicial de cada tramo (base→recogida, recogida→entrega,
  entrega→base).
- En STH-WP: el subgoal local sigue recalculándose según la posición del
  robot sobre la ruta ya planificada (`compute_sth_subgoal`) — esto **no** es
  replanificación global, es solo "mirar más adelante en la misma ruta".
- En SUB-WP: el avance normal por la secuencia de waypoints (`_wp_idx`).

## Qué se desactiva

Únicamente la sustitución de `full_path` por una ruta A* nueva calculada a
mitad de episodio a partir de una detección LIDAR de obstáculo dinámico
(`_try_replan_lidar` → `_replanificar_lidar`). Se controla con el nuevo
parámetro `WebotsEnv(..., enable_dynamic_replanning=False)`, añadido en
`webots_env.py` de ambas arquitecturas. El código de replanificación no se ha
borrado ni comentado: si por error se llegara a invocar estando desactivado,
las funciones `_try_replan`, `_replanificar`, `_try_replan_lidar` y
`_replanificar_lidar` lanzan `RuntimeError` inmediatamente. Además, el propio
evaluador comprueba al final de cada episodio que `replan_attempts` y
`replan_successes` sean 0, y aborta con un error si no lo son.

## Reproducibilidad del peatón — léase antes de comparar con R1

El evaluador de R1 llama a `env.reset()` **sin semilla explícita**. Gymnasium
siembra `self.np_random` una única vez, con entropía del sistema operativo,
en el primer `reset()` del proceso; a partir de ahí el generador avanza como
un único flujo continuo, sin ningún punto de control. **No hay ninguna
semilla registrada en ningún sitio para R1**, y sus CSV no guardan la
posición inicial del peatón. Por tanto:

- **No es posible reproducir, episodio a episodio, las mismas posiciones de
  peatón que tuvo R1.** No lo intentéis ni lo asumáis en el análisis.
- R0 sí fija una semilla determinista por episodio (`episode_seed`, columna
  del CSV, derivada con `zlib.crc32` de `metodo|seed|almacén|goal|rep` — no
  con `hash()` de Python, que está aleatorizado por proceso). Esto hace que
  **R0 sea internamente reproducible y reanudable** sin perder ni duplicar
  episodios, pero esa semilla es propia de R0 y **no reproduce** las
  posiciones que tuvo R1.
- **La comparación R0 vs R1 debe ser agregada** (tasas de éxito por
  almacén/método/seed), nunca emparejada episodio a episodio.

## Formato del CSV

Mismas columnas que E2.1-R1 (`episodio, goal_id, rep, resultado, pasos,
reward, warehouse_id, metodo, seed, peaton`) — los notebooks existentes
pueden leer estos CSV sin cambios — más las columnas de metadatos:
`policy_stage, evaluation_variant, dynamic_replanning, architecture,
model_seed, warehouse, repetition, episode_seed, result, collision_phase,
simulated_time, replan_attempts, replan_successes,
pedestrian_initial_position`.

## Cómo ejecutar

Todo se lanza desde `simulacion/controllers/run_eval_generalizacion_e2_1_r0.sh`.
Webots no acepta argumentos de línea de comandos para el controller, así que
los parámetros viajan por variables de entorno de bash (mismo patrón que
`run_eval_generalizacion_ped.sh`), no por flags `--tipo-cli`:

```bash
cd simulacion/controllers

# Prueba corta (1 goal x 1 rep) — YA EJECUTADA Y VALIDADA, ver más abajo
SMOKE_TEST=1 ARCH=sthwp WAREHOUSES="02" SEEDS="42" ./run_eval_generalizacion_e2_1_r0.sh
SMOKE_TEST=1 ARCH=subwp WAREHOUSES="02" SEEDS="42" ./run_eval_generalizacion_e2_1_r0.sh

# STH-WP completo (4 almacenes x 3 seeds x 10 reps)
ARCH=sthwp ./run_eval_generalizacion_e2_1_r0.sh

# SUB-WP completo
ARCH=subwp ./run_eval_generalizacion_e2_1_r0.sh

# Protocolo completo (ambas arquitecturas)
./run_eval_generalizacion_e2_1_r0.sh

# Un solo almacén/seed (útil para probar o repartir el trabajo)
ARCH=subwp WAREHOUSES="05" SEEDS="524" ./run_eval_generalizacion_e2_1_r0.sh

# Reanudar tras una interrupción: mismo comando, se saltan los episodios
# ya presentes en el CSV (verificado con el smoke-test, ver más abajo)
./run_eval_generalizacion_e2_1_r0.sh

# Forzar reevaluación completa ignorando lo ya hecho
FORCE=1 ./run_eval_generalizacion_e2_1_r0.sh
```

## Comprobar que la replanificación se mantiene en cero

```bash
cd almacenes/resultados_evaluacion/fase2_e2_1_r0_sin_replanificacion
awk -F, 'FNR>1 && ($22!=0 || $23!=0)' *.csv   # columnas replan_attempts, replan_successes (FNR, no NR: reinicia por archivo)
```
No debería imprimir ninguna fila. Si el pipeline funciona mal, el propio
evaluador ya aborta con `RuntimeError` antes de escribir esa fila, así que en
la práctica esto es una comprobación redundante de refuerzo.

## Dónde se guardan los resultados

`almacenes/resultados_evaluacion/fase2_e2_1_r0_sin_replanificacion/`
(carpeta independiente, no toca ni sobrescribe los CSV de R1).
