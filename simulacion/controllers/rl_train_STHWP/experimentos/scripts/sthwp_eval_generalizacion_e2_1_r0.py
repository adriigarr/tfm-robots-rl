"""
EVALUACIÓN E2.1-R0 | STHWP — replanificación dinámica DESACTIVADA
Variante de sthwp_eval_generalizacion.py (E2.1-R1, replanning LIDAR activo)
para la condición E2.1-R0: los mismos modelos E2.1, con peatón activo, pero
con env.enable_dynamic_replanning=False — el robot nunca recalcula full_path
mid-episodio por detección LIDAR del peatón. La planificación A* inicial de
cada tramo (approach/exit/return) NO se toca, sigue funcionando igual.

No se reentrena ni se modifica ningún peso de los modelos. Se reutilizan
exactamente los mismos checkpoints finales que E2.1-R1:
  pruebas/sthwp_e2_1_s{42,123,524}_final.zip

Reproducibilidad del peatón (ver documentación E2.1_R0_README.md):
El evaluador original (E2.1-R1) llama a env.reset() SIN semilla explícita,
así que Gymnasium siembra self.np_random una única vez con entropía del SO
en el primer reset() del proceso y de ahí en adelante el RNG avanza como un
único flujo continuo — no hay ninguna semilla registrada, y los CSV de R1 no
guardan la posición del peatón. Por tanto NO es posible reproducir episodio
a episodio las mismas posiciones de peatón que tuvo R1.
Para que ESTA evaluación (R0) sea internamente reproducible y reanudable sin
duplicar ni perder episodios, aquí SÍ se fija una semilla determinista por
episodio (derivada de metodo+seed+almacén+goal+rep vía crc32, no vía hash()
de Python que está aleatorizado por proceso). Esta semilla es propia de R0:
NO reproduce las posiciones de peatón usadas en R1. La comparación R0 vs R1
debe hacerse de forma AGREGADA (tasas de éxito, no episodio por episodio).

CSV de salida: mismas columnas que E2.1-R1 (compatibles con los notebooks)
más las columnas de metadatos de la condición experimental.

Parametrizado por variables de entorno (fijadas por el runner):
  EVAL_WH            — id de almacén, "02".."05"
  EVAL_SEED          — semilla del modelo, "42" | "123" | "524"
  EVAL_OUT_CSV       — ruta absoluta del CSV de salida
  EVAL_MAX_STEPS     — límite de pasos por episodio
  EVAL_N_REPS        — repeticiones por goal (por defecto 10)
  EVAL_GOAL          — opcional, restringe a un único goal_id (p.ej. "goal_003")
  EVAL_SMOKE_TEST    — "1" para ejecutar solo 1 goal x 1 rep y verificar el pipeline
  EVAL_FORCE         — "1" para re-ejecutar episodios ya presentes en el CSV
"""

import os, sys, csv, zlib, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

CONTROLLER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

WH          = os.environ["EVAL_WH"]
SEED        = os.environ["EVAL_SEED"]
CSV_PATH    = os.environ["EVAL_OUT_CSV"]
N_REPS      = int(os.environ.get("EVAL_N_REPS", 10))
GOAL_FILTER = os.environ.get("EVAL_GOAL", "") or None
SMOKE_TEST  = os.environ.get("EVAL_SMOKE_TEST", "0") == "1"
FORCE       = os.environ.get("EVAL_FORCE", "0") == "1"

# Mismo rango real de peatón (leído del --trajectory de cada .wbt) que usa
# sthwp_eval_generalizacion.py — no se cambia nada de la geometría/peatón.
PED_RANGES = {
    "02": [{"x": -6.6,        "y": (-13.3, 13.3), "z": 1.27}],
    "03": [{"x": (-8.7, -0.1), "y": -5.6,          "z": 1.27}],
    "04": [{"x": -19.8,       "y": (-28.5, 28.5), "z": 1.27}],
    "05": [{"x": 0.0,         "y": (-30.0, 30.0), "z": 1.27}],
}

MAP_PATH   = os.path.join(CONTROLLER_DIR, f"warehouse_map{WH}.json")
MODEL_PATH = os.path.join(CONTROLLER_DIR, "pruebas", f"sthwp_e2_1_s{SEED}_final")
RUN_ID     = f"sthwp_eval_gen_e2_1_r0_wh{WH}_s{SEED}"

if not os.path.exists(MODEL_PATH + ".zip"):
    raise FileNotFoundError(f"Modelo E2.1 no encontrado: {MODEL_PATH}.zip")

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

TIMESTEP_S = 0.032  # WebotsEnv.TIMESTEP_MS = 32

FIELDS = [
    # columnas originales (compatibles con notebooks de fase 2)
    "episodio", "goal_id", "rep", "resultado", "pasos", "reward",
    "warehouse_id", "metodo", "seed", "peaton",
    # metadatos de la condición experimental
    "policy_stage", "evaluation_variant", "dynamic_replanning",
    "architecture", "model_seed", "warehouse", "repetition",
    "episode_seed", "result", "collision_phase", "simulated_time",
    "replan_attempts", "replan_successes", "pedestrian_initial_position",
]


def derive_episode_seed(metodo, model_seed, wh, goal_idx, rep):
    """Semilla determinista (crc32, NO hash() de Python — está aleatorizado
    por proceso) propia de E2.1-R0. No reproduce las posiciones de R1."""
    key = f"{metodo}|{model_seed}|{wh}|{goal_idx}|{rep}"
    return zlib.crc32(key.encode()) & 0xFFFFFFFF


def cargar_episodios_completados(csv_path):
    """(goal_id, rep) ya presentes en el CSV — para reanudar sin repetir."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("resultado"):
                done.add((row["goal_id"], row["rep"]))
    return done


env = WebotsEnv(map_path=MAP_PATH, stage=6, ped_obs=True,
                enable_dynamic_replanning=False)
env._max_steps = int(os.environ["EVAL_MAX_STEPS"])
env._ped_ranges = PED_RANGES[WH]
model = PPO.load(MODEL_PATH, env=env)

goal_ids_all = list(env.goal_ids)
if GOAL_FILTER:
    goal_indices = [i for i, gid in enumerate(goal_ids_all) if gid == GOAL_FILTER]
    if not goal_indices:
        raise ValueError(f"EVAL_GOAL={GOAL_FILTER} no existe en WH{WH} (goals válidos: {goal_ids_all[:5]}...)")
else:
    goal_indices = list(range(len(goal_ids_all)))

if SMOKE_TEST:
    goal_indices = goal_indices[:1]
    n_reps_run = 1
else:
    n_reps_run = N_REPS

N_GOALS = len(goal_indices)
n_total_previstos = N_GOALS * n_reps_run

completados = set() if FORCE else cargar_episodios_completados(CSV_PATH)
escribir_header = not os.path.exists(CSV_PATH) or FORCE
if FORCE and os.path.exists(CSV_PATH):
    os.remove(CSV_PATH)
if escribir_header:
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

print(f"\n{'='*70}")
print(f" EVALUACIÓN E2.1-R0 (sin replanificación dinámica) | {RUN_ID}")
if SMOKE_TEST:
    print(" *** MODO SMOKE-TEST: 1 goal x 1 repetición ***")
print(f" WH{WH} — {N_GOALS} goals x {n_reps_run} rep(s) = {n_total_previstos} episodios previstos")
print(f" Ya completados (se saltan): {len(completados)}")
print(f"{'='*70}\n")

n_ex = 0
n_hecho = 0
t0 = time.time()
ep_global = len(completados)

for goal_idx in goal_indices:
    goal_id = goal_ids_all[goal_idx]
    env._sample_goal_approach = lambda gi=goal_idx: gi
    for rep in range(1, n_reps_run + 1):
        if (goal_id, str(rep)) in completados:
            continue

        episode_seed = derive_episode_seed("sthwp", SEED, WH, goal_idx, rep)
        ep_global += 1
        obs, _ = env.reset(seed=episode_seed)

        ped_pos = list(env._ped_prev_pos[0]) if env._ped_prev_pos else None

        done = truncated = False
        pasos = 0
        reward_ep = 0.0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            pasos += 1
            reward_ep += reward

        resultado = (
            "exito"        if info.get("exito") else
            "col_approach" if info.get("colision") and not info.get("llego_estanteria") else
            "col_exit"     if info.get("colision") and info.get("llego_estanteria") and not info.get("llego_espera") and not env._en_retorno else
            "col_return"   if info.get("colision") else
            "truncado"
        )
        collision_phase = resultado.replace("col_", "") if resultado.startswith("col_") else ""

        replan_attempts  = env._replan_attempts
        replan_successes = env._replan_successes
        if replan_attempts != 0 or replan_successes != 0:
            raise RuntimeError(
                f"[E2.1-R0] replan_attempts={replan_attempts} replan_successes={replan_successes} "
                f"en goal {goal_id} rep {rep} — deberían ser 0 con enable_dynamic_replanning=False. "
                "Esto indica un fallo del mecanismo de desactivación, no un resultado válido."
            )

        if resultado == "exito":
            n_ex += 1
        n_hecho += 1

        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow({
                "episodio": ep_global, "goal_id": goal_id, "rep": rep, "resultado": resultado,
                "pasos": pasos, "reward": round(reward_ep, 2),
                "warehouse_id": WH, "metodo": "sthwp", "seed": SEED, "peaton": 1,
                "policy_stage": "E2.1", "evaluation_variant": "E2.1-R0",
                "dynamic_replanning": False,
                "architecture": "sthwp", "model_seed": SEED, "warehouse": WH,
                "repetition": rep, "episode_seed": episode_seed, "result": resultado,
                "collision_phase": collision_phase,
                "simulated_time": round(pasos * TIMESTEP_S, 3),
                "replan_attempts": replan_attempts, "replan_successes": replan_successes,
                "pedestrian_initial_position": str(ped_pos) if ped_pos else "",
            })

        restantes = n_total_previstos - len(completados) - n_hecho
        elapsed = time.time() - t0
        eta_s = (elapsed / n_hecho) * restantes if n_hecho else 0
        print(f"  [{goal_id}] ({goal_idx+1}/{len(goal_ids_all)}) rep {rep}/{n_reps_run}  "
              f"{resultado:<15} pasos={pasos}  restantes={restantes}  ETA={eta_s/60:.1f}min")

print(f"\n{'='*70}")
print(f" RESUMEN — {RUN_ID}")
if n_hecho:
    print(f"  Éxito (episodios ejecutados en esta sesión): {n_ex}/{n_hecho} ({n_ex/n_hecho*100:.1f}%)")
else:
    print("  Nada que ejecutar — todos los episodios ya estaban completados.")
print(f"{'='*70}\n[CSV] {CSV_PATH}")

env.supervisor.simulationQuit(0)
