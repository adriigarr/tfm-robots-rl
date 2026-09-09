"""
EVALUACIÓN DE GENERALIZACIÓN | SUBWP
Modelo E2.1 (mejor de la serie, obs realista LIDAR 5m, 40 dims, sin supervisor)
evaluado en almacenes no vistos en entrenamiento (WH02-05).

Fase 1 (sin peatón, EVAL_PED=0): 1 episodio por goal (sin repeticiones). El
simulador es determinista de punta a punta (posición inicial fija, sin
peatón, política deterministic=True), así que repetir un mismo goal
reproduce el mismo resultado — ver memoria del protocolo.

Fase 2 (con peatón, EVAL_PED=1): el peatón SÍ se aleatoriza de verdad en
cada reset() cuando ped_obs=True (_randomize_pedestrians() en webots_env.py
usa self.np_random.uniform sobre su corredor) — aquí repetir el goal aporta
información real, por eso se usa EVAL_N_REPS > 1.

IMPORTANTE — fix de rango del peatón: webots_env.py trae _ped_ranges
hardcodeado con las coordenadas del corredor de WH01, que no coinciden con
la posición real del peatón en los mundos warehouse_ped_0{2..5}.wbt (verificado
leyendo el --trajectory de cada DEF PEDESTRIAN_* en cada .wbt). Este script
sobrescribe env._ped_ranges con la tabla correcta por almacén justo tras
crear el entorno, sin tocar webots_env.py (compartido con otros experimentos).

IMPORTANTE — nº de peatones: el único baseline real validado de E2.1 con
peatón (sthwp_infer_e2_1_s42.csv, WH01) usa UN solo peatón físico (pasillo
corto de 8m). Los mundos warehouse_ped_0{2..5}.wbt se dejaron inicialmente
con 2 peatones por error de diseño — con 2 el éxito caía a 0% (confirmado
por ablation: con 1 peatón físico, 95% éxito en WH02; con 2, 0%). Los 8
mundos de evaluación (.wbt y _subwp.wbt) se recortaron a un único
DEF PEDESTRIAN_1 para que coincida con las condiciones de entrenamiento.

Parametrizado por variables de entorno (fijadas por el runner):
  EVAL_WH         — id de almacén, "02".."05"
  EVAL_SEED       — semilla del modelo, "42" | "123" | "524"
  EVAL_OUT_CSV    — ruta absoluta del CSV de salida
  EVAL_MAX_STEPS  — límite de pasos por episodio (por defecto 6000)
  EVAL_PED        — "1" para activar peatón (ped_obs=True + rango corregido),
                     "0" (por defecto) para la fase sin peatón
  EVAL_N_REPS     — repeticiones por goal (por defecto 1; usar >1 solo con
                     EVAL_PED=1, donde el peatón aleatorio hace que cada
                     repetición sea un episodio genuinamente distinto)

CSV: episodio, goal_id, rep, resultado, pasos, reward, warehouse_id, metodo, seed, peaton
"""

import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

CONTROLLER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

WH       = os.environ["EVAL_WH"]
SEED     = os.environ["EVAL_SEED"]
CSV_PATH = os.environ["EVAL_OUT_CSV"]
PED      = os.environ.get("EVAL_PED", "0") == "1"
N_REPS   = int(os.environ.get("EVAL_N_REPS", 1))

# Corredor real del único peatón (PEDESTRIAN_1) por almacén, leído del
# --trajectory en simulacion/worlds/warehouse_ped_0{2..5}.wbt.
# Formato igual al de env._ped_ranges: [(x_o_rango, y_o_rango, z), ...]
PED_RANGES = {
    "02": [{"x": -6.6,        "y": (-13.3, 13.3), "z": 1.27}],
    "03": [{"x": (-8.7, -0.1), "y": -5.6,          "z": 1.27}],
    "04": [{"x": -19.8,       "y": (-28.5, 28.5), "z": 1.27}],
    "05": [{"x": 0.0,         "y": (-30.0, 30.0), "z": 1.27}],
}

MAP_PATH   = os.path.join(CONTROLLER_DIR, f"warehouse_map{WH}.json")
MODEL_PATH = os.path.join(CONTROLLER_DIR, "pruebas", f"subwp_e2_1_s{SEED}_final")
RUN_ID     = f"subwp_eval_gen_wh{WH}_s{SEED}" + ("_ped" if PED else "")

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

env = WebotsEnv(map_path=MAP_PATH, stage=6, ped_obs=PED)
env._max_steps = int(os.environ.get("EVAL_MAX_STEPS", 6000))
if PED:
    env._ped_ranges = PED_RANGES[WH]
model = PPO.load(MODEL_PATH, env=env)

N_GOALS = len(env.goal_ids)
FIELDS  = ["episodio", "goal_id", "rep", "resultado", "pasos", "reward",
           "warehouse_id", "metodo", "seed", "peaton"]

print(f"\n{'='*60}")
print(f" Evaluación generalización {'CON PEATÓN' if PED else 'SIN PEATÓN'} | {RUN_ID}")
print(f" WH{WH} — {N_GOALS} goals × {N_REPS} rep(s)")
print(f"{'='*60}\n")

with open(CSV_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writeheader()

n_ex = 0
ep_global = 0
for goal_idx in range(N_GOALS):
    goal_id = env.goal_ids[goal_idx]
    env._sample_goal_approach = lambda gi=goal_idx: gi
    for rep in range(1, N_REPS + 1):
        ep_global += 1
        obs, _ = env.reset()
        done = truncated = False
        pasos = 0; reward_ep = 0.0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            pasos += 1; reward_ep += reward
        resultado = (
            "exito"        if info.get("exito") else
            "col_approach" if info.get("colision") and not info.get("llego_estanteria") else
            "col_exit"     if info.get("colision") and info.get("llego_estanteria") and not info.get("llego_espera") and not env._en_retorno else
            "col_return"   if info.get("colision") else
            "truncado"
        )
        if resultado == "exito":
            n_ex += 1
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(
                {"episodio": ep_global, "goal_id": goal_id, "rep": rep, "resultado": resultado,
                 "pasos": pasos, "reward": round(reward_ep, 2),
                 "warehouse_id": WH, "metodo": "subwp", "seed": SEED, "peaton": int(PED)})
        print(f"  [{goal_id}] ({goal_idx+1}/{N_GOALS}) rep {rep}/{N_REPS}  {resultado:<15}  pasos={pasos}")

n_total = N_GOALS * N_REPS
print(f"\n{'='*60}")
print(f" RESUMEN — {RUN_ID}")
print(f"  Éxito: {n_ex}/{n_total} ({n_ex/n_total*100:.1f}%)")
print(f"{'='*60}\n[CSV] {CSV_PATH}")

env.supervisor.simulationQuit(0)
