"""
Inferencia diagnóstica — SUB-WP s42 v3 — peatones COMPLETAMENTE ESTÁTICOS.

Los peatones NO se mueven en ningún momento del episodio. En cada step se
teleportan a su posición inicial del world file, anulando su movimiento.

Propósito diagnóstico: separar el impacto del movimiento del peatón del
impacto de la reformulación de política por B1.

Comparar con:
  - Estático v1:           ~100%
  - v3 s42 dinámico:        39.8%
  - Este script:            ???%

Salida: inferencia_subwp/resultados/infer_din_v3_s42_staticped.csv
"""

import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

CONTROLLER_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_PATH   = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
MODEL_PATH = os.path.join(CONTROLLER_DIR, "pruebas",
                          "subwp_s42_wp75_stage6_din_v3_final")
RUN_ID     = "din_v3_s42_staticped_subwp"
N_POR_GOAL = 100
OUT_DIR    = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH   = os.path.join(OUT_DIR, "infer_din_v3_s42_staticped.csv")

os.makedirs(OUT_DIR, exist_ok=True)

env   = WebotsEnv(map_path=MAP_PATH, stage=6, ped_obs=True)
env._max_steps = 6000
model = PPO.load(MODEL_PATH, env=env)

STATIC_POSITIONS = [
    {"x": -2.0,  "y": 0.3,  "z": 1.27},   # PEDESTRIAN_1
    {"x": -9.5,  "y": -5.0, "z": 1.27},   # PEDESTRIAN_2
]

def freeze_pedestrians():
    for i, node in enumerate(env._ped_nodes):
        p = STATIC_POSITIONS[i]
        node.getField("translation").setSFVec3f([p["x"], p["y"], p["z"]])

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia DIAGNÓSTICA — peatones ESTÁTICOS | {RUN_ID}")
print(f" Modelo: {os.path.basename(MODEL_PATH)} (final)")
print(f" Peatones: congelados en posición inicial en CADA step")
print(f" {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} ciclos")
print(f"{'='*60}\n")

FIELDS = ["episodio", "goal_id", "resultado", "pasos", "reward"]
stats_por_goal = {
    gid: {"exito": 0, "col_approach": 0, "col_exit": 0, "col_return": 0, "truncado": 0}
    for gid in env.goal_ids
}
ep_global = 0

with open(CSV_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writeheader()

for goal_idx in range(N_GOALS):
    goal_id = env.goal_ids[goal_idx]
    env._sample_goal_approach = lambda gi=goal_idx: gi
    print(f"\n  [{goal_id}] ({goal_idx + 1}/{N_GOALS})")

    for ep in range(1, N_POR_GOAL + 1):
        ep_global += 1
        obs, _ = env.reset()
        freeze_pedestrians()

        done = truncated = False
        pasos = 0
        total_reward = 0.0

        while not done and not truncated:
            freeze_pedestrians()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            pasos += 1
            total_reward += reward

        resultado = (
            "exito"        if info.get("exito")    else
            "col_approach" if info.get("colision") and not info.get("llego_estanteria") else
            "col_exit"     if info.get("colision") and info.get("llego_estanteria") and not info.get("llego_espera") and not env._en_retorno else
            "col_return"   if info.get("colision") else
            "truncado"
        )
        stats_por_goal[goal_id][resultado] += 1

        row = {"episodio": ep_global, "goal_id": goal_id, "resultado": resultado,
               "pasos": pasos, "reward": round(total_reward, 2)}
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

        print(f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<15}  pasos={pasos}")

n_exito   = sum(v["exito"]        for v in stats_por_goal.values())
n_col_ap  = sum(v["col_approach"] for v in stats_por_goal.values())
n_col_ex  = sum(v["col_exit"]     for v in stats_por_goal.values())
n_col_ret = sum(v["col_return"]   for v in stats_por_goal.values())
n_trunc   = sum(v["truncado"]     for v in stats_por_goal.values())

print(f"\n{'='*60}")
print(f" RESUMEN — {N_TOTAL} ciclos | {RUN_ID}")
print(f"{'='*60}")
print(f"  Éxito:         {n_exito:>4}/{N_TOTAL}  ({n_exito/N_TOTAL*100:.1f}%)")
print(f"  Col. approach: {n_col_ap:>4}/{N_TOTAL}  ({n_col_ap/N_TOTAL*100:.1f}%)")
print(f"  Col. exit:     {n_col_ex:>4}/{N_TOTAL}  ({n_col_ex/N_TOTAL*100:.1f}%)")
print(f"  Col. retorno:  {n_col_ret:>4}/{N_TOTAL}  ({n_col_ret/N_TOTAL*100:.1f}%)")
print(f"  Truncado:      {n_trunc:>4}/{N_TOTAL}  ({n_trunc/N_TOTAL*100:.1f}%)")
print(f"\n  DIAGNÓSTICO:")
print(f"    Estático v1:          ~100%")
print(f"    v3 s42 dinámico:       39.8%")
print(f"    v3 s42 ped estáticos:  {n_exito/N_TOTAL*100:.1f}%")
print(f"{'='*60}\n[CSV] {CSV_PATH}")

env.supervisor.simulationQuit(0)
