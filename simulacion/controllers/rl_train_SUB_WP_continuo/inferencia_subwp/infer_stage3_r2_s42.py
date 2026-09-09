"""
Inferencia stage 3 r2 — SUB-WP seed=42 — Exit puro | dropoff corregido.

Salida: inferencia_subwp/resultados/infer_subwp_s42_wp75_r2_stage3.csv
"""

import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

CONTROLLER_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_PATH   = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
MODEL_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_s42_wp75_r2_stage3_final")
RUN_ID     = "subwp_s42_wp75_r2"
STAGE      = 3
N_POR_GOAL = 100
OUT_DIR    = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH   = os.path.join(OUT_DIR, "infer_subwp_s42_wp75_r2_stage3.csv")

os.makedirs(OUT_DIR, exist_ok=True)

env   = WebotsEnv(map_path=MAP_PATH, stage=STAGE, heading_sigma=0.25)
env._max_steps = 4000
model = PPO.load(MODEL_PATH, env=env)

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia stage 3 r2 | {RUN_ID} | exit puro")
print(f" {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} episodios")
print(f"{'='*60}\n")

FIELDS = ["episodio", "goal_id", "resultado", "pasos", "reward"]
stats  = {gid: {"exito": 0, "colision": 0, "truncado": 0} for gid in env.goal_ids}
ep_global = 0

with open(CSV_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writeheader()

for goal_idx in range(N_GOALS):
    goal_id = env.goal_ids[goal_idx]
    env._sample_goal_exit = lambda gi=goal_idx: gi
    print(f"\n  [{goal_id}] ({goal_idx+1}/{N_GOALS})")

    for ep in range(1, N_POR_GOAL + 1):
        ep_global += 1
        obs, _ = env.reset()
        done = truncated = False
        pasos = 0; reward_ep = 0.0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            pasos += 1; reward_ep += reward

        if info.get("exito"):       resultado = "exito";    stats[goal_id]["exito"]    += 1
        elif info.get("colision"):  resultado = "colision"; stats[goal_id]["colision"] += 1
        else:                       resultado = "truncado"; stats[goal_id]["truncado"] += 1

        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(
                {"episodio": ep_global, "goal_id": goal_id, "resultado": resultado,
                 "pasos": pasos, "reward": round(reward_ep, 2)})
        print(f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<10}  pasos={pasos}")

n_ex = sum(v["exito"]    for v in stats.values())
n_co = sum(v["colision"] for v in stats.values())
n_tr = sum(v["truncado"] for v in stats.values())

print(f"\n{'='*60}")
print(f" RESUMEN — {N_TOTAL} ep | {RUN_ID} stage3 r2")
print(f"  Éxito:    {n_ex:>4}/{N_TOTAL} ({n_ex/N_TOTAL*100:.1f}%)")
print(f"  Colisión: {n_co:>4}/{N_TOTAL} ({n_co/N_TOTAL*100:.1f}%)")
print(f"  Truncado: {n_tr:>4}/{N_TOTAL} ({n_tr/N_TOTAL*100:.1f}%)")
print(f"\n  Por goal:")
for gid, v in stats.items():
    tasa = v["exito"]/N_POR_GOAL*100
    print(f"    {gid}: {v['exito']}/{N_POR_GOAL} ({tasa:.0f}%){'  ⚠' if tasa < 50 else ''}")
print(f"{'='*60}\n[CSV] {CSV_PATH}")

env.supervisor.simulationQuit(0)
