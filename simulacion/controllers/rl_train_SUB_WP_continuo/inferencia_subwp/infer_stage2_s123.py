"""
Inferencia stage 2 — SUB-WP seed=123 — Approach puro, todos los goals.

Solo genera estadísticas. Los headings se miden con seed=42 (infer_stage2_s42.py).

100 episodios por goal × 28 goals = 2800 episodios.

Salida: inferencia_subwp/resultados/infer_subwp_s123_stage2.csv
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

MAP_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "warehouse_map01.json")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pruebas", "subwp_s123_stage2_final")
RUN_ID     = "subwp_s123"
STAGE      = 2
N_POR_GOAL = 100
OUT_DIR    = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH   = os.path.join(OUT_DIR, "infer_subwp_s123_stage2.csv")

env   = WebotsEnv(map_path=MAP_PATH, stage=STAGE)
model = PPO.load(MODEL_PATH, env=env)

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia stage 2 SUB-WP | {RUN_ID}")
print(f" {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} episodios")
print(f"{'='*60}\n")

resultados     = []
stats_por_goal = {gid: {"exito": 0, "colision": 0, "truncado": 0}
                  for gid in env.goal_ids}
ep_global      = 0

for goal_idx in range(N_GOALS):
    goal_id = env.goal_ids[goal_idx]
    env._sample_goal_approach = lambda gi=goal_idx: gi
    print(f"\n  [{goal_id}] ({goal_idx + 1}/{N_GOALS})")

    for ep in range(1, N_POR_GOAL + 1):
        ep_global += 1
        obs, _ = env.reset()
        done = truncated = False
        pasos = 0
        reward_ep = 0.0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            pasos     += 1
            reward_ep += reward

        es_exito    = bool(info.get("exito",    False))
        es_colision = bool(info.get("colision", False))

        if es_exito:
            resultado = "exito"
            stats_por_goal[goal_id]["exito"] += 1
        elif es_colision:
            resultado = "colision"
            stats_por_goal[goal_id]["colision"] += 1
        else:
            resultado = "truncado"
            stats_por_goal[goal_id]["truncado"] += 1

        resultados.append({
            "episodio":  ep_global,
            "goal_id":   goal_id,
            "resultado": resultado,
            "pasos":     pasos,
            "reward":    round(reward_ep, 2),
        })
        print(f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<12}  pasos={pasos:<5}  "
              f"reward={reward_ep:.1f}")

n_exito = sum(v["exito"]    for v in stats_por_goal.values())
n_col   = sum(v["colision"] for v in stats_por_goal.values())
n_trunc = sum(v["truncado"] for v in stats_por_goal.values())

print(f"\n{'='*60}")
print(f" RESUMEN — {N_TOTAL} ep | {RUN_ID} stage2")
print(f"  Éxito:    {n_exito:>4}/{N_TOTAL}  ({n_exito/N_TOTAL*100:.1f}%)")
print(f"  Colisión: {n_col:>4}/{N_TOTAL}  ({n_col/N_TOTAL*100:.1f}%)")
print(f"  Truncado: {n_trunc:>4}/{N_TOTAL}  ({n_trunc/N_TOTAL*100:.1f}%)")
for gid, v in stats_por_goal.items():
    tasa = v["exito"] / N_POR_GOAL * 100
    flag = "  ⚠" if tasa < 70 else ""
    print(f"    {gid}: {v['exito']}/{N_POR_GOAL} ({tasa:.0f}%){flag}")
print(f"{'='*60}\n")

os.makedirs(OUT_DIR, exist_ok=True)
FIELDS = ["episodio", "goal_id", "resultado", "pasos", "reward"]
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(resultados)
print(f"[CSV] Guardado en {CSV_PATH}")

env.supervisor.simulationQuit(0)
