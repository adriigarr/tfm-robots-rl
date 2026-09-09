"""
Inferencia stage 1 — SUB-WP seed=524 — Approach puro, goal_01.

500 episodios sobre goal_01.

Salida: inferencia_subwp/resultados/infer_subwp_s524_stage1.csv
"""

import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

MAP_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "warehouse_map01.json")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pruebas", "subwp_s524_stage1_final")
RUN_ID     = "subwp_s524"
N_EP       = 100
OUT_DIR    = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH   = os.path.join(OUT_DIR, "infer_subwp_s524_stage1.csv")

env   = WebotsEnv(map_path=MAP_PATH, stage=1)
model = PPO.load(MODEL_PATH, env=env)
env._sample_goal_approach = lambda: 0

print(f"\n{'='*60}")
print(f" Inferencia stage 1 SUB-WP | {RUN_ID} | goal_01 × {N_EP} ep")
print(f"{'='*60}\n")

resultados = []
n_exito = n_col = n_trunc = 0

for ep in range(1, N_EP + 1):
    obs, _ = env.reset()
    done = truncated = False
    pasos = 0; reward_ep = 0.0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        pasos += 1; reward_ep += reward

    es_exito    = bool(info.get("exito",    False))
    es_colision = bool(info.get("colision", False))
    if es_exito:      resultado = "exito";    n_exito += 1
    elif es_colision: resultado = "colision"; n_col += 1
    else:             resultado = "truncado"; n_trunc += 1

    resultados.append({"episodio": ep, "goal_id": "goal_01",
                        "resultado": resultado, "pasos": pasos,
                        "reward": round(reward_ep, 2)})
    if ep % 50 == 0:
        print(f"  ep {ep}/{N_EP}  éxito={n_exito/ep*100:.1f}%")

print(f"\n{'='*60}")
print(f" RESUMEN | {RUN_ID} stage1 | goal_01 × {N_EP} ep")
print(f"  Éxito:    {n_exito}/{N_EP} ({n_exito/N_EP*100:.1f}%)")
print(f"  Colisión: {n_col}/{N_EP}  ({n_col/N_EP*100:.1f}%)")
print(f"  Truncado: {n_trunc}/{N_EP}  ({n_trunc/N_EP*100:.1f}%)")
print(f"{'='*60}\n")

os.makedirs(OUT_DIR, exist_ok=True)
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["episodio","goal_id","resultado","pasos","reward"])
    writer.writeheader(); writer.writerows(resultados)
print(f"[CSV] {CSV_PATH}")
env.supervisor.simulationQuit(0)
