"""
Inferencia stage 5 r2 — SUB-WP seed=42 — Return puro | dropoff corregido.

300 episodios (ruta fija descarga → espera).
Salida: inferencia_subwp/resultados/infer_subwp_s524_wp75_r2_stage5.csv
"""

import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

CONTROLLER_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_PATH   = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
MODEL_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_s524_wp75_r2_stage5_final")
RUN_ID     = "subwp_s524_wp75_r2"
STAGE      = 5
N_EP       = 300
OUT_DIR    = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH   = os.path.join(OUT_DIR, "infer_subwp_s524_wp75_r2_stage5.csv")

os.makedirs(OUT_DIR, exist_ok=True)

env   = WebotsEnv(map_path=MAP_PATH, stage=STAGE)
env._max_steps = 3000
model = PPO.load(MODEL_PATH, env=env)

print(f"\n{'='*60}")
print(f" Inferencia stage 5 r2 | {RUN_ID} | return puro | {N_EP} ep")
print(f"{'='*60}\n")

FIELDS = ["episodio", "resultado", "pasos", "reward"]
resultados = []
n_ex = n_co = n_tr = 0

with open(CSV_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writeheader()

for ep in range(1, N_EP + 1):
    obs, _ = env.reset()
    done = truncated = False
    pasos = 0; reward_ep = 0.0

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        pasos += 1; reward_ep += reward

    if info.get("exito"):      resultado = "exito";    n_ex += 1
    elif info.get("colision"): resultado = "colision"; n_co += 1
    else:                      resultado = "truncado"; n_tr += 1

    row = {"episodio": ep, "resultado": resultado,
           "pasos": pasos, "reward": round(reward_ep, 2)}
    resultados.append(row)
    with open(CSV_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    print(f"  ep {ep:>3}/{N_EP}  {resultado:<10}  pasos={pasos}")

print(f"\n{'='*60}")
print(f" RESUMEN — {N_EP} ep | {RUN_ID} stage5 r2")
print(f"  Éxito:    {n_ex:>3}/{N_EP} ({n_ex/N_EP*100:.1f}%)")
print(f"  Colisión: {n_co:>3}/{N_EP} ({n_co/N_EP*100:.1f}%)")
print(f"  Truncado: {n_tr:>3}/{N_EP} ({n_tr/N_EP*100:.1f}%)")
print(f"{'='*60}\n[CSV] {CSV_PATH}")

env.supervisor.simulationQuit(0)
