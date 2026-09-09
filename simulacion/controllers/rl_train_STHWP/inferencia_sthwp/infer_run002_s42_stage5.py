"""
Inferencia stage 5 — run002 seed=42 — RETORNO PURO (descarga→espera).

Modelo base: run002_s42_stage5_final (entrenado sobre stage4v2_final).
300 episodios deterministas del tramo retorno.
No hay loop por goals: solo existe 1 ruta de retorno (zona_descarga→zona_espera).

Salida: inferencia_sthwp/resultados/infer_run002_s42_stage5.csv
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

MAP_PATH      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "warehouse_map01.json")
MODEL_PATH    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pruebas", "run002_s42_stage5_final")
RUN_ID        = "run002_s42"
HEADING_SIGMA = 0.25
N_TOTAL       = 300
OUT_DIR       = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH      = os.path.join(OUT_DIR, "infer_run002_s42_stage5.csv")

env   = WebotsEnv(map_path=MAP_PATH, stage=5, heading_sigma=HEADING_SIGMA)
model = PPO.load(MODEL_PATH, env=env)
env._max_steps = 3000

print(f"\n{'='*60}")
print(f" Inferencia stage 5 — RETORNO PURO | {RUN_ID}")
print(f" {N_TOTAL} episodios deterministas")
print(f"{'='*60}\n")

resultados    = []
n_exito       = 0
n_colision    = 0
n_truncado    = 0

for ep in range(1, N_TOTAL + 1):
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
        n_exito   += 1
    elif es_colision:
        resultado  = "colision"
        n_colision += 1
    else:
        resultado  = "truncado"
        n_truncado += 1

    resultados.append({
        "episodio":  ep,
        "resultado": resultado,
        "pasos":     pasos,
        "reward":    round(reward_ep, 2),
    })
    print(f"  ep {ep:>3}/{N_TOTAL}  {resultado:<12}  pasos={pasos:<5}  reward={reward_ep:.1f}")

pasos_medio  = sum(r["pasos"]  for r in resultados) / N_TOTAL
reward_medio = sum(r["reward"] for r in resultados) / N_TOTAL

print(f"\n{'='*60}")
print(f" RESUMEN — {N_TOTAL} ep deterministas | {RUN_ID} stage5")
print(f"{'='*60}")
print(f"  Éxito:     {n_exito:>3}/{N_TOTAL}  ({n_exito    / N_TOTAL * 100:.1f}%)")
print(f"  Colisión:  {n_colision:>3}/{N_TOTAL}  ({n_colision / N_TOTAL * 100:.1f}%)")
print(f"  Truncado:  {n_truncado:>3}/{N_TOTAL}  ({n_truncado / N_TOTAL * 100:.1f}%)")
print(f"  Pasos/ep:  {pasos_medio:.1f}  |  Reward/ep: {reward_medio:.1f}")
print(f"{'='*60}\n")

os.makedirs(OUT_DIR, exist_ok=True)
FIELDS = ["episodio", "resultado", "pasos", "reward"]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(resultados)

with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(["RESUMEN_GLOBAL", "", "", ""])
    writer.writerow(["n_total",        N_TOTAL,  "", ""])
    writer.writerow(["exito_%",        f"{n_exito    / N_TOTAL * 100:.1f}", "", ""])
    writer.writerow(["colision_%",     f"{n_colision / N_TOTAL * 100:.1f}", "", ""])
    writer.writerow(["truncado_%",     f"{n_truncado / N_TOTAL * 100:.1f}", "", ""])
    writer.writerow(["pasos_medio",    f"{pasos_medio:.1f}",  "", ""])
    writer.writerow(["reward_medio",   f"{reward_medio:.1f}", "", ""])

print(f"[CSV] Guardado en {CSV_PATH}")
env.supervisor.simulationQuit(0)
