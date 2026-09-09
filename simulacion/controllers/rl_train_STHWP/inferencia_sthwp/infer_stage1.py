"""
Inferencia stage 1 — goal_01, 100 episodios, política determinista.

Carga el modelo run001_s123_stage1_final y evalúa 100 episodios en el
mismo entorno de stage 1 (approach único, goal_01) sin exploración.

Salida: inferencia_sthwp/resultados/infer_run001_s123_stage1.csv
"""

import os
import sys
import csv

# Añadir el directorio padre al path para importar webots_env y demás
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

# ── configuración ─────────────────────────────────────────────────────────────
MAP_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "warehouse_map01.json")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pruebas", "run001_s123_stage1_final")
N_EPISODIOS = 100
RUN_ID      = "run001_s123"
OUT_DIR     = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH    = os.path.join(OUT_DIR, f"infer_{RUN_ID}_stage1.csv")

# ── cargar entorno y modelo ───────────────────────────────────────────────────
env   = WebotsEnv(map_path=MAP_PATH, stage=1)
model = PPO.load(MODEL_PATH, env=env)

print(f"\n{'='*55}")
print(f" Inferencia stage 1 | {RUN_ID} | {N_EPISODIOS} episodios")
print(f"{'='*55}\n")

# ── bucle de inferencia ───────────────────────────────────────────────────────
resultados = []
n_exitos     = 0
n_colisiones = 0
n_truncados  = 0

obs, _ = env.reset()

for ep in range(1, N_EPISODIOS + 1):
    obs, _ = env.reset()
    done       = False
    truncated  = False
    pasos      = 0
    reward_ep  = 0.0

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        pasos     += 1
        reward_ep += reward

    es_exito    = bool(info.get("exito",    False))
    es_colision = bool(info.get("colision", False))
    es_truncado = bool(info.get("truncado", False))

    if es_exito:
        n_exitos += 1
        resultado = "exito"
    elif es_colision:
        n_colisiones += 1
        resultado = "colision"
    else:
        n_truncados += 1
        resultado = "truncado"

    resultados.append({
        "episodio":  ep,
        "resultado": resultado,
        "pasos":     pasos,
        "reward":    round(reward_ep, 2),
    })

    print(f"  ep {ep:>3}/{N_EPISODIOS}  {resultado:<10}  pasos={pasos:<5}  reward={reward_ep:.1f}")

# ── resumen ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f" RESUMEN — {N_EPISODIOS} episodios deterministas")
print(f"{'='*55}")
print(f"  Éxito:     {n_exitos:>3} / {N_EPISODIOS}  ({n_exitos / N_EPISODIOS * 100:.1f}%)")
print(f"  Colisión:  {n_colisiones:>3} / {N_EPISODIOS}  ({n_colisiones / N_EPISODIOS * 100:.1f}%)")
print(f"  Truncado:  {n_truncados:>3} / {N_EPISODIOS}  ({n_truncados / N_EPISODIOS * 100:.1f}%)")
pasos_medio = sum(r["pasos"]  for r in resultados) / N_EPISODIOS
reward_medio = sum(r["reward"] for r in resultados) / N_EPISODIOS
print(f"  Pasos/ep:  {pasos_medio:.1f}")
print(f"  Reward/ep: {reward_medio:.1f}")
print(f"{'='*55}\n")

# ── guardar CSV ───────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["episodio", "resultado", "pasos", "reward"])
    writer.writeheader()
    writer.writerows(resultados)

# resumen al final del CSV
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(["RESUMEN", "", "", ""])
    writer.writerow(["exito_%",    f"{n_exitos    / N_EPISODIOS * 100:.1f}", "", ""])
    writer.writerow(["colision_%", f"{n_colisiones / N_EPISODIOS * 100:.1f}", "", ""])
    writer.writerow(["truncado_%", f"{n_truncados  / N_EPISODIOS * 100:.1f}", "", ""])
    writer.writerow(["pasos_medio",  f"{pasos_medio:.1f}", "", ""])
    writer.writerow(["reward_medio", f"{reward_medio:.1f}", "", ""])

print(f"[CSV] Resultados guardados en {CSV_PATH}")

env.supervisor.simulationQuit(0)
