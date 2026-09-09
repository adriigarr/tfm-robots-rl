"""
Inferencia stage 5 — SUB-WP seed=42 — Return puro, política determinista.

300 episodios (ruta fija descarga → espera, sin variabilidad de goal).
Salida: inferencia_subwp/resultados/infer_subwp_s524_wp75_stage5.csv
"""

import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

MAP_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "warehouse_map01.json")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pruebas", "subwp_s524_wp75_stage5_final")
RUN_ID     = "subwp_s524_wp75"
STAGE      = 5
N_EP       = 300
OUT_DIR    = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH   = os.path.join(OUT_DIR, "infer_subwp_s524_wp75_stage5.csv")

env   = WebotsEnv(map_path=MAP_PATH, stage=STAGE)
model = PPO.load(MODEL_PATH, env=env)
env._max_steps = 3000

print(f"\n{'='*60}")
print(f" Inferencia stage 5 SUB-WP | {RUN_ID} | {N_EP} episodios")
print(f" Return puro: descarga → espera")
print(f"{'='*60}\n")

resultados = []
n_exitos = n_colisiones = n_truncados = 0

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

    if es_exito:
        resultado = "exito";    n_exitos    += 1
    elif es_colision:
        resultado = "colision"; n_colisiones += 1
    else:
        resultado = "truncado"; n_truncados  += 1

    resultados.append({"episodio": ep, "resultado": resultado,
                        "pasos": pasos, "reward": round(reward_ep, 2)})
    print(f"  ep {ep:>3}/{N_EP}  {resultado:<10}  pasos={pasos:<5}  reward={reward_ep:.1f}")

pasos_medio  = sum(r["pasos"]  for r in resultados) / N_EP
reward_medio = sum(r["reward"] for r in resultados) / N_EP

print(f"\n{'='*60}")
print(f" RESUMEN — {N_EP} ep | {RUN_ID} stage5")
print(f"  Éxito:    {n_exitos:>3}/{N_EP}  ({n_exitos/N_EP*100:.1f}%)")
print(f"  Colisión: {n_colisiones:>3}/{N_EP}  ({n_colisiones/N_EP*100:.1f}%)")
print(f"  Truncado: {n_truncados:>3}/{N_EP}  ({n_truncados/N_EP*100:.1f}%)")
print(f"  Pasos/ep: {pasos_medio:.1f}  |  Reward/ep: {reward_medio:.1f}")
print(f"{'='*60}\n")

os.makedirs(OUT_DIR, exist_ok=True)
FIELDS = ["episodio", "resultado", "pasos", "reward"]
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader(); writer.writerows(resultados)
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(["RESUMEN_GLOBAL", "", "", ""])
    writer.writerow(["n_total",      N_EP, "", ""])
    writer.writerow(["exito_%",      f"{n_exitos    / N_EP * 100:.1f}", "", ""])
    writer.writerow(["colision_%",   f"{n_colisiones / N_EP * 100:.1f}", "", ""])
    writer.writerow(["truncado_%",   f"{n_truncados  / N_EP * 100:.1f}", "", ""])
    writer.writerow(["pasos_medio",  f"{pasos_medio:.1f}", "", ""])
    writer.writerow(["reward_medio", f"{reward_medio:.1f}", "", ""])
print(f"[CSV] {CSV_PATH}")
env.supervisor.simulationQuit(0)
