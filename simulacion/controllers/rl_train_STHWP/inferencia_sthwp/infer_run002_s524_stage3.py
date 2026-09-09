"""
Inferencia stage 3 — run002 seed=524.

28 goals, exit puro con teleport (heading teórico + ruido σ=0.25), política determinista.
30 episodios por goal (840 total).
Salida: inferencia_sthwp/resultados/infer_run002_s524_stage3.csv
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

# ── configuración ─────────────────────────────────────────────────────────────
MAP_PATH      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "warehouse_map01.json")
MODEL_PATH    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pruebas", "run002_s524_stage3_final")
RUN_ID        = "run002_s524"
STAGE         = 3
HEADING_SIGMA = 0.25
N_POR_GOAL    = 100
OUT_DIR       = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH      = os.path.join(OUT_DIR, f"infer_{RUN_ID}_stage{STAGE}.csv")

# ── cargar entorno y modelo ───────────────────────────────────────────────────
env   = WebotsEnv(map_path=MAP_PATH, stage=STAGE, heading_sigma=HEADING_SIGMA)
model = PPO.load(MODEL_PATH, env=env)

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia stage {STAGE} | {RUN_ID} | {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} total")
print(f" heading_sigma={HEADING_SIGMA} rad ({HEADING_SIGMA * 57.3:.1f}°)")
print(f"{'='*60}\n")

# ── bucle de inferencia ───────────────────────────────────────────────────────
resultados     = []
stats_por_goal = {gid: {"exito": 0, "colision": 0, "truncado": 0} for gid in env.goal_ids}
ep_global      = 0

for goal_idx in range(N_GOALS):
    goal_id = env.goal_ids[goal_idx]
    env._sample_goal_exit = lambda gi=goal_idx: gi

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
            resultado = "exito";    stats_por_goal[goal_id]["exito"] += 1
        elif es_colision:
            resultado = "colision"; stats_por_goal[goal_id]["colision"] += 1
        else:
            resultado = "truncado"; stats_por_goal[goal_id]["truncado"] += 1

        resultados.append({
            "episodio":        ep_global,
            "goal_id":         goal_id,
            "tipo_episodio":   "exit",
            "resultado":       resultado,
            "pasos":           pasos,
            "reward":          round(reward_ep, 2),
            "llego_estanteria": bool(info.get("llego_estanteria", True)),
        })

        print(f"    ep {ep:>2}/{N_POR_GOAL}  {resultado:<10}  pasos={pasos:<5}  reward={reward_ep:.1f}")

# ── resumen global ─────────────────────────────────────────────────────────────
n_exitos     = sum(v["exito"]    for v in stats_por_goal.values())
n_colisiones = sum(v["colision"] for v in stats_por_goal.values())
n_truncados  = sum(v["truncado"] for v in stats_por_goal.values())
pasos_medio  = sum(r["pasos"]   for r in resultados) / N_TOTAL
reward_medio = sum(r["reward"]  for r in resultados) / N_TOTAL

print(f"\n{'='*60}")
print(f" RESUMEN GLOBAL — {N_TOTAL} episodios deterministas")
print(f"{'='*60}")
print(f"  Éxito:     {n_exitos:>4} / {N_TOTAL}  ({n_exitos / N_TOTAL * 100:.1f}%)")
print(f"  Colisión:  {n_colisiones:>4} / {N_TOTAL}  ({n_colisiones / N_TOTAL * 100:.1f}%)")
print(f"  Truncado:  {n_truncados:>4} / {N_TOTAL}  ({n_truncados / N_TOTAL * 100:.1f}%)")
print(f"  Pasos/ep:  {pasos_medio:.1f}")
print(f"  Reward/ep: {reward_medio:.1f}")
print(f"\n  Por goal:")
for gid, v in stats_por_goal.items():
    tasa = v["exito"] / N_POR_GOAL * 100
    flag = "  ⚠" if tasa < 70 else ""
    print(f"    {gid}: {v['exito']}/{N_POR_GOAL} ({tasa:.0f}%){flag}")
print(f"{'='*60}\n")

# ── guardar CSV ───────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
FIELDS = ["episodio", "goal_id", "tipo_episodio", "resultado", "pasos", "reward", "llego_estanteria"]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(resultados)

with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(["RESUMEN_GLOBAL", "", "", "", "", "", ""])
    writer.writerow(["n_total",      N_TOTAL, "", "", "", "", ""])
    writer.writerow(["exito_%",      f"{n_exitos    / N_TOTAL * 100:.1f}", "", "", "", "", ""])
    writer.writerow(["colision_%",   f"{n_colisiones / N_TOTAL * 100:.1f}", "", "", "", "", ""])
    writer.writerow(["truncado_%",   f"{n_truncados  / N_TOTAL * 100:.1f}", "", "", "", "", ""])
    writer.writerow(["pasos_medio",  f"{pasos_medio:.1f}", "", "", "", "", ""])
    writer.writerow(["reward_medio", f"{reward_medio:.1f}", "", "", "", "", ""])
    writer.writerow([])
    writer.writerow(["RESUMEN_POR_GOAL", "tipo", "exito", "colision", "truncado", "exito_%", ""])
    for gid, v in stats_por_goal.items():
        writer.writerow([gid, "exit", v["exito"], v["colision"], v["truncado"],
                         f"{v['exito'] / N_POR_GOAL * 100:.1f}", ""])

print(f"[CSV] Guardado en {CSV_PATH}")

env.supervisor.simulationQuit(0)
