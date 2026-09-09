"""
Inferencia stage 4 — run002 seed=524 — CICLO COMPLETO ENCADENADO.

Cada episodio es un ciclo completo: approach (zona espera → estantería) +
exit (estantería → zona descarga) sin reset intermedio. El entorno stage=4
ya implementa esta transición automática en step() cuando llega a la estantería.

  PROB_APPROACH_S4 = 1.0  →  todos los episodios son ciclo completo
  Éxito = robot llega a la zona de descarga (ambas fases completadas)

Resultados por episodio:
  - "exito"            : ciclo completo exitoso (approach + exit)
  - "colision_approach": colisión antes de llegar a la estantería
  - "colision_exit"    : colisión después de llegar a la estantería
  - "truncado"         : timeout (2500 steps)

100 episodios por goal × 28 goals = 2800 episodios totales.
Salida: inferencia_sthwp/resultados/infer_run002_s524_stage4.csv
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

# ── configuración ─────────────────────────────────────────────────────────────
MAP_PATH      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "warehouse_map01.json")
MODEL_PATH    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pruebas", "run002_s524_stage4_final")
RUN_ID        = "run002_s524"
STAGE         = 4
HEADING_SIGMA = 0.25
N_POR_GOAL    = 100
OUT_DIR       = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH      = os.path.join(OUT_DIR, f"infer_{RUN_ID}_stage{STAGE}.csv")

# ── cargar entorno y modelo ───────────────────────────────────────────────────
env   = WebotsEnv(map_path=MAP_PATH, stage=STAGE, heading_sigma=HEADING_SIGMA)
model = PPO.load(MODEL_PATH, env=env)

# Forzar ciclo completo: todos los episodios son approach → exit encadenado.
# step() transiciona automáticamente cuando el robot llega a la estantería.
env.PROB_APPROACH_S4 = 1.0

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia stage {STAGE} — CICLO COMPLETO ENCADENADO | {RUN_ID}")
print(f" {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} episodios")
print(f" Éxito = approach + exit completados en un único episodio")
print(f"{'='*60}\n")

# ── bucle de inferencia ───────────────────────────────────────────────────────
resultados     = []
stats_por_goal = {
    gid: {"exito": 0, "colision_approach": 0, "colision_exit": 0, "truncado": 0}
    for gid in env.goal_ids
}
ep_global = 0

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
        llego_est   = bool(info.get("llego_estanteria", False))

        if es_exito:
            # Ambas fases completadas
            resultado = "exito"
            stats_por_goal[goal_id]["exito"] += 1
        elif es_colision and not llego_est:
            # Colisión antes de llegar a la estantería → fallo en approach
            resultado = "colision_approach"
            stats_por_goal[goal_id]["colision_approach"] += 1
        elif es_colision and llego_est:
            # Colisión después de llegar a la estantería → fallo en exit
            resultado = "colision_exit"
            stats_por_goal[goal_id]["colision_exit"] += 1
        else:
            resultado = "truncado"
            stats_por_goal[goal_id]["truncado"] += 1

        resultados.append({
            "episodio":        ep_global,
            "goal_id":         goal_id,
            "tipo_episodio":   "ciclo_completo",
            "resultado":       resultado,
            "pasos":           pasos,
            "reward":          round(reward_ep, 2),
            "llego_estanteria": llego_est,
        })

        print(f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<20}  pasos={pasos:<5}  "
              f"llego={llego_est}  reward={reward_ep:.1f}")

# ── resumen global ─────────────────────────────────────────────────────────────
n_exito   = sum(v["exito"]            for v in stats_por_goal.values())
n_col_ap  = sum(v["colision_approach"] for v in stats_por_goal.values())
n_col_ex  = sum(v["colision_exit"]     for v in stats_por_goal.values())
n_trunc   = sum(v["truncado"]          for v in stats_por_goal.values())
n_col_tot = n_col_ap + n_col_ex
n_llego   = sum(1 for r in resultados if r["llego_estanteria"])

pasos_medio  = sum(r["pasos"]  for r in resultados) / N_TOTAL
reward_medio = sum(r["reward"] for r in resultados) / N_TOTAL

print(f"\n{'='*60}")
print(f" RESUMEN GLOBAL — {N_TOTAL} episodios deterministas (ciclo completo)")
print(f"{'='*60}")
print(f"  Éxito ciclo completo: {n_exito:>4} / {N_TOTAL}  ({n_exito / N_TOTAL * 100:.1f}%)")
print(f"  Colisión (approach):  {n_col_ap:>4} / {N_TOTAL}  ({n_col_ap / N_TOTAL * 100:.1f}%)")
print(f"  Colisión (exit):      {n_col_ex:>4} / {N_TOTAL}  ({n_col_ex / N_TOTAL * 100:.1f}%)")
print(f"  Colisión (total):     {n_col_tot:>4} / {N_TOTAL}  ({n_col_tot / N_TOTAL * 100:.1f}%)")
print(f"  Truncado:             {n_trunc:>4} / {N_TOTAL}  ({n_trunc / N_TOTAL * 100:.1f}%)")
print(f"  Llegó estantería:     {n_llego:>4} / {N_TOTAL}  ({n_llego / N_TOTAL * 100:.1f}%)")
print(f"  Pasos/ep:             {pasos_medio:.1f}")
print(f"  Reward/ep:            {reward_medio:.1f}")
print(f"\n  Por goal:")
for gid, v in stats_por_goal.items():
    tasa = v["exito"] / N_POR_GOAL * 100
    flag = "  ⚠" if tasa < 70 else ""
    print(f"    {gid}: {v['exito']}/{N_POR_GOAL} exito ({tasa:.0f}%)  "
          f"col_ap={v['colision_approach']}  col_ex={v['colision_exit']}  "
          f"trunc={v['truncado']}{flag}")
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
    writer.writerow(["n_total",              N_TOTAL,  "", "", "", "", ""])
    writer.writerow(["exito_%",              f"{n_exito   / N_TOTAL * 100:.1f}", "", "", "", "", ""])
    writer.writerow(["colision_approach_%",  f"{n_col_ap  / N_TOTAL * 100:.1f}", "", "", "", "", ""])
    writer.writerow(["colision_exit_%",      f"{n_col_ex  / N_TOTAL * 100:.1f}", "", "", "", "", ""])
    writer.writerow(["colision_total_%",     f"{n_col_tot / N_TOTAL * 100:.1f}", "", "", "", "", ""])
    writer.writerow(["truncado_%",           f"{n_trunc   / N_TOTAL * 100:.1f}", "", "", "", "", ""])
    writer.writerow(["llego_estanteria_%",   f"{n_llego   / N_TOTAL * 100:.1f}", "", "", "", "", ""])
    writer.writerow(["pasos_medio",          f"{pasos_medio:.1f}", "", "", "", "", ""])
    writer.writerow(["reward_medio",         f"{reward_medio:.1f}", "", "", "", "", ""])
    writer.writerow([])
    writer.writerow(["RESUMEN_POR_GOAL", "exito", "col_approach", "col_exit", "truncado", "exito_%", ""])
    for gid, v in stats_por_goal.items():
        writer.writerow([gid, v["exito"], v["colision_approach"], v["colision_exit"],
                         v["truncado"], f"{v['exito'] / N_POR_GOAL * 100:.1f}", ""])

print(f"[CSV] Guardado en {CSV_PATH}")

env.supervisor.simulationQuit(0)
