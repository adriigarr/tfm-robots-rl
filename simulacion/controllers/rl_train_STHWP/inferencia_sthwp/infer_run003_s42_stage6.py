"""
Inferencia ciclo completo — run003 seed=42 — stage 6 (modelo unificado).

Un único modelo gestiona approach → exit → return en un solo episodio.
100 episodios por goal × 28 goals = 2800 ciclos.

Clasificación:
  exito        — ciclo completo (zona_espera alcanzada)
  col_approach — colisión antes de llegar a la estantería
  col_exit     — colisión entre estantería y zona_descarga
  col_retorno  — colisión entre zona_descarga y zona_espera
  truncado     — timeout sin colisión

World: warehouse_1_static.wbt (sin peatones).
Salida: inferencia_sthwp/resultados/infer_run003_s42_stage6.csv
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

CONTROLLER_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_PATH   = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
MODEL_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "run003_s42_stage6_final")
RUN_ID     = "run003_s42"
N_POR_GOAL = 100
OUT_DIR    = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH   = os.path.join(OUT_DIR, "infer_run003_s42_stage6.csv")

os.makedirs(OUT_DIR, exist_ok=True)

env   = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25)
env._max_steps = 7000
model = PPO.load(MODEL_PATH, env=env)

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia CICLO COMPLETO stage6 | {RUN_ID}")
print(f" Modelo: {os.path.basename(MODEL_PATH)}")
print(f" {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} ciclos")
print(f" World: warehouse_1_static.wbt (sin peatones)")
print(f"{'='*60}\n")

FIELDS = [
    "episodio", "goal_id", "resultado",
    "pasos_total", "reward_total",
    "llego_estanteria", "llego_descarga",
]

stats_por_goal = {
    gid: {"exito": 0, "col_approach": 0, "col_exit": 0,
          "col_retorno": 0, "truncado": 0}
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
        done = truncated = False
        pasos = 0
        total_reward = 0.0
        reached_shelf    = False
        reached_descarga = False

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            pasos += 1
            total_reward += reward
            # rastrear fases internas
            if env._hacia_descarga:
                reached_shelf = True
            if env._en_retorno:
                reached_descarga = True

        exito   = bool(info.get("exito",    False))
        colision = bool(info.get("colision", False))

        if exito:
            resultado = "exito"
            stats_por_goal[goal_id]["exito"] += 1
        elif colision:
            if not reached_shelf:
                resultado = "col_approach"
                stats_por_goal[goal_id]["col_approach"] += 1
            elif not reached_descarga:
                resultado = "col_exit"
                stats_por_goal[goal_id]["col_exit"] += 1
            else:
                resultado = "col_retorno"
                stats_por_goal[goal_id]["col_retorno"] += 1
        else:
            resultado = "truncado"
            stats_por_goal[goal_id]["truncado"] += 1

        row = {
            "episodio":         ep_global,
            "goal_id":          goal_id,
            "resultado":        resultado,
            "pasos_total":      pasos,
            "reward_total":     round(total_reward, 2),
            "llego_estanteria": int(reached_shelf),
            "llego_descarga":   int(reached_descarga),
        }
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

        print(f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<15}  pasos={pasos}")

# ── Resumen ──────────────────────────────────────────────────────────────────
n_exito   = sum(v["exito"]       for v in stats_por_goal.values())
n_col_ap  = sum(v["col_approach"] for v in stats_por_goal.values())
n_col_ex  = sum(v["col_exit"]    for v in stats_por_goal.values())
n_col_ret = sum(v["col_retorno"] for v in stats_por_goal.values())
n_trunc   = sum(v["truncado"]    for v in stats_por_goal.values())

print(f"\n{'='*60}")
print(f" RESUMEN — {N_TOTAL} ciclos | {RUN_ID}")
print(f"{'='*60}")
print(f"  Éxito:            {n_exito:>4}/{N_TOTAL}  ({n_exito/N_TOTAL*100:.1f}%)")
print(f"  Col. approach:    {n_col_ap:>4}/{N_TOTAL}  ({n_col_ap/N_TOTAL*100:.1f}%)")
print(f"  Col. exit:        {n_col_ex:>4}/{N_TOTAL}  ({n_col_ex/N_TOTAL*100:.1f}%)")
print(f"  Col. retorno:     {n_col_ret:>4}/{N_TOTAL}  ({n_col_ret/N_TOTAL*100:.1f}%)")
print(f"  Truncado:         {n_trunc:>4}/{N_TOTAL}  ({n_trunc/N_TOTAL*100:.1f}%)")
print(f"\n  Por goal:")
for gid, v in stats_por_goal.items():
    tasa = v["exito"] / N_POR_GOAL * 100
    flag = "  ⚠" if tasa < 80 else ""
    print(f"    {gid}: {v['exito']}/{N_POR_GOAL} ({tasa:.0f}%)"
          f"  col_ap={v['col_approach']}  col_ex={v['col_exit']}"
          f"  col_ret={v['col_retorno']}  trunc={v['truncado']}{flag}")
print(f"{'='*60}\n")
print(f"[CSV] {CSV_PATH}")

env.supervisor.simulationQuit(0)
