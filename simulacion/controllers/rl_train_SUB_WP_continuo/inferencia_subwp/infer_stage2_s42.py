"""
Inferencia stage 2 — SUB-WP seed=42 — Approach puro, todos los goals.

Mide el heading real de llegada a cada estantería (ángulo del compás en el
momento de éxito). El JSON resultante se usa en stage 3 para el teleport.

100 episodios por goal × 28 goals = 2800 episodios.

Salidas:
  inferencia_subwp/resultados/infer_subwp_s42_stage2.csv
  inferencia_subwp/resultados/arrival_headings_stage2.json   ← compartido por los 3 seeds
"""

import os
import sys
import csv
import json
import math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

MAP_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "warehouse_map01.json")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pruebas", "subwp_s42_stage2_final")
RUN_ID     = "subwp_s42"
STAGE      = 2
N_POR_GOAL = 100
OUT_DIR    = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH   = os.path.join(OUT_DIR, "infer_subwp_s42_stage2.csv")
JSON_PATH  = os.path.join(OUT_DIR, "arrival_headings_stage2.json")

env   = WebotsEnv(map_path=MAP_PATH, stage=STAGE)
model = PPO.load(MODEL_PATH, env=env)

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia stage 2 SUB-WP | {RUN_ID}")
print(f" {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} episodios")
print(f" Midiendo headings reales de llegada...")
print(f"{'='*60}\n")

resultados     = []
stats_por_goal = {gid: {"exito": 0, "colision": 0, "truncado": 0}
                  for gid in env.goal_ids}
headings_raw   = defaultdict(list)   # goal_id → lista de headings en llegada
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
        last_heading = None

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            pasos     += 1
            reward_ep += reward

            # capturar heading continuo; al llegar lo tendremos actualizado
            compass_val  = env.compass.getValues()
            last_heading = math.atan2(compass_val[0], compass_val[1])

        es_exito    = bool(info.get("exito",    False))
        es_colision = bool(info.get("colision", False))

        if es_exito:
            resultado = "exito"
            stats_por_goal[goal_id]["exito"] += 1
            if last_heading is not None:
                headings_raw[goal_id].append(last_heading)
        elif es_colision:
            resultado = "colision"
            stats_por_goal[goal_id]["colision"] += 1
        else:
            resultado = "truncado"
            stats_por_goal[goal_id]["truncado"] += 1

        resultados.append({
            "episodio":      ep_global,
            "goal_id":       goal_id,
            "resultado":     resultado,
            "pasos":         pasos,
            "reward":        round(reward_ep, 2),
            "heading_rad":   round(last_heading, 4) if last_heading is not None else "",
        })
        print(f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<12}  pasos={pasos:<5}  "
              f"heading={math.degrees(last_heading):.1f}°" if last_heading else
              f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<12}  pasos={pasos:<5}")

n_exito   = sum(v["exito"]    for v in stats_por_goal.values())
n_col     = sum(v["colision"] for v in stats_por_goal.values())
n_trunc   = sum(v["truncado"] for v in stats_por_goal.values())

print(f"\n{'='*60}")
print(f" RESUMEN — {N_TOTAL} ep | {RUN_ID} stage2")
print(f"  Éxito:    {n_exito:>4}/{N_TOTAL}  ({n_exito/N_TOTAL*100:.1f}%)")
print(f"  Colisión: {n_col:>4}/{N_TOTAL}  ({n_col/N_TOTAL*100:.1f}%)")
print(f"  Truncado: {n_trunc:>4}/{N_TOTAL}  ({n_trunc/N_TOTAL*100:.1f}%)")
print(f"\n  Por goal:")
for gid, v in stats_por_goal.items():
    tasa = v["exito"] / N_POR_GOAL * 100
    n_h  = len(headings_raw[gid])
    flag = "  ⚠" if tasa < 70 else ""
    print(f"    {gid}: {v['exito']}/{N_POR_GOAL} ({tasa:.0f}%)  headings={n_h}{flag}")
print(f"{'='*60}\n")

os.makedirs(OUT_DIR, exist_ok=True)

# CSV de resultados
FIELDS = ["episodio", "goal_id", "resultado", "pasos", "reward", "heading_rad"]
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(resultados)
print(f"[CSV] Guardado en {CSV_PATH}")

# JSON de headings: media y std por goal
def _mean_angle(angles):
    """Media circular para ángulos en radianes."""
    if not angles:
        return None
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    return math.atan2(sin_sum / len(angles), cos_sum / len(angles))

def _std_angle(angles, mean_a):
    if len(angles) < 2:
        return 0.0
    diffs = [((a - mean_a + math.pi) % (2 * math.pi)) - math.pi for a in angles]
    return math.sqrt(sum(d * d for d in diffs) / (len(diffs) - 1))

headings_json = {}
for gid in env.goal_ids:
    hs = headings_raw[gid]
    if hs:
        mean_r = _mean_angle(hs)
        std_r  = _std_angle(hs, mean_r)
        headings_json[gid] = {
            "mean_rad":   round(mean_r, 4),
            "std_rad":    round(std_r, 4),
            "mean_deg":   round(math.degrees(mean_r), 1),
            "n_samples":  len(hs),
        }
    else:
        headings_json[gid] = {"mean_rad": None, "std_rad": None,
                               "mean_deg": None, "n_samples": 0}
        print(f"[WARN] {gid}: sin headings medidos (0 éxitos)")

with open(JSON_PATH, "w") as f:
    json.dump(headings_json, f, indent=2)
print(f"[JSON] Headings guardados en {JSON_PATH}")

env.supervisor.simulationQuit(0)
