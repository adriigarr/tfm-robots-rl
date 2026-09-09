"""
Inferencia stage 2 — run002 seed=524.

28 goals, approach puro, política determinista. 100 episodios por goal (2800 total).

Además del CSV habitual, genera:
  resultados/arrival_headings_stage2.json
    → heading real del robot (rad) al llegar a cada estantería.
    → Usado por webots_env.py en stage 3 para el teleport con orientación realista.

Salida: inferencia_sthwp/resultados/infer_run002_s524_stage2.csv
        inferencia_sthwp/resultados/arrival_headings_stage2.json
"""

import os
import sys
import csv
import json
import math
import cmath

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

# ── configuración ─────────────────────────────────────────────────────────────
MAP_PATH    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "warehouse_map01.json")
MODEL_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pruebas", "run002_s524_stage2_final")
RUN_ID      = "run002_s524"
STAGE       = 2
N_POR_GOAL  = 100
OUT_DIR     = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH    = os.path.join(OUT_DIR, f"infer_{RUN_ID}_stage{STAGE}.csv")
HEADINGS_JSON_PATH = os.path.join(OUT_DIR, "arrival_headings_stage2.json")

# ── helpers ángulos circulares ─────────────────────────────────────────────────

def _read_robot_heading(env):
    """Lee el heading actual del robot desde Webots (radianes, [-π, π])."""
    rot = env.robot_node.getField("rotation").getSFRotation()
    ax, ay, az, angle = rot
    # Convención: [0, 1, 0, -heading] → heading = -angle
    # Equivalente: [0, -1, 0, heading] → heading = angle
    heading = -angle if ay > 0 else angle
    return (heading + math.pi) % (2 * math.pi) - math.pi


def _circular_mean(angles):
    if not angles:
        return 0.0
    return cmath.phase(sum(cmath.exp(1j * a) for a in angles))


def _circular_std(angles):
    if len(angles) < 2:
        return 0.0
    R = abs(sum(cmath.exp(1j * a) for a in angles)) / len(angles)
    return math.sqrt(max(0.0, -2.0 * math.log(R + 1e-12)))


# ── cargar entorno y modelo ───────────────────────────────────────────────────
env   = WebotsEnv(map_path=MAP_PATH, stage=STAGE)
model = PPO.load(MODEL_PATH, env=env)

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia stage {STAGE} | {RUN_ID} | {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} total")
print(f" Registra heading de llegada → {HEADINGS_JSON_PATH}")
print(f"{'='*60}\n")

# ── bucle de inferencia ───────────────────────────────────────────────────────
resultados        = []
stats_por_goal    = {gid: {"exito": 0, "colision": 0, "truncado": 0} for gid in env.goal_ids}
headings_por_goal = {gid: [] for gid in env.goal_ids}   # solo episodios con éxito
ep_global         = 0

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

        # Heading de llegada: solo cuando el robot llega a la estantería
        heading_llegada = float("nan")
        if es_exito:
            heading_llegada = _read_robot_heading(env)
            headings_por_goal[goal_id].append(heading_llegada)

        if es_exito:
            resultado = "exito";    stats_por_goal[goal_id]["exito"] += 1
        elif es_colision:
            resultado = "colision"; stats_por_goal[goal_id]["colision"] += 1
        else:
            resultado = "truncado"; stats_por_goal[goal_id]["truncado"] += 1

        resultados.append({
            "episodio":           ep_global,
            "goal_id":            goal_id,
            "tipo_episodio":      "approach",
            "resultado":          resultado,
            "pasos":              pasos,
            "reward":             round(reward_ep, 2),
            "llego_estanteria":   bool(info.get("llego_estanteria", False)),
            "heading_llegada_rad": round(heading_llegada, 4) if not math.isnan(heading_llegada) else "",
        })

        print(f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<10}  pasos={pasos:<5}  "
              f"heading={f'{math.degrees(heading_llegada):.1f}°' if not math.isnan(heading_llegada) else 'n/a':<8}  "
              f"reward={reward_ep:.1f}")

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
print(f"\n  Headings de llegada (media ± σ) por goal:")
for gid in env.goal_ids:
    hs = headings_por_goal[gid]
    if hs:
        mu  = _circular_mean(hs)
        sig = _circular_std(hs)
        print(f"    {gid}: n={len(hs):>3}  μ={math.degrees(mu):>7.1f}°  σ={math.degrees(sig):>5.1f}°")
    else:
        print(f"    {gid}: sin éxito — no hay heading registrado")
print(f"{'='*60}\n")

# ── guardar CSV ───────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
FIELDS = ["episodio", "goal_id", "tipo_episodio", "resultado", "pasos",
          "reward", "llego_estanteria", "heading_llegada_rad"]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(resultados)

with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(["RESUMEN_GLOBAL", "", "", "", "", "", "", ""])
    writer.writerow(["n_total",      N_TOTAL, "", "", "", "", "", ""])
    writer.writerow(["exito_%",      f"{n_exitos    / N_TOTAL * 100:.1f}", "", "", "", "", "", ""])
    writer.writerow(["colision_%",   f"{n_colisiones / N_TOTAL * 100:.1f}", "", "", "", "", "", ""])
    writer.writerow(["truncado_%",   f"{n_truncados  / N_TOTAL * 100:.1f}", "", "", "", "", "", ""])
    writer.writerow(["pasos_medio",  f"{pasos_medio:.1f}", "", "", "", "", "", ""])
    writer.writerow(["reward_medio", f"{reward_medio:.1f}", "", "", "", "", "", ""])
    writer.writerow([])
    writer.writerow(["RESUMEN_POR_GOAL", "tipo", "exito", "colision", "truncado",
                     "exito_%", "heading_mean_deg", "heading_sigma_deg"])
    for gid in env.goal_ids:
        v  = stats_por_goal[gid]
        hs = headings_por_goal[gid]
        if hs:
            mu  = math.degrees(_circular_mean(hs))
            sig = math.degrees(_circular_std(hs))
        else:
            mu, sig = float("nan"), float("nan")
        writer.writerow([gid, "approach", v["exito"], v["colision"], v["truncado"],
                         f"{v['exito'] / N_POR_GOAL * 100:.1f}",
                         f"{mu:.1f}" if not math.isnan(mu) else "n/a",
                         f"{sig:.1f}" if not math.isnan(sig) else "n/a"])

print(f"[CSV] Guardado en {CSV_PATH}")

# ── guardar JSON de headings de llegada ───────────────────────────────────────
heading_stats = {}
for gid in env.goal_ids:
    hs = headings_por_goal[gid]
    if hs:
        mu  = _circular_mean(hs)
        sig = _circular_std(hs)
        heading_stats[gid] = {
            "mean_rad":  round(mu, 5),
            "sigma_rad": round(sig, 5),
            "mean_deg":  round(math.degrees(mu), 2),
            "sigma_deg": round(math.degrees(sig), 2),
            "n":         len(hs),
        }
    else:
        heading_stats[gid] = None   # sin datos → webots_env usará fallback

with open(HEADINGS_JSON_PATH, "w") as f:
    json.dump(heading_stats, f, indent=2)

print(f"[JSON] Headings de llegada guardados en {HEADINGS_JSON_PATH}")

env.supervisor.simulationQuit(0)
