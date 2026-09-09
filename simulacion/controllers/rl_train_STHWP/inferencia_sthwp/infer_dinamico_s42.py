"""
Inferencia con obstáculos dinámicos — STH-WP seed=42.

Idéntico al ciclo completo estático pero con 50 ep/goal (en lugar de 100)
y world file con 2 peatones activos (PEDESTRIAN_1, PEDESTRIAN_2).

  ap+exit : run002_s42_stage4v2_final
  retorno : run002_s42_stage5_final
  50 ep × 28 goals = 1400 ciclos

Salida: inferencia_sthwp/resultados/infer_dinamico_s42.csv
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

MAP_PATH      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "warehouse_map01.json")
MODEL_S4_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pruebas", "run002_s42_stage4v2_final")
MODEL_S5_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pruebas", "run002_s42_stage5_final")
RUN_ID        = "dinamico_s42_sthwp"
HEADING_SIGMA = 0.25
N_POR_GOAL    = 100
OUT_DIR       = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH      = os.path.join(OUT_DIR, "infer_dinamico_s42.csv")

env      = WebotsEnv(map_path=MAP_PATH, stage=4, heading_sigma=HEADING_SIGMA)
model_s4 = PPO.load(MODEL_S4_PATH, env=env)
model_s5 = PPO.load(MODEL_S5_PATH, env=env)

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia DINÁMICA (peatones) — STH-WP | {RUN_ID}")
print(f" ap+exit : {os.path.basename(MODEL_S4_PATH)}")
print(f" retorno : {os.path.basename(MODEL_S5_PATH)}")
print(f" {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} ciclos")
print(f" Obstáculos: PEDESTRIAN_1 (corredor central) + PEDESTRIAN_2 (corredor izq.)")
print(f"{'='*60}\n")

resultados     = []
stats_por_goal = {
    gid: {"exito": 0, "col_approach": 0, "col_exit": 0, "col_retorno": 0,
          "truncado_ap_ex": 0, "truncado_ret": 0}
    for gid in env.goal_ids
}
ep_global = 0

for goal_idx in range(N_GOALS):
    goal_id = env.goal_ids[goal_idx]
    print(f"\n  [{goal_id}] ({goal_idx + 1}/{N_GOALS})")

    for ep in range(1, N_POR_GOAL + 1):
        ep_global += 1

        # ── Fase 1: approach + exit ────────────────────────────────────────────
        env.stage              = 4
        env._max_steps         = 4000
        env.PROB_APPROACH_S4   = 1.0
        env._sample_goal_approach = lambda gi=goal_idx: gi

        obs, _ = env.reset()
        done = truncated = False
        pasos_s4  = 0
        reward_s4 = 0.0

        while not done and not truncated:
            action, _ = model_s4.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            pasos_s4  += 1
            reward_s4 += reward

        exito_s4   = bool(info.get("exito",            False))
        col_s4     = bool(info.get("colision",         False))
        llego_est  = bool(info.get("llego_estanteria", False))
        llego_desc = exito_s4

        pasos_s5  = 0
        reward_s5 = 0.0

        if exito_s4:
            # ── Fase 2: retorno ────────────────────────────────────────────────
            env.stage      = 5
            env._max_steps = 3000

            obs, _ = env.reset()
            done = truncated = False

            while not done and not truncated:
                action, _ = model_s5.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                pasos_s5  += 1
                reward_s5 += reward

            exito_s5 = bool(info.get("exito",    False))
            col_s5   = bool(info.get("colision", False))

            if exito_s5:
                resultado = "exito"
                stats_por_goal[goal_id]["exito"] += 1
            elif col_s5:
                resultado = "col_retorno"
                stats_por_goal[goal_id]["col_retorno"] += 1
            else:
                resultado = "truncado_ret"
                stats_por_goal[goal_id]["truncado_ret"] += 1
        else:
            if col_s4 and not llego_est:
                resultado = "col_approach"
                stats_por_goal[goal_id]["col_approach"] += 1
            elif col_s4 and llego_est:
                resultado = "col_exit"
                stats_por_goal[goal_id]["col_exit"] += 1
            else:
                resultado = "truncado_ap_ex"
                stats_por_goal[goal_id]["truncado_ap_ex"] += 1

        pasos_total  = pasos_s4 + pasos_s5
        reward_total = round(reward_s4 + reward_s5, 2)

        resultados.append({
            "episodio":         ep_global,
            "goal_id":          goal_id,
            "resultado":        resultado,
            "pasos_ap_ex":      pasos_s4,
            "pasos_retorno":    pasos_s5,
            "pasos_total":      pasos_total,
            "reward_ap_ex":     round(reward_s4, 2),
            "reward_retorno":   round(reward_s5, 2),
            "reward_total":     reward_total,
            "llego_estanteria": int(llego_est),
            "llego_descarga":   int(llego_desc),
        })
        print(f"    ep {ep:>2}/{N_POR_GOAL}  {resultado:<20}  "
              f"pasos={pasos_total:<5}  ({pasos_s4}+{pasos_s5})")

# ── Resumen ────────────────────────────────────────────────────────────────────
n_exito    = sum(v["exito"]          for v in stats_por_goal.values())
n_col_ap   = sum(v["col_approach"]   for v in stats_por_goal.values())
n_col_ex   = sum(v["col_exit"]       for v in stats_por_goal.values())
n_col_ret  = sum(v["col_retorno"]    for v in stats_por_goal.values())
n_trunc_ae = sum(v["truncado_ap_ex"] for v in stats_por_goal.values())
n_trunc_r  = sum(v["truncado_ret"]   for v in stats_por_goal.values())
n_llego_d  = sum(1 for r in resultados if r["llego_descarga"])
pasos_medio  = sum(r["pasos_total"]  for r in resultados) / N_TOTAL
reward_medio = sum(r["reward_total"] for r in resultados) / N_TOTAL

print(f"\n{'='*60}")
print(f" RESUMEN — {N_TOTAL} ciclos | {RUN_ID} | PEATONES ACTIVOS")
print(f"{'='*60}")
print(f"  Éxito ciclo completo: {n_exito:>4}/{N_TOTAL}  ({n_exito/N_TOTAL*100:.1f}%)")
print(f"  Colisión approach:    {n_col_ap:>4}/{N_TOTAL}  ({n_col_ap/N_TOTAL*100:.1f}%)")
print(f"  Colisión exit:        {n_col_ex:>4}/{N_TOTAL}  ({n_col_ex/N_TOTAL*100:.1f}%)")
print(f"  Colisión retorno:     {n_col_ret:>4}/{N_TOTAL}  ({n_col_ret/N_TOTAL*100:.1f}%)")
print(f"  Truncado ap/ex:       {n_trunc_ae:>4}/{N_TOTAL}  ({n_trunc_ae/N_TOTAL*100:.1f}%)")
print(f"  Truncado retorno:     {n_trunc_r:>4}/{N_TOTAL}  ({n_trunc_r/N_TOTAL*100:.1f}%)")
print(f"  Llegó descarga:       {n_llego_d:>4}/{N_TOTAL}  ({n_llego_d/N_TOTAL*100:.1f}%)")
print(f"  Pasos/ciclo: {pasos_medio:.1f}  |  Reward/ciclo: {reward_medio:.1f}")
print(f"\n  Por goal:")
for gid, v in stats_por_goal.items():
    tasa = v["exito"] / N_POR_GOAL * 100
    flag = "  ⚠" if tasa < 70 else ""
    print(f"    {gid}: {v['exito']}/{N_POR_GOAL} ({tasa:.0f}%)"
          f"  col_ap={v['col_approach']}  col_ex={v['col_exit']}"
          f"  col_ret={v['col_retorno']}"
          f"  trunc={v['truncado_ap_ex']+v['truncado_ret']}{flag}")
print(f"{'='*60}\n")

os.makedirs(OUT_DIR, exist_ok=True)
FIELDS = [
    "episodio", "goal_id", "resultado",
    "pasos_ap_ex", "pasos_retorno", "pasos_total",
    "reward_ap_ex", "reward_retorno", "reward_total",
    "llego_estanteria", "llego_descarga",
]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(resultados)

PAD = [""] * (len(FIELDS) - 2)
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(["RESUMEN_GLOBAL"] + [""] * (len(FIELDS) - 1))
    writer.writerow(["metodo",             "STH-WP"]                           + PAD)
    writer.writerow(["seed",               "42"]                               + PAD)
    writer.writerow(["obstaculos",         "peatones_activos"]                 + PAD)
    writer.writerow(["n_total",            N_TOTAL]                            + PAD)
    writer.writerow(["exito_%",            f"{n_exito/N_TOTAL*100:.1f}"]       + PAD)
    writer.writerow(["col_approach_%",     f"{n_col_ap/N_TOTAL*100:.1f}"]      + PAD)
    writer.writerow(["col_exit_%",         f"{n_col_ex/N_TOTAL*100:.1f}"]      + PAD)
    writer.writerow(["col_retorno_%",      f"{n_col_ret/N_TOTAL*100:.1f}"]     + PAD)
    writer.writerow(["truncado_ap_ex_%",   f"{n_trunc_ae/N_TOTAL*100:.1f}"]    + PAD)
    writer.writerow(["truncado_ret_%",     f"{n_trunc_r/N_TOTAL*100:.1f}"]     + PAD)
    writer.writerow(["llego_descarga_%",   f"{n_llego_d/N_TOTAL*100:.1f}"]     + PAD)
    writer.writerow(["pasos_medio",        f"{pasos_medio:.1f}"]               + PAD)
    writer.writerow(["reward_medio",       f"{reward_medio:.1f}"]              + PAD)
    writer.writerow([])
    writer.writerow(["RESUMEN_POR_GOAL", "exito", "col_approach", "col_exit",
                     "col_retorno", "truncado_ap_ex", "truncado_ret", "exito_%", "", "", "", ""])
    for gid, v in stats_por_goal.items():
        writer.writerow([
            gid, v["exito"], v["col_approach"], v["col_exit"],
            v["col_retorno"], v["truncado_ap_ex"], v["truncado_ret"],
            f"{v['exito']/N_POR_GOAL*100:.1f}", "", "", "", "",
        ])

print(f"[CSV] Guardado en {CSV_PATH}")
env.supervisor.simulationQuit(0)
