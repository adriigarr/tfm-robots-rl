"""
Inferencia ciclo completo — SUB-WP seed=123.

Modelo: subwp_s123_wp75_stage5_final (conoce approach + exit + return)
Stage=6: approach → exit → return en un único episodio encadenado.
100 episodios por goal × 28 goals = 2800 episodios totales.

Clasificación de resultados:
  exito            — ciclo completo (espera → estantería → descarga → espera)
  colision_approach — bumper durante approach
  colision_exit     — bumper durante exit
  colision_return   — bumper durante return
  truncado          — timeout sin completar el ciclo

Salida: inferencia_subwp/resultados/infer_subwp_s123_wp75_ciclo.csv
"""

import os, sys, csv, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

MAP_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "warehouse_map01.json")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pruebas",
                          "subwp_s123_wp75_stage5_final")
RUN_ID      = "subwp_s123_wp75"
N_POR_GOAL  = 100
OUT_DIR     = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH    = os.path.join(OUT_DIR, "infer_subwp_s123_wp75_ciclo.csv")

env   = WebotsEnv(map_path=MAP_PATH, stage=6)
model = PPO.load(MODEL_PATH, env=env)
env._max_steps = 6000   # approach + exit + return (~4200 pasos esperados)

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia CICLO COMPLETO SUB-WP | {RUN_ID}")
print(f" Modelo: stage5_final  |  Stage=6  |  det=True")
print(f" {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} episodios")
print(f"{'='*60}\n")

resultados     = []
stats_por_goal = {
    gid: {"exito": 0, "col_approach": 0, "col_exit": 0, "col_return": 0, "truncado": 0}
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
        pasos = 0; reward_ep = 0.0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            pasos += 1; reward_ep += reward

        es_exito    = bool(info.get("exito",    False))
        es_colision = bool(info.get("colision", False))

        if es_exito:
            resultado = "exito"
            stats_por_goal[goal_id]["exito"] += 1
        elif es_colision:
            if env._en_retorno:
                resultado = "colision_return"
                stats_por_goal[goal_id]["col_return"] += 1
            elif env._hacia_descarga:
                resultado = "colision_exit"
                stats_por_goal[goal_id]["col_exit"] += 1
            else:
                resultado = "colision_approach"
                stats_por_goal[goal_id]["col_approach"] += 1
        else:
            resultado = "truncado"
            stats_por_goal[goal_id]["truncado"] += 1

        resultados.append({
            "episodio":  ep_global,
            "goal_id":   goal_id,
            "resultado": resultado,
            "pasos":     pasos,
            "reward":    round(reward_ep, 2),
        })
        print(f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<20}  pasos={pasos:<5}  reward={reward_ep:.1f}")

n_exito   = sum(v["exito"]       for v in stats_por_goal.values())
n_col_ap  = sum(v["col_approach"] for v in stats_por_goal.values())
n_col_ex  = sum(v["col_exit"]     for v in stats_por_goal.values())
n_col_ret = sum(v["col_return"]   for v in stats_por_goal.values())
n_trunc   = sum(v["truncado"]     for v in stats_por_goal.values())
n_col_tot = n_col_ap + n_col_ex + n_col_ret
pasos_medio  = sum(r["pasos"]  for r in resultados) / N_TOTAL
reward_medio = sum(r["reward"] for r in resultados) / N_TOTAL

print(f"\n{'='*60}")
print(f" RESUMEN — {N_TOTAL} ep | {RUN_ID} ciclo completo")
print(f"{'='*60}")
print(f"  Éxito ciclo completo:  {n_exito:>4}/{N_TOTAL}  ({n_exito/N_TOTAL*100:.1f}%)")
print(f"  Colisión (approach):   {n_col_ap:>4}/{N_TOTAL}  ({n_col_ap/N_TOTAL*100:.1f}%)")
print(f"  Colisión (exit):       {n_col_ex:>4}/{N_TOTAL}  ({n_col_ex/N_TOTAL*100:.1f}%)")
print(f"  Colisión (return):     {n_col_ret:>4}/{N_TOTAL}  ({n_col_ret/N_TOTAL*100:.1f}%)")
print(f"  Colisión (total):      {n_col_tot:>4}/{N_TOTAL}  ({n_col_tot/N_TOTAL*100:.1f}%)")
print(f"  Truncado:              {n_trunc:>4}/{N_TOTAL}  ({n_trunc/N_TOTAL*100:.1f}%)")
print(f"  Pasos/ep: {pasos_medio:.1f}  |  Reward/ep: {reward_medio:.1f}")
print(f"\n  Por goal:")
for gid, v in stats_por_goal.items():
    tasa = v["exito"] / N_POR_GOAL * 100
    flag = "  ⚠" if tasa < 70 else ""
    print(f"    {gid}: {v['exito']}/{N_POR_GOAL} ({tasa:.0f}%)"
          f"  col_ap={v['col_approach']}  col_ex={v['col_exit']}"
          f"  col_ret={v['col_return']}  trunc={v['truncado']}{flag}")
print(f"{'='*60}\n")

os.makedirs(OUT_DIR, exist_ok=True)
FIELDS = ["episodio", "goal_id", "resultado", "pasos", "reward"]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(resultados)

with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(["RESUMEN_GLOBAL", "", "", "", ""])
    writer.writerow(["n_total",            N_TOTAL,  "", "", ""])
    writer.writerow(["exito_%",            f"{n_exito   / N_TOTAL * 100:.1f}", "", "", ""])
    writer.writerow(["colision_approach_%", f"{n_col_ap  / N_TOTAL * 100:.1f}", "", "", ""])
    writer.writerow(["colision_exit_%",     f"{n_col_ex  / N_TOTAL * 100:.1f}", "", "", ""])
    writer.writerow(["colision_return_%",   f"{n_col_ret / N_TOTAL * 100:.1f}", "", "", ""])
    writer.writerow(["colision_total_%",    f"{n_col_tot / N_TOTAL * 100:.1f}", "", "", ""])
    writer.writerow(["truncado_%",          f"{n_trunc   / N_TOTAL * 100:.1f}", "", "", ""])
    writer.writerow(["pasos_medio",         f"{pasos_medio:.1f}", "", "", ""])
    writer.writerow(["reward_medio",        f"{reward_medio:.1f}", "", "", ""])
    writer.writerow([])
    writer.writerow(["RESUMEN_POR_GOAL", "exito", "col_approach", "col_exit", "col_return",
                     "truncado", "exito_%"])
    for gid, v in stats_por_goal.items():
        writer.writerow([gid, v["exito"], v["col_approach"], v["col_exit"],
                         v["col_return"], v["truncado"],
                         f"{v['exito'] / N_POR_GOAL * 100:.1f}"])

print(f"[CSV] {CSV_PATH}")
env.supervisor.simulationQuit(0)
