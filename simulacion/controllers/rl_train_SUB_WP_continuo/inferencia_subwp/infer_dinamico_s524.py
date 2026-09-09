"""
Inferencia con obstáculos dinámicos — SUB-WP seed=123.

50 ep/goal → 100 ep/goal × 28 goals = 2800 ciclos.
World file warehouse_1_subwp.wbt con PEDESTRIAN_1 y PEDESTRIAN_2 activos.

Modelo: subwp_s524_wp75_stage5_final (approach + exit + return, stage=6)

Salida: inferencia_subwp/resultados/infer_dinamico_s524.csv
"""

import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

CONTROLLER_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_PATH   = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
MODEL_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_s524_wp75_stage5_final")
RUN_ID     = "dinamico_s524_subwp"
N_POR_GOAL = 100
OUT_DIR    = os.path.join(CONTROLLER_DIR, "inferencia_subwp", "resultados")
CSV_PATH   = os.path.join(OUT_DIR, "infer_dinamico_s524.csv")

os.makedirs(OUT_DIR, exist_ok=True)

env   = WebotsEnv(map_path=MAP_PATH, stage=6)
model = PPO.load(MODEL_PATH, env=env)
env._max_steps = 6000

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia DINÁMICA (peatones) — SUB-WP | {RUN_ID}")
print(f" Modelo: subwp_s524_wp75_stage5_final  |  Stage=6  |  det=True")
print(f" {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} episodios")
print(f" Obstáculos: PEDESTRIAN_1 + PEDESTRIAN_2 activos")
print(f" CSV: {CSV_PATH}")
print(f"{'='*60}\n")

FIELDS = ["episodio", "goal_id", "resultado", "pasos", "reward"]
stats_por_goal = {
    gid: {"exito": 0, "col_approach": 0, "col_exit": 0, "col_return": 0, "truncado": 0}
    for gid in env.goal_ids
}
ep_global = 0

# Escribir cabecera del CSV al inicio — datos se guardan fila a fila
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

        row = {"episodio": ep_global, "goal_id": goal_id, "resultado": resultado,
               "pasos": pasos, "reward": round(reward_ep, 2)}

        # Guardar fila inmediatamente — no acumulamos en memoria
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

        print(f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<20}  pasos={pasos:<5}  reward={reward_ep:.1f}")

n_exito   = sum(v["exito"]        for v in stats_por_goal.values())
n_col_ap  = sum(v["col_approach"] for v in stats_por_goal.values())
n_col_ex  = sum(v["col_exit"]     for v in stats_por_goal.values())
n_col_ret = sum(v["col_return"]   for v in stats_por_goal.values())
n_trunc   = sum(v["truncado"]     for v in stats_por_goal.values())
n_col_tot = n_col_ap + n_col_ex + n_col_ret
pasos_medio  = 0.0  # calculado tras leer el CSV si hace falta
reward_medio = 0.0

print(f"\n{'='*60}")
print(f" RESUMEN — {ep_global} ep | {RUN_ID} | PEATONES ACTIVOS")
print(f"{'='*60}")
print(f"  Éxito:              {n_exito:>4}/{N_TOTAL}  ({n_exito/N_TOTAL*100:.1f}%)")
print(f"  Colisión approach:  {n_col_ap:>4}/{N_TOTAL}  ({n_col_ap/N_TOTAL*100:.1f}%)")
print(f"  Colisión exit:      {n_col_ex:>4}/{N_TOTAL}  ({n_col_ex/N_TOTAL*100:.1f}%)")
print(f"  Colisión return:    {n_col_ret:>4}/{N_TOTAL}  ({n_col_ret/N_TOTAL*100:.1f}%)")
print(f"  Colisión total:     {n_col_tot:>4}/{N_TOTAL}  ({n_col_tot/N_TOTAL*100:.1f}%)")
print(f"  Truncado:           {n_trunc:>4}/{N_TOTAL}  ({n_trunc/N_TOTAL*100:.1f}%)")
print(f"\n  Por goal:")
for gid, v in stats_por_goal.items():
    tasa = v["exito"] / N_POR_GOAL * 100
    flag = "  ⚠" if tasa < 70 else ""
    print(f"    {gid}: {v['exito']}/{N_POR_GOAL} ({tasa:.0f}%)"
          f"  col_ap={v['col_approach']}  col_ex={v['col_exit']}"
          f"  col_ret={v['col_return']}  trunc={v['truncado']}{flag}")
print(f"{'='*60}\n")

# Añadir resumen al CSV
with open(CSV_PATH, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([])
    w.writerow(["RESUMEN_GLOBAL", "", "", "", ""])
    w.writerow(["metodo",              "SUB-WP",                                "", "", ""])
    w.writerow(["seed",               "524",                                    "", "", ""])
    w.writerow(["obstaculos",         "peatones_activos",                       "", "", ""])
    w.writerow(["n_total",            ep_global,                                "", "", ""])
    w.writerow(["exito_%",            f"{n_exito   / N_TOTAL * 100:.1f}",       "", "", ""])
    w.writerow(["colision_approach_%", f"{n_col_ap  / N_TOTAL * 100:.1f}",       "", "", ""])
    w.writerow(["colision_exit_%",     f"{n_col_ex  / N_TOTAL * 100:.1f}",       "", "", ""])
    w.writerow(["colision_return_%",   f"{n_col_ret / N_TOTAL * 100:.1f}",       "", "", ""])
    w.writerow(["colision_total_%",    f"{n_col_tot / N_TOTAL * 100:.1f}",       "", "", ""])
    w.writerow(["truncado_%",          f"{n_trunc   / N_TOTAL * 100:.1f}",       "", "", ""])
    w.writerow([])
    w.writerow(["RESUMEN_POR_GOAL", "exito", "col_approach", "col_exit",
                "col_return", "truncado", "exito_%"])
    for gid, v in stats_por_goal.items():
        w.writerow([gid, v["exito"], v["col_approach"], v["col_exit"],
                    v["col_return"], v["truncado"],
                    f"{v['exito'] / N_POR_GOAL * 100:.1f}"])

print(f"[CSV] {CSV_PATH}")
env.supervisor.simulationQuit(0)
