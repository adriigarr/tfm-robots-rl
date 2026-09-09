"""
Inferencia dinámica v4 — SUB-WP seed=524 — con logging de replanning por fase.

Igual que infer_din_v4_s524.py pero registra por episodio:
  - cuántos replans se dispararon en cada fase (approach / exit / retorno)
  - si el replan ocurrió en la fase donde terminó el episodio

Salida:
  resultados/infer_din_v4_s524_replanlog.csv   (detalle por episodio)
  resultados/infer_din_v4_s524_replanlog_summary.txt
"""

import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

CONTROLLER_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_PATH   = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
MODEL_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_s524_wp75_stage6_din_v4_final")
RUN_ID     = "din_v4_s524_replanlog"
N_POR_GOAL = 100
OUT_DIR    = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH   = os.path.join(OUT_DIR, "infer_din_v4_s524_replanlog.csv")
TXT_PATH   = os.path.join(OUT_DIR, "infer_din_v4_s524_replanlog_summary.txt")

os.makedirs(OUT_DIR, exist_ok=True)

env   = WebotsEnv(map_path=MAP_PATH, stage=6, ped_obs=True)
env._max_steps = 6000
model = PPO.load(MODEL_PATH, env=env)

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia DINÁMICA v4 + replan log | {RUN_ID}")
print(f" {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} ciclos")
print(f"{'='*60}\n")

FIELDS = ["episodio", "goal_id", "resultado",
          "pasos", "reward",
          "replans_approach", "replans_exit", "replans_return",
          "replans_total", "fase_final"]

rows = []
ep_global = 0

# acumuladores para el resumen
acc = {
    "exito":        {"replans_ap":0,"replans_ex":0,"replans_ret":0,"n":0},
    "col_approach": {"replans_ap":0,"replans_ex":0,"replans_ret":0,"n":0},
    "col_exit":     {"replans_ap":0,"replans_ex":0,"replans_ret":0,"n":0},
    "col_return":   {"replans_ap":0,"replans_ex":0,"replans_ret":0,"n":0},
    "truncado":     {"replans_ap":0,"replans_ex":0,"replans_ret":0,"n":0},
}
replans_by_phase = {"approach": 0, "exit": 0, "return": 0}

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

        # contadores de replan por fase para este episodio
        rep_ap = rep_ex = rep_ret = 0
        prev_replans = 0

        while not done and not truncated:
            # fase actual antes del step
            if env._en_retorno:
                fase = "return"
            elif env._hacia_descarga:
                fase = "exit"
            else:
                fase = "approach"

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            pasos += 1
            total_reward += reward

            # detectar si ocurrió un replan en este step
            if env._n_replans > prev_replans:
                nuevos = env._n_replans - prev_replans
                if fase == "approach": rep_ap += nuevos
                elif fase == "exit":   rep_ex += nuevos
                else:                  rep_ret += nuevos
                prev_replans = env._n_replans

        # fase donde terminó el episodio
        if env._en_retorno:
            fase_final = "return"
        elif env._hacia_descarga:
            fase_final = "exit"
        else:
            fase_final = "approach"

        resultado = (
            "exito"        if info.get("exito")    else
            "col_approach" if info.get("colision") and not info.get("llego_estanteria") else
            "col_exit"     if info.get("colision") and info.get("llego_estanteria") and not info.get("llego_espera") and not env._en_retorno else
            "col_return"   if info.get("colision") else
            "truncado"
        )

        replans_by_phase["approach"] += rep_ap
        replans_by_phase["exit"]     += rep_ex
        replans_by_phase["return"]   += rep_ret
        acc[resultado]["replans_ap"]  += rep_ap
        acc[resultado]["replans_ex"]  += rep_ex
        acc[resultado]["replans_ret"] += rep_ret
        acc[resultado]["n"]           += 1

        row = {"episodio": ep_global, "goal_id": goal_id, "resultado": resultado,
               "pasos": pasos, "reward": round(total_reward, 2),
               "replans_approach": rep_ap, "replans_exit": rep_ex,
               "replans_return": rep_ret,
               "replans_total": rep_ap + rep_ex + rep_ret,
               "fase_final": fase_final}
        rows.append(row)
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

        print(f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<15}  "
              f"replans ap={rep_ap} ex={rep_ex} ret={rep_ret}  pasos={pasos}")

# ── resumen ────────────────────────────────────────────────────────────────
n_exito   = acc["exito"]["n"]
n_col_ap  = acc["col_approach"]["n"]
n_col_ex  = acc["col_exit"]["n"]
n_col_ret = acc["col_return"]["n"]
n_trunc   = acc["truncado"]["n"]
total_replans = sum(replans_by_phase.values())

lines = []
lines.append(f"\n{'='*65}")
lines.append(f" RESUMEN — {N_TOTAL} ciclos | {RUN_ID}")
lines.append(f"{'='*65}")
lines.append(f"  Éxito:         {n_exito:>4}/{N_TOTAL}  ({n_exito/N_TOTAL*100:.1f}%)")
lines.append(f"  Col. approach: {n_col_ap:>4}/{N_TOTAL}  ({n_col_ap/N_TOTAL*100:.1f}%)")
lines.append(f"  Col. exit:     {n_col_ex:>4}/{N_TOTAL}  ({n_col_ex/N_TOTAL*100:.1f}%)")
lines.append(f"  Col. retorno:  {n_col_ret:>4}/{N_TOTAL}  ({n_col_ret/N_TOTAL*100:.1f}%)")
lines.append(f"  Truncado:      {n_trunc:>4}/{N_TOTAL}  ({n_trunc/N_TOTAL*100:.1f}%)")
lines.append("")
lines.append(f"  Replans totales: {total_replans}  ({total_replans/N_TOTAL:.2f}/ep)")
lines.append(f"    En approach:  {replans_by_phase['approach']}  ({replans_by_phase['approach']/N_TOTAL:.2f}/ep)")
lines.append(f"    En exit:      {replans_by_phase['exit']}  ({replans_by_phase['exit']/N_TOTAL:.2f}/ep)")
lines.append(f"    En retorno:   {replans_by_phase['return']}  ({replans_by_phase['return']/N_TOTAL:.2f}/ep)")
lines.append("")
lines.append("  Replans medios por resultado:")
for res, data in acc.items():
    if data["n"] == 0:
        continue
    total_r = data["replans_ap"] + data["replans_ex"] + data["replans_ret"]
    lines.append(f"    {res:<15} n={data['n']:>4}  "
                 f"ap={data['replans_ap']/data['n']:.2f}  "
                 f"ex={data['replans_ex']/data['n']:.2f}  "
                 f"ret={data['replans_ret']/data['n']:.2f}  "
                 f"total={total_r/data['n']:.2f}")
lines.append(f"{'='*65}")
lines.append(f"[CSV] {CSV_PATH}")

summary = "\n".join(lines)
print(summary)
with open(TXT_PATH, "w") as f:
    f.write(summary)

env.supervisor.simulationQuit(0)
