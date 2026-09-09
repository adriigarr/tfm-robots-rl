"""
Inferencia stage 6 din v4 r2 — SUB-WP seed=123 — Ciclo completo SIN peatones físicos.

World: warehouse_1_subwp_noped.wbt (peatones con controller=<none>, sin bounding object).
Modelo: din_v4_r2_final (48-dim obs, ped_obs=True). Los nodos PEDESTRIAN_1/2 existen pero
no se mueven ni colisionan → mide capacidad del ciclo aislada de bloqueos de peatones.

100 ep × 28 goals = 2800 ciclos.
Salida: inferencia_subwp/resultados/infer_din_v4_r2_noped_s123.csv
"""

import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

CONTROLLER_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_PATH   = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
MODEL_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_s123_wp75_r2_stage6_din_v4_final")
RUN_ID     = "din_v4_r2_noped_s123"
N_POR_GOAL = 100
OUT_DIR    = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH   = os.path.join(OUT_DIR, "infer_din_v4_r2_noped_s123.csv")

os.makedirs(OUT_DIR, exist_ok=True)

env   = WebotsEnv(map_path=MAP_PATH, stage=6, ped_obs=True)
env._max_steps = 6000
model = PPO.load(MODEL_PATH, env=env)

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia dinámica v4 r2 NOPED | {RUN_ID}")
print(f" {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} ciclos")
print(f" World: warehouse_1_subwp_noped.wbt (sin peatones físicos)")
print(f"{'='*60}\n")

FIELDS = ["episodio", "goal_id", "resultado", "pasos", "reward"]
stats  = {gid: {"exito": 0, "col_approach": 0, "col_exit": 0, "col_return": 0, "truncado": 0}
          for gid in env.goal_ids}
ep_global = 0

with open(CSV_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writeheader()

for goal_idx in range(N_GOALS):
    goal_id = env.goal_ids[goal_idx]
    env._sample_goal_approach = lambda gi=goal_idx: gi
    print(f"\n  [{goal_id}] ({goal_idx+1}/{N_GOALS})")

    for ep in range(1, N_POR_GOAL + 1):
        ep_global += 1
        obs, _ = env.reset()
        done = truncated = False
        pasos = 0; reward_ep = 0.0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            pasos += 1; reward_ep += reward

        resultado = (
            "exito"        if info.get("exito") else
            "col_approach" if info.get("colision") and not info.get("llego_estanteria") else
            "col_exit"     if info.get("colision") and info.get("llego_estanteria") and not info.get("llego_espera") and not env._en_retorno else
            "col_return"   if info.get("colision") else
            "truncado"
        )
        stats[goal_id][resultado] += 1

        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(
                {"episodio": ep_global, "goal_id": goal_id, "resultado": resultado,
                 "pasos": pasos, "reward": round(reward_ep, 2)})
        print(f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<15}  pasos={pasos}")

n_ex  = sum(v["exito"]        for v in stats.values())
n_cap = sum(v["col_approach"] for v in stats.values())
n_cex = sum(v["col_exit"]     for v in stats.values())
n_crt = sum(v["col_return"]   for v in stats.values())
n_tr  = sum(v["truncado"]     for v in stats.values())

print(f"\n{'='*60}")
print(f" RESUMEN — {N_TOTAL} ciclos | {RUN_ID}")
print(f"  Éxito:         {n_ex:>4}/{N_TOTAL} ({n_ex/N_TOTAL*100:.1f}%)")
print(f"  Col. approach: {n_cap:>4}/{N_TOTAL} ({n_cap/N_TOTAL*100:.1f}%)")
print(f"  Col. exit:     {n_cex:>4}/{N_TOTAL} ({n_cex/N_TOTAL*100:.1f}%)")
print(f"  Col. retorno:  {n_crt:>4}/{N_TOTAL} ({n_crt/N_TOTAL*100:.1f}%)")
print(f"  Truncado:      {n_tr:>4}/{N_TOTAL} ({n_tr/N_TOTAL*100:.1f}%)")
print(f"\n  Por goal:")
for gid, v in stats.items():
    tasa = v["exito"]/N_POR_GOAL*100
    print(f"    {gid}: {v['exito']}/{N_POR_GOAL} ({tasa:.0f}%){'  ⚠' if tasa < 30 else ''}")
print(f"{'='*60}\n[CSV] {CSV_PATH}")

env.supervisor.simulationQuit(0)
