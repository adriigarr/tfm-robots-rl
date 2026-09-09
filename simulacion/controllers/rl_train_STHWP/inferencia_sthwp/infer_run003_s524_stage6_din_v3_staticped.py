"""
Inferencia diagnóstica — STH-WP s524 v3 — peatones COMPLETAMENTE ESTÁTICOS.

Los peatones NO se mueven en ningún momento del episodio. En cada step se
teleportan a su posición inicial del world file, anulando su movimiento.

Diferencia vs fixedped ablation anterior: aquí los peatones son estáticos
DURANTE el episodio, no solo en el reset. El fixedped anterior solo fijaba
la posición de inicio — los peatones seguían moviéndose una vez iniciado.

Propósito diagnóstico: separar el impacto del movimiento del peatón del
impacto de la reformulación de política por B1. Si el éxito sube cerca del
100% (como en v1 estático), toda la caída es por el movimiento del peatón.
Si sube solo parcialmente, el modelo B1 ha modificado la política base.

Comparar con:
  - Estático v1:           ~100% (sin peatones en entrenamiento)
  - v3 s524 dinámico:       45.4% (peatones en movimiento)
  - Este script:            ???%  (peatones presentes pero congelados)

Salida: inferencia_sthwp/resultados/infer_run003_s524_stage6_din_v3_staticped.csv
"""

import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

CONTROLLER_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_PATH   = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
MODEL_PATH = os.path.join(CONTROLLER_DIR, "pruebas",
                          "checkpoints_run003_s524_stage6_din_v3",
                          "run003_s524_stage6_din_v3_2801472_steps")
RUN_ID     = "run003_s524_din_v3_staticped"
N_POR_GOAL = 100
OUT_DIR    = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH   = os.path.join(OUT_DIR, "infer_run003_s524_stage6_din_v3_staticped.csv")

os.makedirs(OUT_DIR, exist_ok=True)

env   = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 7000
model = PPO.load(MODEL_PATH, env=env)

# Posiciones fijas (world file defaults)
STATIC_POSITIONS = [
    {"x": -2.0,  "y": 0.3,  "z": 1.27},   # PEDESTRIAN_1
    {"x": -9.5,  "y": -5.0, "z": 1.27},   # PEDESTRIAN_2
]

def freeze_pedestrians():
    """Teleporta los peatones a su posición inicial en cada step."""
    for i, node in enumerate(env._ped_nodes):
        p = STATIC_POSITIONS[i]
        node.getField("translation").setSFVec3f([p["x"], p["y"], p["z"]])

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia DIAGNÓSTICA — peatones ESTÁTICOS | {RUN_ID}")
print(f" Modelo: {os.path.basename(MODEL_PATH)} (ckpt pico 2.8M)")
print(f" Peatones: congelados en posición inicial en CADA step")
print(f" {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} ciclos")
print(f"{'='*60}\n")

FIELDS = ["episodio", "goal_id", "resultado",
          "pasos_total", "reward_total",
          "llego_estanteria", "llego_descarga"]

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
        freeze_pedestrians()   # congelar también tras el reset

        done = truncated = False
        pasos = 0
        total_reward = 0.0
        reached_shelf    = False
        reached_descarga = False

        while not done and not truncated:
            freeze_pedestrians()   # congelar en cada step
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            pasos += 1
            total_reward += reward
            if env._hacia_descarga:
                reached_shelf = True
            if env._en_retorno:
                reached_descarga = True

        exito    = bool(info.get("exito",    False))
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

        row = {"episodio": ep_global, "goal_id": goal_id, "resultado": resultado,
               "pasos_total": pasos, "reward_total": round(total_reward, 2),
               "llego_estanteria": int(reached_shelf), "llego_descarga": int(reached_descarga)}
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

        print(f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<15}  pasos={pasos}")

n_exito   = sum(v["exito"]        for v in stats_por_goal.values())
n_col_ap  = sum(v["col_approach"] for v in stats_por_goal.values())
n_col_ex  = sum(v["col_exit"]     for v in stats_por_goal.values())
n_col_ret = sum(v["col_retorno"]  for v in stats_por_goal.values())
n_trunc   = sum(v["truncado"]     for v in stats_por_goal.values())

print(f"\n{'='*60}")
print(f" RESUMEN — {N_TOTAL} ciclos | {RUN_ID}")
print(f"{'='*60}")
print(f"  Éxito:         {n_exito:>4}/{N_TOTAL}  ({n_exito/N_TOTAL*100:.1f}%)")
print(f"  Col. approach: {n_col_ap:>4}/{N_TOTAL}  ({n_col_ap/N_TOTAL*100:.1f}%)")
print(f"  Col. exit:     {n_col_ex:>4}/{N_TOTAL}  ({n_col_ex/N_TOTAL*100:.1f}%)")
print(f"  Col. retorno:  {n_col_ret:>4}/{N_TOTAL}  ({n_col_ret/N_TOTAL*100:.1f}%)")
print(f"  Truncado:      {n_trunc:>4}/{N_TOTAL}  ({n_trunc/N_TOTAL*100:.1f}%)")
print(f"\n  Por goal:")
for gid, v in stats_por_goal.items():
    tasa = v["exito"] / N_POR_GOAL * 100
    flag = "  ⚠" if tasa < 30 else ""
    print(f"    {gid}: {v['exito']}/{N_POR_GOAL} ({tasa:.0f}%)"
          f"  cap={v['col_approach']} cex={v['col_exit']}"
          f"  cret={v['col_retorno']} tr={v['truncado']}{flag}")
print(f"\n  DIAGNÓSTICO:")
print(f"    Estático v1:        ~100%")
print(f"    v3 s524 dinámico:    45.4%")
print(f"    v3 s524 ped estáticos: {n_exito/N_TOTAL*100:.1f}%")
print(f"{'='*60}\n[CSV] {CSV_PATH}")

env.supervisor.simulationQuit(0)
