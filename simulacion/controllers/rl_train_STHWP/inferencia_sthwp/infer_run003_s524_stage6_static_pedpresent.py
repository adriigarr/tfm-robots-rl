"""
Inferencia diagnóstica — STH-WP s524 modelo ESTÁTICO v1 — peatones presentes pero congelados.

Modelo: run003_s524_stage6_final  →  entrenado SIN peatones, ped_obs=False (40 dims).
Mundo:  warehouse_1.wbt  →  PEDESTRIAN_1 y PEDESTRIAN_2 presentes en el escenario.
Los peatones se teleportan a su posición inicial en CADA step (completamente estáticos).
El modelo NO los observa (ped_obs=False) — solo los "ve" por LIDAR si los cruza.

Pregunta: ¿el modelo v1 (que alcanza ~100% en su mundo original sin peatones)
mantiene ese rendimiento cuando los peatones están físicamente presentes
como obstáculos estáticos en el mundo?

Si sí (~100%): el modelo base es robusto; la caída al 45% en v3 se debe a que
  ped_obs=True corrompe las acciones aunque el peatón no se mueva.
Si no (~45%): los peatones estáticos físicamente obstaculizan la ruta A*,
  y el problema es geométrico — el peatón está en el camino planificado.

Comparar con:
  - v1 estático (sin peatones en mundo):     ~100%
  - v3 s524 peatones estáticos (ped_obs=True): 45.4%
  - Este script (ped_obs=False, peatones presentes): ???%

Salida: inferencia_sthwp/resultados/infer_run003_s524_stage6_static_pedpresent.csv
"""

import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

CONTROLLER_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_PATH   = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
MODEL_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "run003_s524_stage6_final")
RUN_ID     = "run003_s524_stage6_static_pedpresent"
N_POR_GOAL = 100
OUT_DIR    = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH   = os.path.join(OUT_DIR, "infer_run003_s524_stage6_static_pedpresent.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ped_obs=False — mismo espacio de observación que en entrenamiento v1 (40 dims)
env   = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=False)
env._max_steps = 7000
model = PPO.load(MODEL_PATH, env=env)

# Peatones presentes en el mundo pero congelados en cada step
STATIC_POSITIONS = [
    {"x": -2.0,  "y": 0.3,  "z": 1.27},   # PEDESTRIAN_1
    {"x": -9.5,  "y": -5.0, "z": 1.27},   # PEDESTRIAN_2
]

# Obtener nodos manualmente (ped_obs=False no los carga en env._ped_nodes)
def get_ped_nodes():
    nodes = []
    for name in ["PEDESTRIAN_1", "PEDESTRIAN_2"]:
        node = env.supervisor.getFromDef(name)
        if node:
            nodes.append(node)
    return nodes

ped_nodes = get_ped_nodes()
print(f"[INFO] {len(ped_nodes)} nodo(s) de peatón encontrado(s) para congelar")

def freeze_pedestrians():
    for i, node in enumerate(ped_nodes):
        p = STATIC_POSITIONS[i]
        node.getField("translation").setSFVec3f([p["x"], p["y"], p["z"]])

N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia DIAGNÓSTICA | {RUN_ID}")
print(f" Modelo: {os.path.basename(MODEL_PATH)} (v1, ped_obs=False, 40 dims)")
print(f" Peatones: presentes en mundo, congelados en cada step")
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
        freeze_pedestrians()

        done = truncated = False
        pasos = 0
        total_reward = 0.0
        reached_shelf    = False
        reached_descarga = False

        while not done and not truncated:
            freeze_pedestrians()
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
print(f"\n  DIAGNÓSTICO CADENA COMPLETA:")
print(f"    v1 sin peatones en mundo:          ~100%")
print(f"    v1 ped_obs=False, peat. estáticos:  {n_exito/N_TOTAL*100:.1f}%   ← ESTE SCRIPT")
print(f"    v3 ped_obs=True,  peat. estáticos:  45.4%")
print(f"    v3 ped_obs=True,  peat. dinámicos:  45.4%")
print(f"{'='*60}\n[CSV] {CSV_PATH}")

env.supervisor.simulationQuit(0)
