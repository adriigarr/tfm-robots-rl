"""
Inferencia v1 estático + REPLANIFICACIÓN — SUB-WP s524, peatones congelados.

Modelo: subwp_s524_wp75_stage5_final (v1, ped_obs=False, 40 dims).
Salida: inferencia_subwp/resultados/infer_subwp_s524_stage5_replanning.csv
"""

import os, sys, csv, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO
from global_planner import (build_grid, astar, mundo_a_celdas, celdas_a_mundo,
                             subsample_path, find_nearest_free)

CONTROLLER_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_PATH   = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
MODEL_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_s524_wp75_stage5_final")
RUN_ID   = "subwp_s524_static_replanning"
N_POR_GOAL = 100
OUT_DIR  = os.path.join(os.path.dirname(__file__), "resultados")
CSV_PATH = os.path.join(OUT_DIR, "infer_subwp_s524_stage5_replanning.csv")

DIST_PERP_THRESHOLD = 0.6
INFLATE_CELLS       = 3
REPLAN_COOLDOWN     = 40

STATIC_POSITIONS = [
    {"x": -2.0,  "y": 0.3,  "z": 1.27},
    {"x": -9.5,  "y": -5.0, "z": 1.27},
]

os.makedirs(OUT_DIR, exist_ok=True)

env   = WebotsEnv(map_path=MAP_PATH, stage=6, ped_obs=False)
env._max_steps = 6000
model = PPO.load(MODEL_PATH, env=env)

grid_nav, map_info = build_grid(MAP_PATH, margin=1.2)
ORIGEN     = map_info["origin"]
RESOLUTION = map_info["resolution"]
ROWS, COLS = grid_nav.shape

ped_nodes = [n for name in ["PEDESTRIAN_1", "PEDESTRIAN_2"]
             if (n := env.supervisor.getFromDef(name)) is not None]
PED_POSITIONS_STATIC = [(p["x"], p["y"]) for p in STATIC_POSITIONS]


def freeze_pedestrians():
    for i, node in enumerate(ped_nodes):
        p = STATIC_POSITIONS[i]
        node.getField("translation").setSFVec3f([p["x"], p["y"], p["z"]])


def dist_punto_segmento(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-9:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def pedestrian_bloquea(rx, ry, sx, sy):
    return any(dist_punto_segmento(px, py, rx, ry, sx, sy) < DIST_PERP_THRESHOLD
               for px, py in PED_POSITIONS_STATIC)


def replanificar(env):
    pos      = env.robot_node.getField("translation").getSFVec3f()
    rx, ry   = pos[0], pos[1]
    goal_pos = env.full_path[-1]
    grid_tmp = grid_nav.copy()
    for px, py in PED_POSITIONS_STATIC:
        pr, pc = mundo_a_celdas(px, py, ORIGEN, RESOLUTION)
        for dr in range(-INFLATE_CELLS, INFLATE_CELLS + 1):
            for dc in range(-INFLATE_CELLS, INFLATE_CELLS + 1):
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    grid_tmp[nr, nc] = 1
    start_cell = mundo_a_celdas(rx, ry, ORIGEN, RESOLUTION)
    goal_cell  = mundo_a_celdas(goal_pos[0], goal_pos[1], ORIGEN, RESOLUTION)
    if grid_tmp[start_cell[0]][start_cell[1]] == 1:
        start_cell = find_nearest_free(grid_tmp, start_cell)
    if grid_tmp[goal_cell[0]][goal_cell[1]] == 1:
        goal_cell = find_nearest_free(grid_tmp, goal_cell)
    if start_cell is None or goal_cell is None:
        return False
    path_cells = astar(grid_tmp, start_cell, goal_cell)
    if path_cells is None:
        return False
    path_world = [celdas_a_mundo(r, c, ORIGEN, RESOLUTION) for r, c in path_cells]
    path_world = subsample_path(path_world, step=2.0)
    if not path_world:
        return False
    path_world[-1] = goal_pos
    env.full_path = path_world
    env._wp_idx   = 0
    return True


N_GOALS = len(env.goal_ids)
N_TOTAL = N_GOALS * N_POR_GOAL

print(f"\n{'='*60}")
print(f" Inferencia v1 ESTÁTICO + REPLANNING SUB-WP | {RUN_ID}")
print(f" Modelo: {os.path.basename(MODEL_PATH)} (v1, ped_obs=False)")
print(f" Peatones congelados. Replanning: perp < {DIST_PERP_THRESHOLD}m")
print(f" {N_GOALS} goals × {N_POR_GOAL} ep = {N_TOTAL} ciclos")
print(f"{'='*60}\n")

FIELDS = ["episodio", "goal_id", "resultado", "pasos_total", "reward_total",
          "llego_estanteria", "llego_descarga", "n_replans"]

stats_por_goal = {gid: {"exito": 0, "col_approach": 0, "col_exit": 0,
                         "col_return": 0, "truncado": 0}
                  for gid in env.goal_ids}
ep_global     = 0
total_replans = 0

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
        pasos = 0; total_reward = 0.0
        reached_shelf = reached_descarga = False
        replan_cooldown = replans_ep = 0

        while not done and not truncated:
            freeze_pedestrians()
            wp_idx = min(env._wp_idx, len(env.full_path) - 1)
            sx, sy = env.full_path[wp_idx]
            pos = env.robot_node.getField("translation").getSFVec3f()
            rx, ry = pos[0], pos[1]
            if replan_cooldown == 0 and pedestrian_bloquea(rx, ry, sx, sy):
                if replanificar(env):
                    replan_cooldown = REPLAN_COOLDOWN
                    replans_ep += 1
            if replan_cooldown > 0:
                replan_cooldown -= 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            pasos += 1; total_reward += reward
            if env._hacia_descarga: reached_shelf = True
            if env._en_retorno:     reached_descarga = True

        total_replans += replans_ep
        resultado = (
            "exito"        if info.get("exito") else
            "col_approach" if info.get("colision") and not reached_shelf else
            "col_exit"     if info.get("colision") and not reached_descarga else
            "col_return"   if info.get("colision") else
            "truncado"
        )
        stats_por_goal[goal_id][resultado] += 1

        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(
                {"episodio": ep_global, "goal_id": goal_id, "resultado": resultado,
                  "pasos_total": pasos, "reward_total": round(total_reward, 2),
                  "llego_estanteria": int(reached_shelf), "llego_descarga": int(reached_descarga),
                  "n_replans": replans_ep})
        print(f"    ep {ep:>3}/{N_POR_GOAL}  {resultado:<15}  pasos={pasos}  replans={replans_ep}")

n_exito  = sum(v["exito"]        for v in stats_por_goal.values())
n_col_ap = sum(v["col_approach"] for v in stats_por_goal.values())
n_col_ex = sum(v["col_exit"]     for v in stats_por_goal.values())
n_col_ret= sum(v["col_return"]   for v in stats_por_goal.values())
n_trunc  = sum(v["truncado"]     for v in stats_por_goal.values())

print(f"\n{'='*60}")
print(f" RESUMEN — {N_TOTAL} ciclos | {RUN_ID}")
print(f"{'='*60}")
print(f"  Éxito:         {n_exito:>4}/{N_TOTAL}  ({n_exito/N_TOTAL*100:.1f}%)")
print(f"  Col. approach: {n_col_ap:>4}/{N_TOTAL}  ({n_col_ap/N_TOTAL*100:.1f}%)")
print(f"  Col. exit:     {n_col_ex:>4}/{N_TOTAL}  ({n_col_ex/N_TOTAL*100:.1f}%)")
print(f"  Col. retorno:  {n_col_ret:>4}/{N_TOTAL}  ({n_col_ret/N_TOTAL*100:.1f}%)")
print(f"  Truncado:      {n_trunc:>4}/{N_TOTAL}  ({n_trunc/N_TOTAL*100:.1f}%)")
print(f"  Replans total: {total_replans}")
print(f"\n  Por goal:")
for gid, v in stats_por_goal.items():
    tasa = v["exito"] / N_POR_GOAL * 100
    flag = "  ⚠" if tasa < 30 else ""
    print(f"    {gid}: {v['exito']}/{N_POR_GOAL} ({tasa:.0f}%)"
          f"  cap={v['col_approach']} cex={v['col_exit']}"
          f"  cret={v['col_return']} tr={v['truncado']}{flag}")
print(f"\n  CADENA DE DIAGNÓSTICO:")
print(f"    v1 peatones estáticos sin replan: 15.5%")
print(f"    v1 peatones estáticos + replan:  {n_exito/N_TOTAL*100:.1f}%   ← ESTE SCRIPT")
print(f"    v3 peatones dinámicos sin replan: 39.8%")
print(f"{'='*60}\n[CSV] {CSV_PATH}")

env.supervisor.simulationQuit(0)
