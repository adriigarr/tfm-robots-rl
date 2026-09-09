"""
Calcula distancias euclídeas y longitudes de path A* para cada goal:
  - zona_espera → goal  (acercamiento)
  - goal → zona_descarga  (salida)

Ejecutar desde el directorio rl_train_STHWP_continuo/:
    python3 calcular_distancias.py
"""

import json
import math
import sys
import os

# Importar el planificador global del mismo directorio
sys.path.insert(0, os.path.dirname(__file__))
from global_planner import plan_path

MAP_PATH    = "warehouse_map01.json"
OUTPUT_CSV  = "distancias_goals.csv"

# ── cargar mapa y posiciones ──────────────────────────────────────────────────

with open(MAP_PATH) as f:
    warehouse = json.load(f)

zona_espera   = (warehouse["zones"]["waiting"]["x"],  warehouse["zones"]["waiting"]["y"])
zona_descarga = (warehouse["zones"]["dropoff"]["x"],  warehouse["zones"]["dropoff"]["y"])
shelves       = warehouse["shelves"]  # lista de 28, ordenada por goal_id

def path_length(waypoints):
    """Longitud total de un path como suma de segmentos euclídeos."""
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for i in range(len(waypoints) - 1):
        dx = waypoints[i+1][0] - waypoints[i][0]
        dy = waypoints[i+1][1] - waypoints[i][1]
        total += math.sqrt(dx*dx + dy*dy)
    return total

def euclidean(a, b):
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

# ── calcular para cada goal ───────────────────────────────────────────────────

rows = []

for shelf in shelves:
    goal_id  = shelf["goal_id"]
    goal_pos = (shelf["goal_x"], shelf["goal_y"])

    # ── espera → goal ─────────────────────────────────────────────────────────
    try:
        path_ap = plan_path(MAP_PATH, zona_espera, goal_pos,
                            nav_margin=1.2, goal_margin=0.3, subsample=False)
        len_ap_astar = path_length(path_ap)
    except Exception as e:
        print(f"[WARN] {goal_id} espera→goal A* error: {e}")
        len_ap_astar = float("nan")

    euc_ap = euclidean(zona_espera, goal_pos)

    # ── goal → descarga ───────────────────────────────────────────────────────
    try:
        path_ex = plan_path(MAP_PATH, goal_pos, zona_descarga,
                            nav_margin=1.2, goal_margin=0.3, subsample=False)
        len_ex_astar = path_length(path_ex)
    except Exception as e:
        print(f"[WARN] {goal_id} goal→descarga A* error: {e}")
        len_ex_astar = float("nan")

    euc_ex = euclidean(goal_pos, zona_descarga)

    rows.append({
        "goal_id":       goal_id,
        "goal_x":        goal_pos[0],
        "goal_y":        goal_pos[1],
        "euc_espera_goal":    round(euc_ap, 2),
        "astar_espera_goal":  round(len_ap_astar, 2),
        "euc_goal_descarga":  round(euc_ex, 2),
        "astar_goal_descarga":round(len_ex_astar, 2),
        "euc_total":     round(euc_ap + euc_ex, 2),
        "astar_total":   round(len_ap_astar + len_ex_astar, 2),
    })

    print(f"{goal_id:8s}  approach: euc={euc_ap:5.2f}m  A*={len_ap_astar:6.2f}m  |  "
          f"exit: euc={euc_ex:5.2f}m  A*={len_ex_astar:6.2f}m")

# ── escribir CSV ──────────────────────────────────────────────────────────────

import csv
fieldnames = list(rows[0].keys())
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nResultados guardados en {OUTPUT_CSV}")
