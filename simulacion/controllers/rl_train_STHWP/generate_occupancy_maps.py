"""
Regenera los mapas de ocupacion (occ_grid.png, occ_grid_all_values.png,
goals_accessibility.png) a partir del JSON del almacen (warehouse_map01.json),
que es la misma fuente que usa el planificador A* en entrenamiento
(global_planner.build_grid). No requiere Webots.

Uso:
    python3 generate_occupancy_maps.py
"""
import json
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from global_planner import build_grid, mundo_a_celdas, find_nearest_free

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_PATH = os.path.join(HERE, "warehouse_map01.json")
OUT_DIR = HERE

NAV_MARGIN = 1.2   # igual que grid_nav en webots_env._precompute_paths
GOAL_MARGIN = 0.3  # igual que grid_goal en webots_env._precompute_paths


def grid_bounds(map_info):
    origen = map_info["origin"]
    resolution = map_info["resolution"]
    x_min, y_min = origen
    x_max = x_min + map_info["width"] * resolution
    y_max = y_min + map_info["height"] * resolution
    return x_min, x_max, y_min, y_max


def reachable_mask(grid, start_cell):
    rows, cols = grid.shape
    sr, sc = start_cell
    reachable = np.zeros((rows, cols), dtype=bool)
    if not (0 <= sr < rows and 0 <= sc < cols) or grid[sr, sc] == 1:
        return reachable
    q = deque([(sr, sc)])
    reachable[sr, sc] = True
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and grid[nr, nc] == 0:
                reachable[nr, nc] = True
                q.append((nr, nc))
    return reachable


def main():
    with open(MAP_PATH) as f:
        warehouse = json.load(f)

    grid_nav, map_info = build_grid(MAP_PATH, margin=NAV_MARGIN)
    bounds = grid_bounds(map_info)
    x_min, x_max, y_min, y_max = bounds
    origen, resolution = map_info["origin"], map_info["resolution"]

    # ── occ_grid.png ────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 8))
    plt.imshow(grid_nav, origin="lower", extent=(x_min, x_max, y_min, y_max),
               cmap="gray_r", interpolation="nearest")
    plt.title(f"Occupancy Grid — {os.path.basename(MAP_PATH)} (margin={NAV_MARGIN} m)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, "occ_grid.png")
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"Guardado {out1}")

    # ── occ_grid_all_values.png (con goals y zonas superpuestos) ──────────────
    plt.figure(figsize=(10, 9))
    plt.imshow(grid_nav, origin="lower", extent=(x_min, x_max, y_min, y_max),
               cmap="gray_r", interpolation="nearest")
    gx = [s["goal_x"] for s in warehouse["shelves"]]
    gy = [s["goal_y"] for s in warehouse["shelves"]]
    plt.scatter(gx, gy, s=18, c="crimson", label="goals", zorder=3)
    wz = warehouse["zones"]["waiting"]
    dz = warehouse["zones"]["dropoff"]
    plt.scatter([wz["x"]], [wz["y"]], marker="s", s=90, c="orange", label="waiting", zorder=4)
    plt.scatter([dz["x"]], [dz["y"]], marker="*", s=160, c="limegreen", label="dropoff", zorder=4)
    plt.title(f"Occupancy Grid + goals/zonas — {os.path.basename(MAP_PATH)}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "occ_grid_all_values.png")
    plt.savefig(out2, dpi=200)
    plt.close()
    print(f"Guardado {out2}")

    # ── goals_accessibility.png ────────────────────────────────────────────
    start_cell = mundo_a_celdas(wz["x"], wz["y"], origen, resolution)
    reach = reachable_mask(grid_nav, start_cell)

    plt.figure(figsize=(9, 8))
    plt.imshow(grid_nav, origin="lower", extent=(x_min, x_max, y_min, y_max),
               cmap="gray_r", interpolation="nearest")
    plt.scatter([wz["x"]], [wz["y"]], marker="x", s=90, c="blue", label="start (waiting)", zorder=4)

    greens_x, greens_y, reds_x, reds_y = [], [], [], []
    for shelf in warehouse["shelves"]:
        cell = mundo_a_celdas(shelf["goal_x"], shelf["goal_y"], origen, resolution)
        r, c = cell
        in_bounds = 0 <= r < grid_nav.shape[0] and 0 <= c < grid_nav.shape[1]
        if in_bounds and grid_nav[r, c] == 1:
            # el goal cae dentro del margen de inflado (normal, es junto a la
            # estanteria): igual que hace el planificador real, se snapea a la
            # celda libre mas cercana antes de comprobar conectividad
            snapped = find_nearest_free(grid_nav, cell)
            r, c = snapped if snapped is not None else cell
            in_bounds = 0 <= r < grid_nav.shape[0] and 0 <= c < grid_nav.shape[1]
        ok = in_bounds and reach[r, c]
        (greens_x if ok else reds_x).append(shelf["goal_x"])
        (greens_y if ok else reds_y).append(shelf["goal_y"])

    if greens_x:
        plt.scatter(greens_x, greens_y, s=25, c="limegreen", label="reachable", zorder=3)
    if reds_x:
        plt.scatter(reds_x, reds_y, s=25, c="red", label="not reachable", zorder=3)

    plt.title("Goal accessibility on occupancy grid")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, "goals_accessibility.png")
    plt.savefig(out3, dpi=200)
    plt.close()
    print(f"Guardado {out3}")
    print(f"Goals no alcanzables: {len(reds_x)} / {len(warehouse['shelves'])}")


if __name__ == "__main__":
    main()
