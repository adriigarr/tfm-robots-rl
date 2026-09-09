import json
import math
import os
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def build_grid(map_path, shelf_size=(1.9, 0.8), margin=0.3):
    with open(map_path, "r") as f:
        warehouse = json.load(f)

    info = warehouse["map"]
    rows = info["height"]
    cols = info["width"]
    origen = info["origin"]
    resolution = info["resolution"]

    grid = np.zeros((rows, cols), dtype=np.int8)

    for shelf in warehouse["shelves"]:
        cx, cy = shelf["shelf_x"], shelf["shelf_y"]
        x_min = cx - shelf_size[0] / 2 - margin
        x_max = cx + shelf_size[0] / 2 + margin
        y_min = cy - shelf_size[1] / 2 - margin
        y_max = cy + shelf_size[1] / 2 + margin
        row_min, col_min = mundo_a_celdas(x_min, y_min, origen, resolution)
        row_max, col_max = mundo_a_celdas(x_max, y_max, origen, resolution)
        row_min = max(0, row_min)
        row_max = min(rows - 1, row_max)
        col_min = max(0, col_min)
        col_max = min(cols - 1, col_max)
        grid[row_min:row_max + 1, col_min:col_max + 1] = 1

    for wall in warehouse.get("walls", []):
        wx, wy = wall["x"], wall["y"]
        ww, wh = wall["w"], wall["h"]
        x_min = wx - ww / 2 - margin
        x_max = wx + ww / 2 + margin
        y_min = wy - wh / 2 - margin
        y_max = wy + wh / 2 + margin
        row_min, col_min = mundo_a_celdas(x_min, y_min, origen, resolution)
        row_max, col_max = mundo_a_celdas(x_max, y_max, origen, resolution)
        row_min = max(0, row_min)
        row_max = min(rows - 1, row_max)
        col_min = max(0, col_min)
        col_max = min(cols - 1, col_max)
        grid[row_min:row_max + 1, col_min:col_max + 1] = 1

    map_info = {
        "origin": origen,
        "resolution": resolution,
        "width": cols,
        "height": rows,
    }
    return grid, map_info


def astar(grid, start, goal):
    rows, cols = grid.shape

    def h(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            vecino = (current[0] + dr, current[1] + dc)
            r, c = vecino
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if grid[r][c] == 1:
                continue
            coste = 1.414 if dr != 0 and dc != 0 else 1.0
            g_nuevo = g_score[current] + coste
            if g_nuevo < g_score.get(vecino, float("inf")):
                came_from[vecino] = current
                g_score[vecino] = g_nuevo
                f = g_nuevo + h(vecino, goal)
                heapq.heappush(open_set, (f, vecino))

    return None


def mundo_a_celdas(x, y, origen, resolution):
    col = int((x - origen[0]) / resolution)
    row = int((y - origen[1]) / resolution)
    return row, col


def celdas_a_mundo(row, col, origen, resolution):
    x = origen[0] + col * resolution
    y = origen[1] + row * resolution
    return x, y


def find_nearest_free(grid, cell):
    rows, cols = grid.shape
    r, c = cell
    if grid[r][c] == 0:
        return cell
    for radio in range(1, 20):
        for dr in range(-radio, radio + 1):
            for dc in range(-radio, radio + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr][nc] == 0:
                        return (nr, nc)
    return None


def plan_path(map_path, start_pos, goal_pos,
              nav_margin=1.2, goal_margin=0.3, subsample=True,
              verbose=True,
              grid_nav=None, grid_goal=None, map_info=None):
    """
    Planifica una ruta A* desde start_pos hasta goal_pos.

    Si se pasan grid_nav, grid_goal y map_info pre-construidos, se omite
    el coste de build_grid (útil para precomputar muchas rutas en batch).
    """
    if grid_nav is None or grid_goal is None or map_info is None:
        grid_nav, map_info = build_grid(map_path, margin=nav_margin)
        grid_goal, _ = build_grid(map_path, margin=goal_margin)

    origen = map_info["origin"]
    resolution = map_info["resolution"]

    start_cell = mundo_a_celdas(start_pos[0], start_pos[1], origen, resolution)
    goal_cell = mundo_a_celdas(goal_pos[0], goal_pos[1], origen, resolution)

    rows, cols = grid_nav.shape

    if verbose:
        print(f"Grid: {rows}x{cols}")
        print(f"Start cell: {start_cell} → grid_nav: {grid_nav[start_cell[0]][start_cell[1]]}")
        print(f"Goal cell:  {goal_cell}  → grid_goal: {grid_goal[goal_cell[0]][goal_cell[1]]}")

    if grid_nav[start_cell[0]][start_cell[1]] == 1:
        start_cell = find_nearest_free(grid_nav, start_cell)
        if verbose:
            print(f"Start ajustado a: {start_cell}")

    if grid_goal[goal_cell[0]][goal_cell[1]] == 1:
        goal_cell = find_nearest_free(grid_goal, goal_cell)
        if verbose:
            print(f"Goal ajustado a: {goal_cell}")

    goal_cell_nav = goal_cell
    if grid_nav[goal_cell[0]][goal_cell[1]] == 1:
        goal_cell_nav = find_nearest_free(grid_nav, goal_cell)
        if verbose:
            print(f"Goal ajustado en grid_nav a: {goal_cell_nav}")

    path_cells = astar(grid_nav, start_cell, goal_cell_nav)
    if path_cells is None:
        return None

    path_world = [celdas_a_mundo(r, c, origen, resolution) for r, c in path_cells]

    if subsample:
        path_world = subsample_path(path_world, step=2.0)
        if verbose:
            print(f"[A*] Ruta subsampled: {len(path_world)} waypoints")
    else:
        if verbose:
            print(f"[A*] Ruta completa: {len(path_world)} puntos")

    return path_world


def subsample_path(path_world, step=2.0):
    if not path_world or len(path_world) < 2:
        return path_world
    result = [path_world[0]]
    dist_acum = 0.0
    for i in range(1, len(path_world)):
        px, py = path_world[i - 1]
        cx, cy = path_world[i]
        dist_acum += math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        if dist_acum >= step:
            result.append(path_world[i])
            dist_acum = 0.0
    if result[-1] != path_world[-1]:
        result.append(path_world[-1])
    return result


def compute_sth_subgoal(full_path, robot_pos, d_ahead=2.0):
    """
    STH-WP: devuelve el punto más lejano de full_path dentro del radio d_ahead
    desde robot_pos. Si ningún punto cae dentro del radio, devuelve el más cercano.
    """
    rx, ry = robot_pos
    subgoal = None
    for wp in full_path:
        dx = wp[0] - rx
        dy = wp[1] - ry
        if math.sqrt(dx * dx + dy * dy) <= d_ahead:
            subgoal = wp
    if subgoal is None:
        subgoal = min(full_path, key=lambda wp: (wp[0] - rx) ** 2 + (wp[1] - ry) ** 2)
    return subgoal


# ─────────────────────────────────────────────────────────────────────────────
# Visualización
# ─────────────────────────────────────────────────────────────────────────────

def visualizar_paths_goal(map_path, grid_nav, map_info, shelf_idx,
                           path_approach, path_exit, out_dir="path_plots"):
    """
    Genera una imagen para un goal concreto mostrando:
      - Grid del almacén (obstáculos en gris)
      - Path de approach (zona_espera → estantería) en azul
      - Path de exit (estantería → zona_descarga) en naranja
      - Cada waypoint individual marcado con un punto
      - Puntos clave (espera, estantería, descarga) marcados con estrellas

    Guarda la imagen en out_dir/path_goal_XX.png.
    """
    with open(map_path, "r") as f:
        warehouse = json.load(f)

    origen     = map_info["origin"]
    resolution = map_info["resolution"]
    shelf      = warehouse["shelves"][shelf_idx]
    goal_id    = shelf["goal_id"]

    def w2c(x, y):
        col = int((x - origen[0]) / resolution)
        row = int((y - origen[1]) / resolution)
        return row, col

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(grid_nav, cmap="gray_r", origin="lower", alpha=0.6)

    # ── approach path ─────────────────────────────────────────────────────────
    if path_approach and len(path_approach) >= 2:
        rows_a = [w2c(x, y)[0] for x, y in path_approach]
        cols_a = [w2c(x, y)[1] for x, y in path_approach]
        ax.plot(cols_a, rows_a, "-", color="steelblue", linewidth=1.5,
                label=f"approach ({len(path_approach)} wps)", zorder=2)
        ax.plot(cols_a, rows_a, "o", color="steelblue", markersize=4,
                zorder=3)

    # ── exit path ─────────────────────────────────────────────────────────────
    if path_exit and len(path_exit) >= 2:
        rows_e = [w2c(x, y)[0] for x, y in path_exit]
        cols_e = [w2c(x, y)[1] for x, y in path_exit]
        ax.plot(cols_e, rows_e, "-", color="darkorange", linewidth=1.5,
                label=f"exit ({len(path_exit)} wps)", zorder=2)
        ax.plot(cols_e, rows_e, "o", color="darkorange", markersize=4,
                zorder=3)

    # ── puntos clave ──────────────────────────────────────────────────────────
    zona_espera   = warehouse["zones"]["waiting"]
    zona_descarga = warehouse["zones"]["dropoff"]

    r, c = w2c(zona_espera["x"], zona_espera["y"])
    ax.plot(c, r, "*", color="green", markersize=14, zorder=4, label="zona espera")

    r, c = w2c(shelf["goal_x"], shelf["goal_y"])
    ax.plot(c, r, "*", color="steelblue", markersize=14, zorder=4, label="estantería")

    r, c = w2c(zona_descarga["x"], zona_descarga["y"])
    ax.plot(c, r, "*", color="red", markersize=14, zorder=4, label="zona descarga")

    ax.set_title(f"Paths — {goal_id}  |  approach: {len(path_approach or [])} pts  |  exit: {len(path_exit or [])} pts")
    ax.legend(loc="upper right", fontsize=9)
    ax.axis("off")

    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"path_{goal_id}.png")
    plt.savefig(filename, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Main: genera imágenes de todos los paths del almacén
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MAP_PATH = "warehouse_map01.json"
    OUT_DIR  = "path_plots"

    with open(MAP_PATH, "r") as f:
        warehouse = json.load(f)

    zona_espera   = (warehouse["zones"]["waiting"]["x"],
                     warehouse["zones"]["waiting"]["y"])
    zona_descarga = (warehouse["zones"]["dropoff"]["x"],
                     warehouse["zones"]["dropoff"]["y"])

    # construir grids una sola vez
    grid_nav,  map_info = build_grid(MAP_PATH, margin=1.2)
    grid_goal, _        = build_grid(MAP_PATH, margin=0.3)

    print(f"Grid: {grid_nav.shape[0]}×{grid_nav.shape[1]}  |  "
          f"{int(grid_nav.sum())} celdas bloqueadas de {grid_nav.size}")
    print(f"Generando imágenes para {len(warehouse['shelves'])} goals → {OUT_DIR}/\n")

    for idx, shelf in enumerate(warehouse["shelves"]):
        goal_id  = shelf["goal_id"]
        goal_pos = (shelf["goal_x"], shelf["goal_y"])

        path_approach = plan_path(
            MAP_PATH, zona_espera, goal_pos,
            subsample=False, verbose=False,
            grid_nav=grid_nav, grid_goal=grid_goal, map_info=map_info,
        )
        path_exit = plan_path(
            MAP_PATH, goal_pos, zona_descarga,
            subsample=False, verbose=False,
            grid_nav=grid_nav, grid_goal=grid_goal, map_info=map_info,
        )

        n_app  = len(path_approach) if path_approach else 0
        n_exit = len(path_exit)     if path_exit     else 0
        print(f"  {goal_id}: approach={n_app} pts, exit={n_exit} pts", end="  ")

        visualizar_paths_goal(
            MAP_PATH, grid_nav, map_info, idx,
            path_approach, path_exit, out_dir=OUT_DIR,
        )

    print(f"\nListo. {len(warehouse['shelves'])} imágenes en ./{OUT_DIR}/")
