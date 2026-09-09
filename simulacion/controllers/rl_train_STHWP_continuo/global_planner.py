# en este fichero se va a construir el planificador global

import json
import numpy as np
import heapq
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def build_grid (map_path, shelf_size=(1.9, 0.8), margin=0.3):
    """
    construccion de el grid de navegación a partir del JSON del almacén
    
    - map math -> ruta al JSON
    - shelf_size -> tamaño de las estanterias en metros
    - margin -> margen de seguridad alrededor de las estanterias
    
    Devuelve:
    - grid -> array con 0 para celdas libres y 1 para celdas ocupadas
    - map_info -> diccionario con el origen, la resolución, ancho y alto
    """
    # cargar el JSON
    with open(map_path, "r") as f:
        warehouse = json.load(f)

    info = warehouse["map"]
    rows = info["height"]
    cols = info["width"]
    origen = info["origin"]
    resolution = info["resolution"]
    
    # crear el grid vacío
    grid = np.zeros((rows, cols), dtype=np.int8)
    
    # para cada estantería marcar las celdas que ocupa como 1
    # añadiendo el margen de seguridad
    for shelf in warehouse["shelves"]:
        cx, cy = shelf["shelf_x"], shelf["shelf_y"]
    
        # límites del rectángulo con margen
        x_min = cx - shelf_size[0]/2 - margin
        x_max = cx + shelf_size[0]/2 + margin
        y_min = cy - shelf_size[1]/2 - margin
        y_max = cy + shelf_size[1]/2 + margin
        
        # convertir a celdas
        row_min, col_min = mundo_a_celdas(x_min, y_min, origen, resolution)
        row_max, col_max = mundo_a_celdas(x_max, y_max, origen, resolution)
        
        # asegurarse de que no nos salimos del grid
        row_min = max(0, row_min)
        row_max = min(rows-1, row_max)
        col_min = max(0, col_min)
        col_max = min(cols-1, col_max)
        
        # marcar celdas como bloqueadas
        grid[row_min:row_max+1, col_min:col_max+1] = 1
    
    # marcar paredes en el grid
    for wall in warehouse.get("walls", []):
        wx, wy = wall["x"], wall["y"]
        ww, wh = wall["w"], wall["h"]
        
        x_min = wx - ww/2 - margin
        x_max = wx + ww/2 + margin
        y_min = wy - wh/2 - margin
        y_max = wy + wh/2 + margin
        
        row_min, col_min = mundo_a_celdas(x_min, y_min, origen, resolution)
        row_max, col_max = mundo_a_celdas(x_max, y_max, origen, resolution)
        
        row_min = max(0, row_min)
        row_max = min(rows-1, row_max)
        col_min = max(0, col_min)
        col_max = min(cols-1, col_max)
        
        grid[row_min:row_max+1, col_min:col_max+1] = 1
    
    # devolver el grid y la info del mapa
    map_info = {
        "origin": origen,
        "resolution": resolution,
        "width": cols,
        "height": rows
    }
    return grid, map_info


def astar (grid, start, goal):
    """
    Algoritmo A* sobre el grid
    
    - grid -> array con 0 para celdas libres y 1 para celdas ocupadas
    - start -> celda de inicio (row, col)
    - goal -> celda objetivo (row, col)
    
    Devuelve la lista de celdas desde inicio hasta goal,
    o None si no hay camino posible
    """
    rows, cols = grid.shape
    
    # heurística: distancia euclidiana
    def h(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    # open_set: (f, celda)
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        # llegamos al goal → reconstruir camino
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        # explorar vecinos (8 direcciones)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),
                        (-1,-1),(-1,1),(1,-1),(1,1)]:
            vecino = (current[0]+dr, current[1]+dc)
            r, c = vecino
            
            # fuera del grid o bloqueado → saltar
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if grid[r][c] == 1:
                continue
            
            # coste del movimiento (diagonal cuesta más)
            coste = 1.414 if dr != 0 and dc != 0 else 1.0
            g_nuevo = g_score[current] + coste
            
            if g_nuevo < g_score.get(vecino, float('inf')):
                came_from[vecino] = current
                g_score[vecino] = g_nuevo
                f = g_nuevo + h(vecino, goal)
                heapq.heappush(open_set, (f, vecino))
    
    return None  # no hay camino

def mundo_a_celdas (x, y, origen, resolution):
    """"
    Convierte las coordenadas del mundo a celdas del grid
    """""
    col = int((x - origen[0]) / resolution)
    row = int((y - origen[1]) / resolution)
    return row, col
    
def celdas_a_mundo (row, col, origen, resolution):
    """
    convierte las celdas del grid en coordenadas del mundo
    """
    x = origen[0] + col * resolution
    y = origen[1] + row * resolution
    return x, y

def find_nearest_free(grid, cell):
    """
    Dado una celda bloqueada, devuelve la celda libre más cercana.
    Busca en espiral creciente alrededor de la celda.
    
    - grid -> array con 0 para celdas libres y 1 para celdas ocupadas
    - cell -> (row, col) celda de partida
    
    Devuelve la celda libre más cercana o None si no encuentra ninguna
    """
    rows, cols = grid.shape
    r, c = cell
    
    # si ya es libre, devolverla directamente
    if grid[r][c] == 0:
        return cell
    
    # buscar en espiral creciente
    for radio in range(1, 20):
        for dr in range(-radio, radio+1):
            for dc in range(-radio, radio+1):
                nr, nc = r+dr, c+dc
                # comprobar que no nos salimos del grid
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr][nc] == 0:
                        return (nr, nc)
    
    return None  # no encontró celda libre

def plan_path(map_path, start_pos, goal_pos, nav_margin=1.2, goal_margin=0.3, subsample=True):
    """
    Función principal:
    - dado un inicio y goal en coordenadas
    - construye la ruta como lista de coordenadas del mundo
    - usa dos grids: uno con margen grande para navegar seguro
      y otro con margen pequeño para acceder al goal
    """
    # 1. construir dos grids: navegación segura y acceso al goal
    grid_nav,  map_info = build_grid(map_path, margin=nav_margin)
    grid_goal, _        = build_grid(map_path, margin=goal_margin)

    origen     = map_info["origin"]
    resolution = map_info["resolution"]

    # 2. convertir posiciones del mundo a celdas
    start_cell = mundo_a_celdas(start_pos[0], start_pos[1], origen, resolution)
    goal_cell  = mundo_a_celdas(goal_pos[0],  goal_pos[1],  origen, resolution)

    rows, cols = grid_nav.shape
    print(f"Grid: {rows}x{cols}")
    print(f"Start cell: {start_cell} → valor en grid_nav: {grid_nav[start_cell[0]][start_cell[1]]}")
    print(f"Goal cell:  {goal_cell}  → valor en grid_goal: {grid_goal[goal_cell[0]][goal_cell[1]]}")
    print(f"Celdas bloqueadas (nav): {grid_nav.sum()} de {rows*cols}")

    # ajustar start si está bloqueado en grid de navegación
    if grid_nav[start_cell[0]][start_cell[1]] == 1:
        start_cell = find_nearest_free(grid_nav, start_cell)
        print(f"Start ajustado a: {start_cell}")

    # ajustar goal si está bloqueado en grid de acceso
    if grid_goal[goal_cell[0]][goal_cell[1]] == 1:
        goal_cell = find_nearest_free(grid_goal, goal_cell)
        print(f"Goal ajustado a: {goal_cell}")

    # 3. ajustar goal en grid_nav para que A* pueda llegar
    goal_cell_nav = goal_cell
    if grid_nav[goal_cell[0]][goal_cell[1]] == 1:
        goal_cell_nav = find_nearest_free(grid_nav, goal_cell)
        print(f"Goal ajustado en grid_nav a: {goal_cell_nav}")

    # ejecutar A* sobre el grid de navegación segura
    path_cells = astar(grid_nav, start_cell, goal_cell_nav)

    # 4. convertir celdas a coordenadas del mundo
    path_world = [
        celdas_a_mundo(r, c, origen, resolution)
        for r, c in path_cells
    ]
    if subsample:
        path_world = subsample_path(path_world, step=2.0)
        print(f"[A*] Ruta subsampled: {len(path_world)} waypoints (margen nav={nav_margin}m)")
    else:
        print(f"[A*] Ruta completa: {len(path_world)} puntos (STH-WP, margen nav={nav_margin}m)")
    return path_world

def subsample_path(path_world, step=2.0):
    """
    Reduce el número de waypoints manteniendo uno cada `step` metros.
    Siempre incluye el primer y último punto.
    """
    if not path_world or len(path_world) < 2:
        return path_world
    
    result = [path_world[0]]
    dist_acum = 0.0
    
    for i in range(1, len(path_world)):
        px, py = path_world[i-1]
        cx, cy = path_world[i]
        dist_acum += math.sqrt((cx-px)**2 + (cy-py)**2)
        
        if dist_acum >= step:
            result.append(path_world[i])
            dist_acum = 0.0
    
    # siempre incluir el último punto
    if result[-1] != path_world[-1]:
        result.append(path_world[-1])
    
    return result

def compute_sth_subgoal(full_path, robot_pos, d_ahead=2.0):
    """
    STH-WP: desde la posición actual del robot, proyecta un círculo
    de radio d_ahead sobre la ruta completa de A* y devuelve el punto
    más lejano dentro de ese radio (el subgoal dinámico).
    """
    rx, ry = robot_pos
    subgoal = None

    for wp in full_path:
        dx = wp[0] - rx
        dy = wp[1] - ry
        if math.sqrt(dx*dx + dy*dy) <= d_ahead:
            subgoal = wp  # seguimos iterando → queremos el más lejano dentro del radio

    # si ningún punto cae dentro del radio (robot muy desviado)
    # devolver el punto más cercano de la ruta
    if subgoal is None:
        subgoal = min(full_path, key=lambda wp: (wp[0]-rx)**2 + (wp[1]-ry)**2)

    return subgoal

def visualizar_grid(grid, map_info, warehouse, filename="grid.png"):
    origen     = map_info["origin"]
    resolution = map_info["resolution"]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(grid, cmap="gray_r", origin="lower")
    
    # zona de espera
    waiting = warehouse["zones"]["waiting"]
    r, c = mundo_a_celdas(waiting["x"], waiting["y"], origen, resolution)
    ax.plot(c, r, "gs", markersize=10, label="zona espera")
    
    # zona de descarga
    dropoff = warehouse["zones"]["dropoff"]
    r, c = mundo_a_celdas(dropoff["x"], dropoff["y"], origen, resolution)
    ax.plot(c, r, "rs", markersize=10, label="zona descarga")
    
    # waypoints del pasillo de retorno
    for i, wp in enumerate(warehouse["return_corridor"]["waypoints"]):
        r, c = mundo_a_celdas(wp["x"], wp["y"], origen, resolution)
        ax.plot(c, r, "b^", markersize=8)
        ax.text(c+1, r+1, wp["id"], fontsize=7, color="blue")
        
    # goals de las estanterías
    for shelf in warehouse["shelves"]:
        r, c = mundo_a_celdas(shelf["goal_x"], shelf["goal_y"], origen, resolution)
        ax.plot(c, r, "y*", markersize=8)
        ax.text(c+1, r+1, shelf["goal_id"], fontsize=6, color="orange")
    
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0],[0], marker="*", color="w", markerfacecolor="yellow", markersize=10))
    labels.append("goals estanterías")
    ax.legend(handles, labels)
    ax.set_title("Grid del almacén")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Grid guardado en {filename}")

def visualizar_camino(grid, path_cells, start_cell, goal_cell, map_info, warehouse, filename="path.png"):
    origen     = map_info["origin"]
    resolution = map_info["resolution"]
    
    plt.figure(figsize=(12, 10))
    plt.imshow(grid, cmap="gray_r", origin="lower")
    
    if path_cells:
        rows = [c[0] for c in path_cells]
        cols = [c[1] for c in path_cells]
        plt.plot(cols, rows, "b-", linewidth=2, label="camino A*")
    
    plt.plot(start_cell[1], start_cell[0], "go", markersize=10, label="inicio")
    plt.plot(goal_cell[1],  goal_cell[0],  "ro", markersize=10, label="goal")
    
    # zona de descarga
    dropoff = warehouse["zones"]["dropoff"]
    r, c = mundo_a_celdas(dropoff["x"], dropoff["y"], origen, resolution)
    plt.plot(c, r, "rs", markersize=10, label="zona descarga")
    
    # waypoints
    for wp in warehouse["return_corridor"]["waypoints"]:
        r, c = mundo_a_celdas(wp["x"], wp["y"], origen, resolution)
        plt.plot(c, r, "b^", markersize=8)
        plt.text(c+1, r+1, wp["id"], fontsize=7, color="blue")
    
    plt.legend()
    plt.title("Camino A* sobre el grid")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Camino guardado en {filename}")
    
def visualizar_ciclo_completo(map_path, shelf_idx=0, filename="ciclo_completo.png", nav_margin=1.2, goal_margin=0.3):
    """
    Visualiza la ruta completa: zona espera → estantería → zona descarga
    """
    with open(map_path, "r") as f:
        warehouse = json.load(f)

    grid_nav, map_info = build_grid(map_path, margin=nav_margin)
    origen     = map_info["origin"]
    resolution = map_info["resolution"]

    zona_espera   = (warehouse["zones"]["waiting"]["x"], warehouse["zones"]["waiting"]["y"])
    zona_descarga = (warehouse["zones"]["dropoff"]["x"],  warehouse["zones"]["dropoff"]["y"])
    shelf         = warehouse["shelves"][shelf_idx]
    goal_estanteria = (shelf["goal_x"], shelf["goal_y"])

    # ruta 1: espera → estantería
    path1 = plan_path(map_path, zona_espera, goal_estanteria, nav_margin=nav_margin, goal_margin=goal_margin, subsample=False)

    # ruta 2: estantería → descarga
    path2 = plan_path(map_path, goal_estanteria, zona_descarga, nav_margin=nav_margin, goal_margin=goal_margin, subsample=False)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(grid_nav, cmap="gray_r", origin="lower")

    # tramo 1 — espera → estantería (azul)
    if path1:
        rows1 = [mundo_a_celdas(x, y, origen, resolution)[0] for x, y in path1]
        cols1 = [mundo_a_celdas(x, y, origen, resolution)[1] for x, y in path1]
        ax.plot(cols1, rows1, "b-", linewidth=2, label=f"espera → {shelf['goal_id']}")

    # tramo 2 — estantería → descarga (naranja)
    if path2:
        rows2 = [mundo_a_celdas(x, y, origen, resolution)[0] for x, y in path2]
        cols2 = [mundo_a_celdas(x, y, origen, resolution)[1] for x, y in path2]
        ax.plot(cols2, rows2, color="orange", linewidth=2, label=f"{shelf['goal_id']} → descarga")

    # puntos clave
    r, c = mundo_a_celdas(zona_espera[0],   zona_espera[1],   origen, resolution)
    ax.plot(c, r, "gs", markersize=12, label="zona espera")

    r, c = mundo_a_celdas(goal_estanteria[0], goal_estanteria[1], origen, resolution)
    ax.plot(c, r, "b*", markersize=14, label="estantería")

    r, c = mundo_a_celdas(zona_descarga[0], zona_descarga[1], origen, resolution)
    ax.plot(c, r, "rs", markersize=12, label="zona descarga")

    ax.legend()
    ax.set_title(f"Ciclo completo: espera → {shelf['goal_id']} → descarga")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Ciclo guardado en {filename}")


if __name__ == "__main__":
    with open("warehouse_map01.json", "r") as f:
        warehouse = json.load(f)
    
    # grid de navegación con margen grande para visualizar
    grid_nav, map_info = build_grid("warehouse_map01.json", margin=0.4)
    visualizar_grid(grid_nav, map_info, warehouse, "grid.png")
    
    origen     = map_info["origin"]
    resolution = map_info["resolution"]
    
    start_pos = (-6.6, -8.5)
    goal_pos = (4.0, -8.3) 
    
    path_world = plan_path("warehouse_map01.json", start_pos, goal_pos)
    
    start_cell = mundo_a_celdas(start_pos[0], start_pos[1], origen, resolution)
    goal_cell  = mundo_a_celdas(goal_pos[0],  goal_pos[1],  origen, resolution)
    
    if path_world:
        path_cells = [mundo_a_celdas(x, y, origen, resolution) for x, y in path_world]
        visualizar_camino(grid_nav, path_cells, start_cell, goal_cell, map_info, warehouse, "path.png")
        print(f"Ruta final: {len(path_world)} waypoints")
    
    # visualizar ciclo completo para cada estantería
    for i in range(len(warehouse["shelves"])):
        visualizar_ciclo_completo(
            "warehouse_map01.json",
            shelf_idx=i,
            filename=f"ciclo_{warehouse['shelves'][i]['goal_id']}.png",
            nav_margin=1.2,
            goal_margin=0.3
        )