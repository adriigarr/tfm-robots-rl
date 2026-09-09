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
    """
    Convierte las coordenadas del mundo a celdas del grid
    """
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

def plan_path (map_path, start_pos, goal_pos):
    """
    Función principal:
    - dado un inicio y goal en coordenadas
    - construye la ruta como lista de coordenadas del mundo
    """
    # 1. construir el grid
    grid, map_info = build_grid(map_path)
    
    origen     = map_info["origin"]
    resolution = map_info["resolution"]
    
    # 2. convertir posiciones del mundo a celdas
    start_cell = mundo_a_celdas(start_pos[0], start_pos[1], origen, resolution)
    goal_cell  = mundo_a_celdas(goal_pos[0],  goal_pos[1],  origen, resolution)
    
    rows, cols = grid.shape
    print(f"Grid: {rows}x{cols}")
    print(f"Start cell: {start_cell} → valor en grid: {grid[start_cell[0]][start_cell[1]]}")
    print(f"Goal cell:  {goal_cell}  → valor en grid: {grid[goal_cell[0]][goal_cell[1]]}")
    print(f"Celdas bloqueadas: {grid.sum()} de {rows*cols}")

    
    # 3. ejecutar A*
    path_cells = astar(grid, start_cell, goal_cell)
    
    if path_cells is None:
        print(f"[A*] No hay camino de {start_pos} a {goal_pos}")
        return None
    
    # 4. convertir celdas a coordenadas del mundo
    path_world = [
        celdas_a_mundo(r, c, origen, resolution)
        for r, c in path_cells
    ]
    path_world = subsample_path(path_world, step=2.0)
    print(f"[A*] Ruta subsampled: {len(path_world)} waypoints")
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


if __name__ == "__main__":
    with open("warehouse_map01.json", "r") as f:
        warehouse = json.load(f)
    
    grid, map_info = build_grid("warehouse_map01.json")
    visualizar_grid(grid, map_info, warehouse, "grid.png")
    
    origen     = map_info["origin"]
    resolution = map_info["resolution"]
    
    start_pos = (-6.6, -8.5)
    goal_pos  = (8.85, -4.25)
    
    # usar plan_path que incluye el subsampling
    path_world = plan_path("warehouse_map01.json", start_pos, goal_pos)
    
    # para visualizar necesitamos también las celdas
    start_cell = mundo_a_celdas(start_pos[0], start_pos[1], origen, resolution)
    goal_cell  = mundo_a_celdas(goal_pos[0],  goal_pos[1],  origen, resolution)
    
    if path_world:
        path_cells = [mundo_a_celdas(x, y, origen, resolution) for x, y in path_world]
        visualizar_camino(grid, path_cells, start_cell, goal_cell, map_info, warehouse, "path.png")
        print(f"Ruta final: {len(path_world)} waypoints")