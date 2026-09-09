"""
Genera dos visualizaciones adicionales para la memoria del TFM, a partir del
mismo warehouse_map01.json y global_planner.py usados por el pipeline STH-WP.

1) obstaculos_inflacion.png
   Mapa de ocupacion con 3 niveles: obstaculo fisico (negro), margen de
   inflacion de navegacion (gris), espacio libre (blanco). Superpone base
   (zona de espera), zona de entrega y ubicaciones de recogida (goals).

2) obstaculo_dinamico_replanificacion.png
   Dos paneles del mismo trayecto (zona_entrega -> goal_23, que cruza el
   pasillo central y=0.3 donde oscila PEDESTRIAN_1). Este es el UNICO peaton
   presente en el mundo final de una sola persona (warehouse_1_1ped.wbt,
   trayectoria x en [-4, 4] a y=0.3); PEDESTRIAN_2 (pasillo izquierdo) solo
   existe en los mundos de dos peatones (warehouse_1.wbt) y no se usa aqui.
     - Antes: ruta A* original sin peaton.
     - Despues: peaton presente cerca del centro de su trayectoria, bloqueo
       temporal alrededor de su posicion, ruta recalculada con A* sobre el
       grid bloqueado.
   Es una figura ilustrativa del concepto de bloqueo temporal + replanificacion
   A*, no una captura literal del comportamiento en tiempo de ejecucion (el
   pipeline real usa un path precomputado + subgoal proyectado, ver
   ObstaculosDinamicos.md).

Uso:
    python3 generate_extra_visualizations.py
"""
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

from global_planner import (build_grid, mundo_a_celdas, celdas_a_mundo,
                             find_nearest_free, astar)

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_PATH = os.path.join(HERE, "warehouse_map01.json")
OUT_DIR = HERE

NAV_MARGIN = 1.2   # margen de inflacion para navegacion (STH-WP: webots_env._precompute_paths, grid_nav)
GOAL_MARGIN = 0.3  # margen mas ajustado usado solo para validar el goal (grid_goal)

# Peaton PEDESTRIAN_1, el UNICO presente en el mundo final de un solo peaton
# (warehouse_1_1ped.wbt): trayectoria fija y=0.3, x oscilando entre -4 y 4,
# velocidad 0.5 m/s. Se ilustra cerca del centro de su recorrido (x=0).
# El pasillo y=0.3 forma parte del atrio central, muy abierto (por eso un
# bloqueo pequeno apenas desvia la ruta y no se aprecia); pero bloquear TODO
# el rango x=[-4,4] de su oscilacion corta tambien el pasillo vertical
# central y desconecta el almacen. Como compromiso realista se bloquea una
# franja de 5 m (x=[-2.5,2.5], la zona central de su recorrido) con la altura
# real del hueco en ese punto (y=[-0.75,1.0]): esto obliga a un rodeo grande
# y claramente visible sin desconectar el mapa.
PED_X, PED_Y = 0.0, 0.3
PED_BLOCK_X0, PED_BLOCK_X1 = -2.5, 2.5
PED_BLOCK_Y0, PED_BLOCK_Y1 = -0.75, 1.0

# Trayecto de demo: zona de entrega -> goal_23, que cruza exactamente el
# pasillo central y=0.3 donde oscila PEDESTRIAN_1.
USE_RETURN_LEG = False


def grid_bounds(map_info):
    x_min, y_min = map_info["origin"]
    resolution = map_info["resolution"]
    x_max = x_min + map_info["width"] * resolution
    y_max = y_min + map_info["height"] * resolution
    return x_min, x_max, y_min, y_max


def path_world(path_cells, origen, resolution):
    return [celdas_a_mundo(r, c, origen, resolution) for r, c in path_cells]


def snap_to_free(grid, x, y, origen, resolution):
    cell = mundo_a_celdas(x, y, origen, resolution)
    r, c = cell
    if grid[r, c] == 1:
        snapped = find_nearest_free(grid, cell)
        if snapped is not None:
            return snapped
    return cell


# ─────────────────────────────────────────────────────────────────────────
# 1) Obstaculos estaticos e inflacion
# ─────────────────────────────────────────────────────────────────────────

def plot_obstacles_inflation(warehouse, map_info):
    x_min, x_max, y_min, y_max = grid_bounds(map_info)

    grid_raw, _ = build_grid(MAP_PATH, margin=0.0)          # solo huella fisica
    grid_goal, _ = build_grid(MAP_PATH, margin=GOAL_MARGIN)  # margen ajustado (goal)
    grid_infl, _ = build_grid(MAP_PATH, margin=NAV_MARGIN)   # margen de navegacion

    # 0 = libre, 1 = margen de navegacion (1.2 m), 2 = margen de goal (0.3 m),
    # 3 = obstaculo fisico. grid_raw ⊆ grid_goal ⊆ grid_infl, así que se pueden
    # superponer en este orden sin conflicto.
    category = np.zeros_like(grid_infl, dtype=np.int8)
    category[grid_infl == 1] = 1
    category[grid_goal == 1] = 2
    category[grid_raw == 1] = 3

    cmap = ListedColormap(["white", "#c8c8c8", "#808080", "black"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    plt.figure(figsize=(10, 9))
    plt.imshow(category, origin="lower", extent=(x_min, x_max, y_min, y_max),
               cmap=cmap, norm=norm, interpolation="nearest")

    wz = warehouse["zones"]["waiting"]
    dz = warehouse["zones"]["dropoff"]
    gx = [s["goal_x"] for s in warehouse["shelves"]]
    gy = [s["goal_y"] for s in warehouse["shelves"]]

    plt.scatter(gx, gy, s=22, c="dodgerblue", edgecolors="white",
                linewidths=0.4, label="ubicaciones de recogida", zorder=4)
    plt.scatter([wz["x"]], [wz["y"]], marker="s", s=110, c="orange",
                edgecolors="black", linewidths=0.6, label="base", zorder=5)
    plt.scatter([dz["x"]], [dz["y"]], marker="*", s=220, c="limegreen",
                edgecolors="black", linewidths=0.6, label="zona de entrega", zorder=5)

    legend_patches = [
        mpatches.Patch(color="black", label="obstáculo físico"),
        mpatches.Patch(color="#808080", label=f"margen del goal ({GOAL_MARGIN} m)"),
        mpatches.Patch(color="#c8c8c8", label=f"margen de navegación ({NAV_MARGIN} m)"),
        mpatches.Patch(color="white", edgecolor="black", label="espacio transitable"),
    ]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=legend_patches + handles, loc="upper right", fontsize=8,
               framealpha=0.9)

    plt.title("Obstáculos estáticos e inflación de navegación")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.figtext(0.5, 0.01,
                "Parámetros del pipeline STH-WP (webots_env._precompute_paths): "
                f"margen de navegación {NAV_MARGIN} m (grid_nav), margen de goal {GOAL_MARGIN} m (grid_goal). "
                "Por eso los goals caen dentro del margen de 1.2 m: solo el margen de 0.3 m los deja libres.",
                ha="center", fontsize=7.5, wrap=True)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(OUT_DIR, "obstaculos_inflacion.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Guardado {out}")


# ─────────────────────────────────────────────────────────────────────────
# 2) Obstaculo dinamico y replanificacion (dos paneles)
# ─────────────────────────────────────────────────────────────────────────

def plot_dynamic_replanning(warehouse, map_info):
    x_min, x_max, y_min, y_max = grid_bounds(map_info)
    origen, resolution = map_info["origin"], map_info["resolution"]

    grid_nav, _ = build_grid(MAP_PATH, margin=NAV_MARGIN)

    wz = warehouse["zones"]["waiting"]
    dz = warehouse["zones"]["dropoff"]
    goal_23 = next(s for s in warehouse["shelves"] if s["goal_id"] == "goal_23")
    # entrega -> goal_23: cruza exactamente el pasillo central y=0.3 donde
    # oscila PEDESTRIAN_1. Simbolos identicos a obstaculos_inflacion.png:
    # base = cuadrado naranja, zona de entrega = estrella verde.
    start_x, start_y = dz["x"], dz["y"]
    goal_x, goal_y = goal_23["goal_x"], goal_23["goal_y"]

    start_cell = snap_to_free(grid_nav, start_x, start_y, origen, resolution)
    goal_cell = snap_to_free(grid_nav, goal_x, goal_y, origen, resolution)

    # ── Antes: ruta original sin peaton ────────────────────────────────────
    path_before = astar(grid_nav, start_cell, goal_cell)
    if path_before is None:
        raise RuntimeError("No se encontro ruta original — revisar start/goal")
    path_before_xy = path_world(path_before, origen, resolution)

    # ── Despues: bloqueo temporal alrededor del peaton + replanificacion ──
    grid_blocked = grid_nav.copy()
    bx0, bx1 = PED_BLOCK_X0, PED_BLOCK_X1
    by0, by1 = PED_BLOCK_Y0, PED_BLOCK_Y1
    r0, c0 = mundo_a_celdas(bx0, by0, origen, resolution)
    r1, c1 = mundo_a_celdas(bx1, by1, origen, resolution)
    r0, r1 = sorted((max(0, r0), min(grid_blocked.shape[0] - 1, r1)))
    c0, c1 = sorted((max(0, c0), min(grid_blocked.shape[1] - 1, c1)))
    grid_blocked[r0:r1 + 1, c0:c1 + 1] = 1

    start_cell_b = start_cell if grid_blocked[start_cell] == 0 else find_nearest_free(grid_blocked, start_cell)
    goal_cell_b = goal_cell if grid_blocked[goal_cell] == 0 else find_nearest_free(grid_blocked, goal_cell)
    path_after = astar(grid_blocked, start_cell_b, goal_cell_b)
    path_after_xy = path_world(path_after, origen, resolution) if path_after else []

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5), sharex=True, sharey=True)

    for ax in axes:
        ax.imshow(grid_nav, origin="lower", extent=(x_min, x_max, y_min, y_max),
                   cmap="gray_r", interpolation="nearest", alpha=0.85, zorder=1)
        ax.scatter([wz["x"]], [wz["y"]], marker="s", s=90, c="orange",
                   edgecolors="black", linewidths=0.6, zorder=5, label="base")
        ax.scatter([dz["x"]], [dz["y"]], marker="*", s=180, c="limegreen",
                   edgecolors="black", linewidths=0.6, zorder=5, label="zona de entrega")
        ax.scatter([goal_x], [goal_y], marker="o", s=60, c="dodgerblue",
                   edgecolors="white", linewidths=0.6, zorder=5, label="goal_23")

    # Panel izquierdo: antes
    ax = axes[0]
    bx_, by_ = zip(*path_before_xy)
    ax.plot(bx_, by_, "--", color="blue", linewidth=2.2, label="ruta original", zorder=4)
    ax.set_title("Antes — ruta global sin peatón", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)

    # Panel derecho: despues
    ax = axes[1]
    ax.plot(bx_, by_, "--", color="blue", linewidth=1.6, alpha=0.5,
            label="ruta original", zorder=3)
    block_rect = mpatches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                                     facecolor="orange", alpha=0.35,
                                     edgecolor="darkorange", linewidth=1.2,
                                     label="bloqueo temporal", zorder=3)
    ax.add_patch(block_rect)
    ax.scatter([PED_X], [PED_Y], marker="o", s=140, c="red",
               edgecolors="black", linewidths=0.6, zorder=6, label="peatón (PEDESTRIAN_1)")
    if path_after_xy:
        ax_, ay_ = zip(*path_after_xy)
        ax.plot(ax_, ay_, "-", color="green", linewidth=2.4,
                label="ruta recalculada", zorder=5)
    ax.set_title("Después — peatón detectado, ruta recalculada", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)

    for ax in axes:
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    fig.suptitle("Obstáculo dinámico y replanificación — representación ilustrativa",
                 fontsize=14, y=0.98)
    plt.figtext(0.5, 0.015,
                "Peatón PEDESTRIAN_1 (warehouse_1_1ped.wbt), único peatón real del mundo final. "
                "Rutas A* del pipeline STH-WP; posición del peatón y bloqueo definidos manualmente "
                "a modo ilustrativo (ver ObstaculosDinamicos.md).",
                ha="center", fontsize=8, wrap=True)
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.05, right=0.98, wspace=0.08)
    out = os.path.join(OUT_DIR, "obstaculo_dinamico_replanificacion.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Guardado {out}")
    print(f"Ruta original: {len(path_before_xy)} celdas | "
          f"Ruta recalculada: {len(path_after_xy)} celdas")


def main():
    with open(MAP_PATH) as f:
        warehouse = json.load(f)
    _, map_info = build_grid(MAP_PATH, margin=NAV_MARGIN)

    plot_obstacles_inflation(warehouse, map_info)
    plot_dynamic_replanning(warehouse, map_info)


if __name__ == "__main__":
    main()
