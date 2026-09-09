"""
Tipo 2: Ciclos completos (espera→goal→descarga) con nav_margin=1.2m
para los 10 goals con peor rendimiento.
"""
import json, sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
from global_planner import build_grid, plan_path, mundo_a_celdas

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

MAP_PATH   = "warehouse_map01.json"
NAV_MARGIN = 1.2
GOAL_MARGIN = 0.3
OUT_DIR    = "grid_nav_plots"
os.makedirs(OUT_DIR, exist_ok=True)

with open(MAP_PATH) as f:
    w = json.load(f)

grid, map_info = build_grid(MAP_PATH, margin=NAV_MARGIN)
origen     = map_info["origin"]
resolution = map_info["resolution"]

zona_espera   = (w['zones']['waiting']['x'],  w['zones']['waiting']['y'])
zona_descarga = (w['zones']['dropoff']['x'],  w['zones']['dropoff']['y'])

shelves_by_id = {s['goal_id']: s for s in w['shelves']}

# Los 10 goals con peor tasa de éxito en v17
peores_10 = [
    ('goal_15', 53.8, 'Tipo 4'),
    ('goal_05', 53.3, 'Tipo 2'),
    ('goal_10', 56.0, 'Tipo 3'),
    ('goal_11', 56.7, 'Tipo 3'),
    ('goal_08', 56.5, 'Tipo 4'),
    ('goal_13', 59.3, 'Tipo 1'),
    ('goal_18', 60.4, 'Tipo 3'),
    ('goal_14', 61.9, 'Tipo 4'),
    ('goal_16', 61.9, 'Tipo 4'),
    ('goal_24', 61.8, 'Tipo 4'),
]

import warnings
warnings.filterwarnings('ignore')

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.flatten()

for ax, (gid, tasa, tipo) in zip(axes, peores_10):
    shelf = shelves_by_id[gid]
    gx, gy = shelf['goal_x'], shelf['goal_y']

    # Fondo: grid con nav_margin=1.2m
    ax.imshow(grid, cmap='gray_r', origin='lower',
              extent=[origen[0], origen[0] + grid.shape[1]*resolution,
                      origen[1], origen[1] + grid.shape[0]*resolution],
              alpha=0.85)

    # Path approach
    try:
        path1 = plan_path(MAP_PATH, zona_espera, (gx, gy),
                          nav_margin=NAV_MARGIN, goal_margin=GOAL_MARGIN, subsample=False)
        if path1:
            xs = [p[0] for p in path1]
            ys = [p[1] for p in path1]
            ax.plot(xs, ys, '-', color='#2980b9', linewidth=2.0, label='approach', zorder=4)
            ax.plot(xs[-1], ys[-1], 'b^', markersize=6, zorder=5)
    except Exception as e:
        pass

    # Path exit
    try:
        path2 = plan_path(MAP_PATH, (gx, gy), zona_descarga,
                          nav_margin=NAV_MARGIN, goal_margin=GOAL_MARGIN, subsample=False)
        if path2:
            xs = [p[0] for p in path2]
            ys = [p[1] for p in path2]
            ax.plot(xs, ys, '-', color='#e67e22', linewidth=2.0, label='exit', zorder=4)
    except Exception as e:
        pass

    # Puntos clave
    ax.plot(*zona_espera,   'gs', markersize=9, zorder=6)
    ax.plot(*zona_descarga, 'r^', markersize=9, zorder=6)
    ax.plot(gx, gy, '*', color='gold', markersize=12, markeredgecolor='black',
            markeredgewidth=0.8, zorder=7)

    ax.set_title(f'{gid}  [{tipo}]  {tasa}% éxito', fontsize=8.5, fontweight='bold')
    ax.set_xlim(origen[0], origen[0] + grid.shape[1]*resolution)
    ax.set_ylim(origen[1], origen[1] + grid.shape[0]*resolution)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=6)

legend_elements = [
    mpatches.Patch(color='#2980b9', label='Approach (espera→goal)'),
    mpatches.Patch(color='#e67e22', label='Exit (goal→descarga)'),
    plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='gold',
               markeredgecolor='black', markersize=10, label='Goal'),
    plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='green',
               markersize=9, label='Zona espera'),
    plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='red',
               markersize=9, label='Zona descarga'),
]
fig.legend(handles=legend_elements, fontsize=8, loc='lower center',
           ncol=5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Ciclos completos — 10 goals con peor rendimiento (v17)\n'
             'Grid fondo: nav_margin=1.2m  |  Paths planificados con nav_margin=1.2m / goal_margin=0.3m',
             fontsize=11, fontweight='bold')
plt.tight_layout()
out = f"{OUT_DIR}/ciclos_10_goals_dificiles.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: {out}")
