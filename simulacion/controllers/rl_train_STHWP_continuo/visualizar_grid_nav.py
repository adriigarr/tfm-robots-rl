"""
Tipo 1: Mapa completo del almacén con nav_margin=1.2m.
Muestra zonas bloqueadas/libres tal como las ve A*, con los 28 goals y anchos de corredor.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from global_planner import build_grid, mundo_a_celdas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

MAP_PATH   = "warehouse_map01.json"
NAV_MARGIN = 1.2
OUT_DIR    = "grid_nav_plots"
os.makedirs(OUT_DIR, exist_ok=True)

with open(MAP_PATH) as f:
    w = json.load(f)

grid, map_info = build_grid(MAP_PATH, margin=NAV_MARGIN)
origen     = map_info["origin"]
resolution = map_info["resolution"]

fig, ax = plt.subplots(figsize=(14, 12))
ax.imshow(grid, cmap="gray_r", origin="lower",
          extent=[origen[0], origen[0] + grid.shape[1]*resolution,
                  origen[1], origen[1] + grid.shape[0]*resolution])

# Goals coloreados por tipo de salida
tipos = {
    1: (['goal_01','goal_02','goal_03','goal_04','goal_07','goal_13',
         'goal_17','goal_20','goal_21','goal_22','goal_28'], 'Tipo 1 (limpia)', '#2ecc71'),
    2: (['goal_05','goal_06','goal_09','goal_25'],            'Tipo 2 (giro >90°)', '#f39c12'),
    3: (['goal_10','goal_11','goal_12','goal_18','goal_19',
         'goal_23','goal_26','goal_27'],                      'Tipo 3 (zigzag mod.)', '#e74c3c'),
    4: (['goal_08','goal_14','goal_15','goal_16','goal_24'],  'Tipo 4 (zigzag fuerte)', '#8e44ad'),
}
colors_by_id = {}
for t, (ids, _, color) in tipos.items():
    for gid in ids:
        colors_by_id[gid] = color

for shelf in w['shelves']:
    gx, gy = shelf['goal_x'], shelf['goal_y']
    gid = shelf['goal_id']
    c = colors_by_id.get(gid, 'cyan')
    ax.plot(gx, gy, 'o', color=c, markersize=7, zorder=5)
    num = gid.replace('goal_', '')
    ax.text(gx+0.15, gy+0.1, num, fontsize=6, color=c, fontweight='bold', zorder=6)

# Zonas clave
zx, zy = w['zones']['waiting']['x'],  w['zones']['waiting']['y']
dx, dy = w['zones']['dropoff']['x'],  w['zones']['dropoff']['y']
ax.plot(zx, zy, 'gs', markersize=12, label='Zona espera', zorder=7)
ax.plot(dx, dy, 'r^', markersize=12, label='Zona descarga', zorder=7)

# Waypoints corredor retorno
for wp in w['return_corridor']['waypoints']:
    ax.plot(wp['x'], wp['y'], 'b^', markersize=6, zorder=6)
    ax.text(wp['x']+0.1, wp['y']+0.1, wp['id'], fontsize=6, color='blue')

# Anotar anchos de corredor navegable clave
annotations = [
    # (x_text, y_text, texto, x1, y1, x2, y2 para la flecha de ancho)
    (-0.95, -1.5,  'Corredor\ncentral\n3.9m', None, None, None, None),
    (-0.95,  3.0,  'Corredor\ncentral\n3.9m', None, None, None, None),
    # Corredor entre bloque 2 sur y bloque 1
    (-4.5,  -6.5,  'Corredor\n2.9m', None, None, None, None),
    # Corredor entre bloque 2 norte y bloque 4 sur
    (-4.5,   0.3,  'Corredor\n2.2m', None, None, None, None),
]
for (tx, ty, texto, *_) in annotations:
    ax.text(tx, ty, texto, fontsize=7.5, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7),
            zorder=8)

# Leyenda tipos
legend_elements = [
    mpatches.Patch(color='#2ecc71', label='Tipo 1 — salida limpia (~25.6% col.)'),
    mpatches.Patch(color='#f39c12', label='Tipo 2 — giro >90° (~28.7% col.)'),
    mpatches.Patch(color='#e74c3c', label='Tipo 3 — zigzag moderado (~34.2% col.)'),
    mpatches.Patch(color='#8e44ad', label='Tipo 4 — zigzag fuerte (~39.4% col.)'),
    mpatches.Patch(color='white',  edgecolor='black', label='Zona libre A* (nav_margin=1.2m)'),
    mpatches.Patch(color='black',  label='Zona bloqueada A* (nav_margin=1.2m)'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='upper left')
ax.set_xlabel('X (m)', fontsize=10)
ax.set_ylabel('Y (m)', fontsize=10)
ax.set_title('Grid del almacén con nav_margin=1.2m\nZonas blancas = navegables por A*  |  Goals coloreados por tipo de salida y tasa de colisión',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.15, color='gray')

out = f"{OUT_DIR}/grid_completo_nav12.png"
plt.tight_layout()
plt.savefig(out, dpi=160, bbox_inches='tight')
plt.close()
print(f"Guardado: {out}")
