"""
Visualiza la proximidad del robot MiR100 a los obstáculos en la posición del goal.
Genera un plot por grupo de goals representativos.
"""
import json, math, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

MAP_PATH = "warehouse_map01.json"
OUT_DIR  = "clearance_plots"
os.makedirs(OUT_DIR, exist_ok=True)

with open(MAP_PATH) as f:
    w = json.load(f)

SHELF_W, SHELF_H = 1.9, 0.8
NAV_MARGIN  = 1.2
GOAL_MARGIN = 0.3
MIR_W, MIR_L   = 0.58, 0.89          # ancho x largo del MiR100
MIR_HALF_DIAG  = math.sqrt((MIR_W/2)**2 + (MIR_L/2)**2)  # ≈ 0.53m

def draw_shelf_zone(ax, sx, sy, margin, color, alpha, label=None):
    """Dibuja el rectángulo inflado de una estantería."""
    x0 = sx - SHELF_W/2 - margin
    y0 = sy - SHELF_H/2 - margin
    w  = SHELF_W + 2*margin
    h  = SHELF_H + 2*margin
    rect = patches.Rectangle((x0, y0), w, h,
                               linewidth=1, edgecolor=color,
                               facecolor=color, alpha=alpha, label=label)
    ax.add_patch(rect)

def draw_wall_zone(ax, wall, margin, color, alpha):
    wx, wy = wall['x'], wall['y']
    ww, wh = wall['w'], wall['h']
    x0 = wx - ww/2 - margin
    y0 = wy - wh/2 - margin
    w  = ww + 2*margin
    h  = wh + 2*margin
    rect = patches.Rectangle((x0, y0), w, h,
                               linewidth=0.5, edgecolor=color,
                               facecolor=color, alpha=alpha)
    ax.add_patch(rect)

def draw_mir_footprint(ax, gx, gy, approach_angle_deg=90):
    """
    Dibuja el footprint del MiR100 en la posición del goal:
    - rectángulo orientado según approach_angle (en reposo)
    - círculo de half-diagonal (peor caso rotación)
    """
    theta = math.radians(approach_angle_deg)
    # Rectángulo orientado
    corners_local = [
        (-MIR_W/2, -MIR_L/2), ( MIR_W/2, -MIR_L/2),
        ( MIR_W/2,  MIR_L/2), (-MIR_W/2,  MIR_L/2)
    ]
    corners_world = []
    for lx, ly in corners_local:
        rx = gx + lx*math.cos(theta) - ly*math.sin(theta)
        ry = gy + lx*math.sin(theta) + ly*math.cos(theta)
        corners_world.append((rx, ry))
    poly = plt.Polygon(corners_world, closed=True,
                       facecolor='royalblue', edgecolor='navy',
                       alpha=0.5, linewidth=1.5, label='MiR100 (reposo)')
    ax.add_patch(poly)
    # Círculo de half-diagonal (rotación)
    circle = plt.Circle((gx, gy), MIR_HALF_DIAG,
                         fill=False, edgecolor='red',
                         linestyle='--', linewidth=1.5, label=f'Half-diag={MIR_HALF_DIAG:.2f}m (giro)')
    ax.add_patch(circle)
    ax.plot(gx, gy, 'r+', markersize=10, markeredgewidth=2)

def plot_goal_clearance(ax, shelf, nearby_shelves, nearby_walls, title):
    sx, sy = shelf['shelf_x'], shelf['shelf_y']
    gx, gy = shelf['goal_x'], shelf['goal_y']

    # Determinar dirección de approach
    approach_angle = 90 if gy < sy else 270  # sur → apunta norte; norte → apunta sur

    # 1. Zona nav_margin (fondo gris oscuro)
    for ns in nearby_shelves:
        draw_shelf_zone(ax, ns['shelf_x'], ns['shelf_y'], NAV_MARGIN,
                        color='#555555', alpha=0.18)
    for wall in nearby_walls:
        draw_wall_zone(ax, wall, NAV_MARGIN, color='#555555', alpha=0.18)

    # 2. Zona goal_margin (gris medio)
    for ns in nearby_shelves:
        draw_shelf_zone(ax, ns['shelf_x'], ns['shelf_y'], GOAL_MARGIN,
                        color='#888888', alpha=0.35)
    for wall in nearby_walls:
        draw_wall_zone(ax, wall, GOAL_MARGIN, color='#888888', alpha=0.35)

    # 3. Física de las estanterías (negro)
    for ns in nearby_shelves:
        draw_shelf_zone(ax, ns['shelf_x'], ns['shelf_y'], 0,
                        color='#222222', alpha=0.85)
    for wall in nearby_walls:
        draw_wall_zone(ax, wall, 0, color='#222222', alpha=0.85)

    # 4. Robot en la posición del goal
    draw_mir_footprint(ax, gx, gy, approach_angle)

    # 5. Goal marker
    ax.plot(gx, gy, 'g*', markersize=14, zorder=10, label='Goal position')

    # 6. Flechas de distancia lateral
    # Cara izq física
    left_face_x = sx - SHELF_W/2
    right_face_x = sx + SHELF_W/2
    ax.annotate('', xy=(left_face_x, gy), xytext=(gx, gy),
                arrowprops=dict(arrowstyle='<->', color='darkorange', lw=1.5))
    ax.text((left_face_x+gx)/2, gy+0.05, f'{gx-left_face_x:.2f}m',
            ha='center', va='bottom', fontsize=7, color='darkorange')
    ax.annotate('', xy=(right_face_x, gy), xytext=(gx, gy),
                arrowprops=dict(arrowstyle='<->', color='darkorange', lw=1.5))
    ax.text((right_face_x+gx)/2, gy+0.05, f'{right_face_x-gx:.2f}m',
            ha='center', va='bottom', fontsize=7, color='darkorange')

    # Cara de acceso
    if gy < sy:
        face_y = sy - SHELF_H/2
    else:
        face_y = sy + SHELF_H/2
    ax.annotate('', xy=(gx, face_y), xytext=(gx, gy),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    mid_y = (face_y + gy) / 2
    ax.text(gx + 0.08, mid_y, f'{abs(face_y-gy):.2f}m',
            ha='left', va='center', fontsize=7, color='purple')

    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # Leyenda minimalista
    legend_elements = [
        patches.Patch(facecolor='#555555', alpha=0.3, label=f'nav_margin={NAV_MARGIN}m'),
        patches.Patch(facecolor='#888888', alpha=0.5, label=f'goal_margin={GOAL_MARGIN}m'),
        patches.Patch(facecolor='#222222', alpha=0.85, label='Shelf físico'),
        patches.Patch(facecolor='royalblue', alpha=0.5, label='MiR100 (reposo)'),
        plt.Line2D([0],[0], color='red', ls='--', label=f'Half-diag {MIR_HALF_DIAG:.2f}m (giro)'),
    ]
    ax.legend(handles=legend_elements, fontsize=6.5, loc='best')
    return ax

# ── helper: buscar shelf por goal_id ──────────────────────────────────────────
shelves_by_id = {s['goal_id']: s for s in w['shelves']}

def shelves_near(sx, sy, radius=4.5):
    return [s for s in w['shelves']
            if abs(s['shelf_x']-sx) < radius and abs(s['shelf_y']-sy) < radius]

def walls_near(sx, sy, radius=5.0):
    result = []
    for wall in w.get('walls', []):
        if abs(wall['x']-sx) < radius and abs(wall['y']-sy) < radius:
            result.append(wall)
    return result

# ── FIGURA 1: goals representativos (fila sur y goals exteriores) ─────────────
selected = [
    ('goal_03', 'goal_03 — caso más ajustado lateralmente (0.32m margen físico)'),
    ('goal_05', 'goal_05 — más cercano a wall1 (d_pared=1.0m)'),
    ('goal_08', 'goal_08 — north face + wall1 (caso más estrecho)'),
    ('goal_11', 'goal_11 — más cercano a wall2 (d_pared=1.0m)'),
    ('goal_15', 'goal_15 — Tipo 4 zigzag + 0.42m margen lateral'),
    ('goal_20', 'goal_20 — Tipo 1 exit pero x=−6.9 (pared oeste)'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, (gid, title) in zip(axes, selected):
    shelf = shelves_by_id[gid]
    sx, sy = shelf['shelf_x'], shelf['shelf_y']
    nearby = shelves_near(sx, sy, radius=4.5)
    walls  = walls_near(sx, sy, radius=5.0)
    plot_goal_clearance(ax, shelf, nearby, walls, title)
    # Ajustar límites con zoom en el goal
    ax.set_xlim(sx - 3.5, sx + 3.5)
    ax.set_ylim(sy - 3.5, sy + 3.5)

fig.suptitle('Proximidad del MiR100 a obstáculos físicos en posición de goal\n'
             '(azul=robot en reposo, círculo rojo=huella máxima durante giro, naranja=dist. lateral, morado=dist. cara acceso)',
             fontsize=10)
plt.tight_layout()
out = f"{OUT_DIR}/clearance_goals_representativos.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: {out}")

# ── FIGURA 2: comparativa de los 4 tipos de goal en fila izquierda ────────────
selected2 = [
    ('goal_07', 'goal_07 — Tipo 1, bloque izq. sur (0.42m)'),
    ('goal_10', 'goal_10 — Tipo 3, bloque izq. norte (0.42m)'),
    ('goal_19', 'goal_19 — Tipo 3, bloque izq. medio (0.42m)'),
    ('goal_22', 'goal_22 — Tipo 1, bloque izq. sup. (0.42m)'),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, (gid, title) in zip(axes, selected2):
    shelf = shelves_by_id[gid]
    sx, sy = shelf['shelf_x'], shelf['shelf_y']
    nearby = shelves_near(sx, sy, radius=4.5)
    walls  = walls_near(sx, sy, radius=5.0)
    plot_goal_clearance(ax, shelf, nearby, walls, title)
    ax.set_xlim(sx - 3.0, sx + 3.0)
    ax.set_ylim(sy - 3.0, sy + 3.0)

fig.suptitle('Bloque izquierdo — margen físico lateral durante giro de salida\n'
             '(todos muestran 0.42m de clearance ↔ 0.09m sobre el límite de 0.3m)',
             fontsize=10)
plt.tight_layout()
out2 = f"{OUT_DIR}/clearance_bloque_izquierdo.png"
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: {out2}")

print("Done.")
