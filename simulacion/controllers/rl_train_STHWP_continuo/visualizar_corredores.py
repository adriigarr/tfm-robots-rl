"""
Tipo 3: Vistas ampliadas de los corredores críticos con anchos navegables anotados.
Cuatro zooms:
  A) Corredor central inferior  (y ∈ [-3, 0.8]) — goals 08,09,10 | 13-16
  B) Corredor central superior  (y ∈ [0, 3.0])  — goals 17-19 | 25-28
  C) Zona wall1 izquierda       (x ∈ [-10, -5.5]) — goals 05,08,17,20
  D) Zona wall2 derecha         (x ∈ [7.5, 11])   — goals 11,14,23,26
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from global_planner import build_grid, plan_path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

MAP_PATH    = "warehouse_map01.json"
NAV_MARGIN  = 1.2
GOAL_MARGIN = 0.3
OUT_DIR     = "grid_nav_plots"
os.makedirs(OUT_DIR, exist_ok=True)

with open(MAP_PATH) as f:
    w = json.load(f)

grid, map_info = build_grid(MAP_PATH, margin=NAV_MARGIN)
grid_g, _      = build_grid(MAP_PATH, margin=GOAL_MARGIN)
origen         = map_info["origin"]
resolution     = map_info["resolution"]

shelves_by_id  = {s['goal_id']: s for s in w['shelves']}
zona_espera    = (w['zones']['waiting']['x'],  w['zones']['waiting']['y'])
zona_descarga  = (w['zones']['dropoff']['x'],  w['zones']['dropoff']['y'])

# collision rates v17
col_rate = {
    'goal_01':31.9,'goal_02':30.8,'goal_03':31.9,'goal_04':26.4,
    'goal_05':46.7,'goal_06':36.5,'goal_07':31.4,'goal_08':43.5,
    'goal_09':34.9,'goal_10':41.3,'goal_11':38.8,'goal_12':32.7,
    'goal_13':40.7,'goal_14':38.1,'goal_15':46.2,'goal_16':38.1,
    'goal_17':33.0,'goal_18':39.6,'goal_19':36.4,'goal_20':28.3,
    'goal_21':26.4,'goal_22':23.6,'goal_23':32.1,'goal_24':38.2,
    'goal_25':30.8,'goal_26':30.2,'goal_27':27.4,'goal_28':25.5,
}
col_color = lambda g: ('#c0392b' if col_rate.get(g,0)>40 else
                        '#e67e22' if col_rate.get(g,0)>33 else '#27ae60')

def draw_background(ax, xlim, ylim):
    ax.imshow(grid, cmap='gray_r', origin='lower',
              extent=[origen[0], origen[0]+grid.shape[1]*resolution,
                      origen[1], origen[1]+grid.shape[0]*resolution],
              alpha=0.9, zorder=1)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal')

def draw_goals(ax, goal_ids):
    for gid in goal_ids:
        s  = shelves_by_id[gid]
        gx, gy = s['goal_x'], s['goal_y']
        c  = col_color(gid)
        ax.plot(gx, gy, '*', color=c, markersize=13,
                markeredgecolor='black', markeredgewidth=0.6, zorder=6)
        ax.annotate(f"{gid}\n{col_rate.get(gid,0):.0f}%",
                    xy=(gx, gy), xytext=(gx+0.15, gy+0.15),
                    fontsize=6.5, color=c, fontweight='bold', zorder=7,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

def annotate_width(ax, x0, x1, y, label, color='navy'):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.6))
    ax.text((x0+x1)/2, y+0.07, label, ha='center', va='bottom',
            fontsize=7.5, color=color, fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

import warnings
warnings.filterwarnings('ignore')

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
(ax_a, ax_b), (ax_c, ax_d) = axes

# ─── A) Corredor central inferior: entre bloques 2 y 3 ──────────────────────
draw_background(ax_a, xlim=(-9.5, 10.5), ylim=(-4.0, 1.5))
draw_goals(ax_a, ['goal_05','goal_06','goal_07','goal_08','goal_09','goal_10',
                   'goal_11','goal_12','goal_13','goal_14','goal_15','goal_16'])

# Ruta exit más representativa: goal_08 → descarga
try:
    p = plan_path(MAP_PATH,(shelves_by_id['goal_08']['goal_x'], shelves_by_id['goal_08']['goal_y']),
                  zona_descarga, nav_margin=NAV_MARGIN, goal_margin=GOAL_MARGIN, subsample=False)
    ax_a.plot([pt[0] for pt in p],[pt[1] for pt in p],'-',color='#e67e22',lw=1.8,alpha=0.8,zorder=4)
except: pass
try:
    p = plan_path(MAP_PATH,(shelves_by_id['goal_15']['goal_x'], shelves_by_id['goal_15']['goal_y']),
                  zona_descarga, nav_margin=NAV_MARGIN, goal_margin=GOAL_MARGIN, subsample=False)
    ax_a.plot([pt[0] for pt in p],[pt[1] for pt in p],'-',color='#c0392b',lw=1.8,alpha=0.8,zorder=4)
except: pass

# Ancho corredor y=-0.9 entre bloque norte (-2.5=-3.1+0.6) y bloq sur (-1.85=-0.9-0.95)
# Bloque 2 (goals 05-07 en y=-4.7): cara norte de estantería = -4.7+0.95=−3.75; inflada: −3.75+1.2=−2.55
# Bloque 3 (goals 08-10 en y=-0.9): cara sur de estantería = -0.9-0.95=-1.85; inflada: -1.85-1.2=-3.05
# Anchura libre entre -3.05 y -2.55 = 0.5m — pero los goals 08-10 ESTÁN en y=-0.9 que ya es la zona de clearance
# El corredor real: entre cara norte bloque sur (y=-4.7+0.4=−4.3) y cara sur bloque central (y=-0.9-0.4=−1.3)
# Con nav_margin el corredor libre es entre las inflaciones
# Bloque goals 05-07 en y=-4.7: cara norte = -4.7+0.4=−4.3 → inflada a -4.3+1.2=−3.1
# Bloque goals 13-16 en y=-4.7: cara norte = -4.7+0.4=−4.3 → inflada igual
# Goals 08-10 en y=-0.9: cara sur = -0.9-0.4=−1.3 → inflada a -1.3-1.2=−2.5
# Anchura libre: de -3.1 a -2.5 = 0.6m (!) es muy estrecho
# Revisando con medidas reales de shelf: 1.9x0.8m, goal a 1.1m del frente
# Estantería face (acceso): la dimensión perpendicular al pasillo
# Mejor: medir en el mapa real las posiciones infladas
annotate_width(ax_a, -3.10, -2.50, -2.8, '0.60m libre\n(corredor A*)', color='navy')
ax_a.axhline(-3.10, color='red', lw=0.8, ls='--', alpha=0.7)
ax_a.axhline(-2.50, color='red', lw=0.8, ls='--', alpha=0.7)
ax_a.set_title('A) Corredor central inferior (bloque 2-3)\nnav_margin=1.2m — goals 05-16', fontsize=9, fontweight='bold')
ax_a.set_xlabel('x [m]', fontsize=8); ax_a.set_ylabel('y [m]', fontsize=8)
ax_a.tick_params(labelsize=7)

# ─── B) Corredor central superior: bloque 4 (y=1.5) y paso a dropoff ────────
draw_background(ax_b, xlim=(-9.5, 10.5), ylim=(0.0, 7.5))
draw_goals(ax_b, ['goal_17','goal_18','goal_19','goal_20','goal_21','goal_22',
                   'goal_23','goal_24','goal_25','goal_26','goal_27','goal_28'])
try:
    p = plan_path(MAP_PATH,(shelves_by_id['goal_20']['goal_x'], shelves_by_id['goal_20']['goal_y']),
                  zona_descarga, nav_margin=NAV_MARGIN, goal_margin=GOAL_MARGIN, subsample=False)
    ax_b.plot([pt[0] for pt in p],[pt[1] for pt in p],'-',color='#e67e22',lw=1.8,alpha=0.8,zorder=4)
except: pass
try:
    p = plan_path(MAP_PATH,(shelves_by_id['goal_26']['goal_x'], shelves_by_id['goal_26']['goal_y']),
                  zona_descarga, nav_margin=NAV_MARGIN, goal_margin=GOAL_MARGIN, subsample=False)
    ax_b.plot([pt[0] for pt in p],[pt[1] for pt in p],'-',color='#c0392b',lw=1.8,alpha=0.8,zorder=4)
except: pass

ax_b.plot(*zona_descarga, 'r^', markersize=11, zorder=8, label='Descarga')
# Ancho entre bloque 4 cara norte (1.5+0.4=1.9→inflada 1.9+1.2=3.1) y wall6 (y=9.0-0.1=8.9→inflada 8.9-1.2=7.7)
annotate_width(ax_b, -9.3, 2.1, 6.8, 'wall6 inflada: x∈[−9.3, 2.1]', color='darkred')
ax_b.axvline(-9.3, color='darkred', lw=0.8, ls='--', alpha=0.7)
ax_b.axvline(2.1, color='darkred', lw=0.8, ls='--', alpha=0.7)
ax_b.text(0, 6.0, 'wall6 bloquea\ndescarga original\n(0.0, 10.5)', fontsize=7,
          color='darkred', ha='center', style='italic',
          bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffe0e0', alpha=0.8))
ax_b.set_title('B) Corredor superior + impacto wall6\nnav_margin=1.2m — goals 17-28', fontsize=9, fontweight='bold')
ax_b.set_xlabel('x [m]', fontsize=8); ax_b.set_ylabel('y [m]', fontsize=8)
ax_b.tick_params(labelsize=7)

# ─── C) Zona wall1 izquierda: x ∈ [-10, -5.5] ───────────────────────────────
draw_background(ax_c, xlim=(-9.8, -4.5), ylim=(-6.5, 7.0))
draw_goals(ax_c, ['goal_05','goal_08','goal_17','goal_20'])
try:
    for gid in ['goal_05','goal_08','goal_17','goal_20']:
        s = shelves_by_id[gid]
        p = plan_path(MAP_PATH,(s['goal_x'],s['goal_y']), zona_descarga,
                      nav_margin=NAV_MARGIN, goal_margin=GOAL_MARGIN, subsample=False)
        ax_c.plot([pt[0] for pt in p],[pt[1] for pt in p],'-',
                  color=col_color(gid), lw=1.8, alpha=0.8, zorder=4)
except: pass

# wall1 física en x=-8.0; cara este x=-7.9; inflada x=-7.9+1.2=-6.7
# Goals en x=-6.9; cara izquierda x=-6.9-0.95=-7.85
# Clearance físico: -7.85 - (-7.9) = 0.05m (casi tocando wall1!)
ax_c.axvline(-8.0, color='gray', lw=2.0, ls='-', alpha=0.9, label='Wall1 física')
ax_c.axvline(-6.7, color='red', lw=1.2, ls='--', alpha=0.8, label='Wall1 inflada (+1.2m)')
ax_c.axvline(-6.9+0.95, color='blue', lw=1.2, ls=':', alpha=0.8, label='Cara este estantería')
annotate_width(ax_c, -8.0, -6.9+0.95, 5.5, f'{-6.9+0.95-(-8.0):.2f}m\nclearance físico', color='darkblue')
ax_c.set_title('C) Zona wall1 izquierda (x=-8.0)\ngoals 05,08,17,20 — con menor clearance', fontsize=9, fontweight='bold')
ax_c.set_xlabel('x [m]', fontsize=8); ax_c.set_ylabel('y [m]', fontsize=8)
ax_c.legend(fontsize=6.5, loc='lower right')
ax_c.tick_params(labelsize=7)

# ─── D) Zona wall2 derecha: x ∈ [7.5, 11] ───────────────────────────────────
draw_background(ax_d, xlim=(4.5, 11.0), ylim=(-6.5, 7.0))
draw_goals(ax_d, ['goal_11','goal_14','goal_23','goal_26'])
try:
    for gid in ['goal_11','goal_14','goal_23','goal_26']:
        s = shelves_by_id[gid]
        p = plan_path(MAP_PATH,(s['goal_x'],s['goal_y']), zona_descarga,
                      nav_margin=NAV_MARGIN, goal_margin=GOAL_MARGIN, subsample=False)
        ax_d.plot([pt[0] for pt in p],[pt[1] for pt in p],'-',
                  color=col_color(gid), lw=1.8, alpha=0.8, zorder=4)
except: pass

# wall2 física en x=9.9; cara oeste x=9.8; inflada x=9.8-1.2=8.6
# Goals en x=8.9; cara derecha x=8.9+0.95=9.85 ≈ wall2!
ax_d.axvline(9.9, color='gray', lw=2.0, ls='-', alpha=0.9, label='Wall2 física')
ax_d.axvline(8.6, color='red', lw=1.2, ls='--', alpha=0.8, label='Wall2 inflada (+1.2m)')
ax_d.axvline(8.9-0.95, color='blue', lw=1.2, ls=':', alpha=0.8, label='Cara oeste estantería')
annotate_width(ax_d, 8.9-0.95, 9.9, 5.5, f'{9.9-(8.9-0.95):.2f}m\nclearance físico', color='darkblue')
ax_d.set_title('D) Zona wall2 derecha (x=9.9)\ngoals 11,14,23,26 — con menor clearance', fontsize=9, fontweight='bold')
ax_d.set_xlabel('x [m]', fontsize=8); ax_d.set_ylabel('y [m]', fontsize=8)
ax_d.legend(fontsize=6.5, loc='lower left')
ax_d.tick_params(labelsize=7)

# ─── leyenda global ───────────────────────────────────────────────────────────
legend_elements = [
    plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='#c0392b',
               markeredgecolor='black', markersize=11, label='>40% colisiones'),
    plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='#e67e22',
               markeredgecolor='black', markersize=11, label='33-40% colisiones'),
    plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='#27ae60',
               markeredgecolor='black', markersize=11, label='<33% colisiones'),
    mpatches.Patch(facecolor='black', label='Obstáculo físico'),
    mpatches.Patch(facecolor='#555', label=f'Zona bloqueada A* (nav_margin={NAV_MARGIN}m)'),
]
fig.legend(handles=legend_elements, fontsize=8.5, loc='lower center',
           ncol=5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Corredores críticos — Vista ampliada con nav_margin=1.2m\n'
             'Zona bloqueada (gris oscuro) = off-limits para A*, navegable para robot',
             fontsize=11, fontweight='bold')
plt.tight_layout()
out = f"{OUT_DIR}/corredores_criticos_zoom.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: {out}")
