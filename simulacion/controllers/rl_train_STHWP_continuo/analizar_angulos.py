"""
Calcula el giro real que el robot debe hacer al inicio del exit:
  heading_salida  →  dirección del primer segmento del path A* goal→descarga

También genera visualizaciones.
"""
import csv, json, math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from global_planner import plan_path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

MAP_PATH    = "warehouse_map01.json"
NAV_MARGIN  = 1.2
GOAL_MARGIN = 0.3
CSV_IN      = "angulos_entrada_salida.csv"
OUT_DIR     = "grid_nav_plots"
os.makedirs(OUT_DIR, exist_ok=True)

with open(MAP_PATH) as f:
    w = json.load(f)
zona_descarga = (w['zones']['dropoff']['x'], w['zones']['dropoff']['y'])

# ── Tasas de colisión reales v17 ─────────────────────────────────────────────
col_rate = {
    'goal_01':31.9,'goal_02':30.8,'goal_03':31.9,'goal_04':26.4,
    'goal_05':46.7,'goal_06':36.5,'goal_07':31.4,'goal_08':43.5,
    'goal_09':34.9,'goal_10':41.3,'goal_11':38.8,'goal_12':32.7,
    'goal_13':40.7,'goal_14':38.1,'goal_15':46.2,'goal_16':38.1,
    'goal_17':33.0,'goal_18':39.6,'goal_19':36.4,'goal_20':28.3,
    'goal_21':26.4,'goal_22':23.6,'goal_23':32.1,'goal_24':38.2,
    'goal_25':30.8,'goal_26':30.2,'goal_27':27.4,'goal_28':25.5,
}

# ── Leer CSV ─────────────────────────────────────────────────────────────────
rows = []
with open(CSV_IN) as f:
    for r in csv.DictReader(f):
        rows.append(r)

# ── Calcular giro real ────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

resultados = []
for r in rows:
    gid = r['shelf_id']
    if not r['salida_x']:
        resultados.append({'goal_id': gid, 'giro_exit_deg': None,
                           'heading_salida': None, 'dir_exit': None,
                           'exit_result': r['hacia_descarga'],
                           'approach_result': r['hacia_estanteria'],
                           'col_rate': col_rate.get(gid, 0)})
        continue

    sx = float(r['salida_x'])
    sy = float(r['salida_y'])
    h  = float(r['heading_salida_deg'])

    # Primer waypoint del path A* exit
    try:
        path = plan_path(MAP_PATH, (sx, sy), zona_descarga,
                         nav_margin=NAV_MARGIN, goal_margin=GOAL_MARGIN,
                         subsample=False)
        if path and len(path) >= 2:
            dx = path[1][0] - path[0][0]
            dy = path[1][1] - path[0][1]
            dir_exit = math.degrees(math.atan2(dy, dx))
        else:
            dir_exit = None
    except:
        dir_exit = None

    if dir_exit is not None:
        giro = dir_exit - h
        giro = (giro + 180) % 360 - 180
    else:
        giro = None

    resultados.append({
        'goal_id':       gid,
        'heading_salida': h,
        'dir_exit':      dir_exit,
        'giro_exit_deg': round(giro, 1) if giro is not None else None,
        'exit_result':   r['hacia_descarga'],
        'approach_result': r['hacia_estanteria'],
        'col_rate':      col_rate.get(gid, 0),
        'sx': sx, 'sy': sy,
    })

# ── Imprimir tabla ────────────────────────────────────────────────────────────
print(f"\n{'Goal':<10} {'Heading arr°':>13} {'Dir exit A*°':>13} {'Giro real°':>11} {'|Giro|°':>8} {'Exit':>8} {'Colisión v17':>13}")
print('-'*78)
for r in resultados:
    g   = r['giro_exit_deg']
    gabs = abs(g) if g is not None else None
    h   = r['heading_salida']
    d   = r['dir_exit']
    hs  = f"{h:.1f}" if h is not None else "N/A"
    ds  = f"{d:.1f}" if d is not None else "N/A"
    gs  = f"{g:.1f}" if g is not None else "N/A"
    gss = f"{gabs:.1f}" if gabs is not None else "N/A"
    print(f"{r['goal_id']:<10} "
          f"{hs:>13} "
          f"{ds:>13} "
          f"{gs:>11} "
          f"{gss:>8} "
          f"{str(r['exit_result']):>8} "
          f"{r['col_rate']:>12.1f}%")

# ── IMAGEN 1: Giro real por goal (barras horizontales) ───────────────────────
datos_validos = [r for r in resultados if r['giro_exit_deg'] is not None]
datos_validos.sort(key=lambda r: abs(r['giro_exit_deg']), reverse=True)

fig, ax = plt.subplots(figsize=(10, 9))
y_pos = range(len(datos_validos))
colores = []
for r in datos_validos:
    if r['col_rate'] > 40:    colores.append('#c0392b')
    elif r['col_rate'] > 33:  colores.append('#e67e22')
    else:                      colores.append('#27ae60')

giros = [r['giro_exit_deg'] for r in datos_validos]
labels = [r['goal_id'] for r in datos_validos]

bars = ax.barh(list(y_pos), giros, color=colores, edgecolor='white', linewidth=0.5)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels, fontsize=8.5)
ax.axvline(0, color='black', lw=0.8)
ax.axvline(90, color='gray', lw=0.8, ls='--', alpha=0.6)
ax.axvline(-90, color='gray', lw=0.8, ls='--', alpha=0.6)
ax.set_xlabel('Giro requerido al inicio del exit [°]\n(positivo = izquierda, negativo = derecha)', fontsize=9)
ax.set_title('Giro real al inicio de la fase exit (heading llegada → dirección A* salida)\n'
             'Medido en la simulación real con ppo_sthwp_17', fontsize=10, fontweight='bold')

# Anotar el valor en cada barra
for bar, r in zip(bars, datos_validos):
    g = r['giro_exit_deg']
    ax.text(g + (3 if g >= 0 else -3), bar.get_y() + bar.get_height()/2,
            f"{g:.1f}°\n{r['col_rate']:.0f}%",
            va='center', ha='left' if g >= 0 else 'right', fontsize=6.5)

legend_elements = [
    mpatches.Patch(color='#c0392b', label='>40% colisiones (v17)'),
    mpatches.Patch(color='#e67e22', label='33-40% colisiones'),
    mpatches.Patch(color='#27ae60', label='<33% colisiones'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/giro_exit_por_goal.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\nGuardado: {OUT_DIR}/giro_exit_por_goal.png")

# ── IMAGEN 2: Correlación |giro| vs tasa de colisión ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
xs = [abs(r['giro_exit_deg']) for r in datos_validos]
ys = [r['col_rate'] for r in datos_validos]

ax.scatter(xs, ys, c=colores, s=80, edgecolors='black', linewidths=0.6, zorder=4)
for r, x, y in zip(datos_validos, xs, ys):
    ax.annotate(r['goal_id'].replace('goal_','g'), xy=(x,y),
                xytext=(x+1, y+0.3), fontsize=6.5, color='#444')

# Línea de tendencia
if len(xs) > 2:
    z = np.polyfit(xs, ys, 1)
    p = np.poly1d(z)
    xr = np.linspace(min(xs), max(xs), 100)
    ax.plot(xr, p(xr), 'r--', lw=1.2, alpha=0.7, label=f'Tendencia (pendiente {z[0]:+.2f}%/°)')
    corr = np.corrcoef(xs, ys)[0,1]
    ax.text(0.05, 0.92, f'r = {corr:.2f}', transform=ax.transAxes,
            fontsize=10, color='red', fontweight='bold')

ax.set_xlabel('|Giro requerido al inicio del exit| [°]', fontsize=10)
ax.set_ylabel('Tasa de colisión — v17 [%]', fontsize=10)
ax.set_title('Correlación entre giro de salida y tasa de colisión', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlacion_giro_colision.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: {OUT_DIR}/correlacion_giro_colision.png")

# ── IMAGEN 3: Mapa completo con flechas de heading real ──────────────────────
from global_planner import build_grid

grid, map_info = build_grid(MAP_PATH, margin=NAV_MARGIN)
origen     = map_info['origin']
resolution = map_info['resolution']

shelves_by_id = {s['goal_id']: s for s in w['shelves']}

fig, ax = plt.subplots(figsize=(13, 11))
ax.imshow(grid, cmap='gray_r', origin='lower',
          extent=[origen[0], origen[0]+grid.shape[1]*resolution,
                  origen[1], origen[1]+grid.shape[0]*resolution],
          alpha=0.85, zorder=1)

for r in resultados:
    gid = r['goal_id']
    if r.get('sx') is None:
        continue
    sx, sy = r['sx'], r['sy']
    h_rad = math.radians(r['heading_salida'])
    d_rad = math.radians(r['dir_exit']) if r['dir_exit'] is not None else None

    # Color según tasa de colisión
    if r['col_rate'] > 40:    c = '#c0392b'
    elif r['col_rate'] > 33:  c = '#e67e22'
    else:                      c = '#27ae60'

    # Flecha de heading de llegada (azul)
    ax.annotate('', xy=(sx + 0.6*math.cos(h_rad), sy + 0.6*math.sin(h_rad)),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2.0), zorder=5)

    # Flecha de dirección A* exit (naranja/roja)
    if d_rad is not None:
        ax.annotate('', xy=(sx + 0.6*math.cos(d_rad), sy + 0.6*math.sin(d_rad)),
                    xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2.0), zorder=5)

    # Punto del goal
    ax.plot(sx, sy, 'o', color=c, markersize=8,
            markeredgecolor='black', markeredgewidth=0.6, zorder=6)

    # Etiqueta con giro
    g = r['giro_exit_deg']
    lbl = f"{gid.replace('goal_','')}\n{g:+.0f}°" if g is not None else gid.replace('goal_','')
    ax.text(sx+0.15, sy+0.12, lbl, fontsize=6, color=c, fontweight='bold', zorder=7,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

ax.plot(*zona_descarga, 'r^', markersize=11, zorder=8)
ax.plot(w['zones']['waiting']['x'], w['zones']['waiting']['y'], 'gs', markersize=11, zorder=8)

legend_elements = [
    mpatches.Patch(color='#2980b9', label='Heading real de llegada'),
    mpatches.Patch(color='#e67e22', label='Dirección A* exit'),
    mpatches.Patch(color='#c0392b', label='>40% colisiones'),
    mpatches.Patch(color='#e67e22', label='33-40% colisiones'),
    mpatches.Patch(color='#27ae60', label='<33% colisiones'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
ax.set_title('Heading real de llegada (azul) vs dirección exit A* (naranja)\n'
             'Etiqueta: número de goal + giro requerido en °', fontsize=10, fontweight='bold')
ax.set_aspect('equal')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/mapa_headings_reales.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: {OUT_DIR}/mapa_headings_reales.png")
