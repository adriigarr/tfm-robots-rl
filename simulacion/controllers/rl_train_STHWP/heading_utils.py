import math
from global_planner import build_grid, plan_path


def compute_theoretical_headings(map_path, goals, zona_espera,
                                  nav_margin=1.2, goal_margin=0.3):
    """
    Para cada goal, calcula el ángulo de llegada teórico del path A* de approach.
    El ángulo es la dirección del último segmento del path (espera → goal),
    en radianes, medido desde el eje +X en sentido antihorario en el plano XY.

    Devuelve dict {goal_idx: heading_rad}.

    Los grids se construyen una sola vez para todos los goals.
    """
    grid_nav, map_info = build_grid(map_path, margin=nav_margin)
    grid_goal, _ = build_grid(map_path, margin=goal_margin)

    headings = {}
    for idx, goal_pos in enumerate(goals):
        path = plan_path(
            map_path, zona_espera, goal_pos,
            nav_margin=nav_margin, goal_margin=goal_margin,
            subsample=False, verbose=False,
            grid_nav=grid_nav, grid_goal=grid_goal, map_info=map_info,
        )
        if path and len(path) >= 2:
            p1 = path[-2]
            p2 = path[-1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            headings[idx] = math.atan2(dy, dx)
        else:
            headings[idx] = 0.0
            print(f"[heading_utils] WARN: no path for goal idx={idx}, using 0.0 rad")

    return headings


def heading_to_webots_rotation(heading_rad):
    """
    Convierte un ángulo de heading (plano XY, desde +X, antihorario)
    al formato de rotación de Webots [axis_x, axis_y, axis_z, angle].

    En este mundo Webots el eje vertical (yaw) es Z (no Y).
    Las estanterías y paredes usan 'rotation 0 0 1 angle', y el viewpoint
    está a z=75m mirando hacia abajo, confirmando Z=arriba.

    La fórmula [0, 0, 1, -heading_rad] produce un yaw correcto:
    - heading=0    → robot mira hacia +X (este)
    - heading=π/2  → robot mira hacia +Y (norte)
    - heading=-π   → robot mira hacia -X (oeste)

    NOTA: la convención anterior [0,1,0,-θ] producía pitch (inclinación)
    alrededor del eje Y horizontal. Para ángulos pequeños (stages 3-4) el
    pitch era tolerable; para θ≈-108° (stage 5 return) el robot volcaba y
    el bumper penetraba el suelo, disparando colisión en el paso 1.
    """
    return [0, 0, 1, -heading_rad]
