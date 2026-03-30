# controllers/rl_train/mapping/occupancy_grid.py
import math
import numpy as np

def make_grid(bounds, cell_size):
    x_min, x_max, y_min, y_max = bounds
    W = int(math.ceil((x_max - x_min) / cell_size))
    H = int(math.ceil((y_max - y_min) / cell_size))
    occ = np.zeros((H, W), dtype=np.uint8)
    origin = (x_min, y_min)
    return occ, origin

def world_to_grid(x, y, origin, cell_size):
    ox, oy = origin
    return int((x - ox) / cell_size), int((y - oy) / cell_size)

def grid_to_world(gx, gy, origin, cell_size):
    ox, oy = origin
    return ox + (gx + 0.5) * cell_size, oy + (gy + 0.5) * cell_size

def rasterize_aabb(occ, aabb, origin, cell_size):
    H, W = occ.shape
    x0, y0, x1, y1 = aabb
    gx0, gy0 = world_to_grid(x0, y0, origin, cell_size)
    gx1, gy1 = world_to_grid(x1, y1, origin, cell_size)
    gx0, gx1 = sorted((gx0, gx1))
    gy0, gy1 = sorted((gy0, gy1))
    gx0 = max(0, min(W - 1, gx0)); gx1 = max(0, min(W - 1, gx1))
    gy0 = max(0, min(H - 1, gy0)); gy1 = max(0, min(H - 1, gy1))
    occ[gy0:gy1 + 1, gx0:gx1 + 1] = 1

def inflate(occ, radius_cells):
    if radius_cells <= 0:
        return occ
    H, W = occ.shape
    out = occ.copy()
    offsets = []
    r2 = radius_cells * radius_cells
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx*dx + dy*dy <= r2:
                offsets.append((dy, dx))
    ys, xs = np.where(occ == 1)
    for y, x in zip(ys, xs):
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                out[ny, nx] = 1
    return out


def _yaw_from_axis_angle(rot):
    ax, ay, az, angle = rot
    # Rotación alrededor de Z
    if abs(az) > 0.9:
        return angle if az >= 0 else -angle
    return 0.0


def aabb2d_from_solid_box_bounding_object(solid_node):
    """
    AABB 2D del boundingObject Box teniendo en cuenta la rotación yaw (Z).
    """
    if solid_node is None:
        return None

    bof = solid_node.getField("boundingObject")
    if bof is None:
        return None
    bo = bof.getSFNode()
    if bo is None:
        return None

    sx = sy = None
    if bo.getTypeName() == "Box":
        size = bo.getField("size").getSFVec3f()
        sx, sy = float(size[0]), float(size[1])
    elif bo.getTypeName() == "Shape":
        geom = bo.getField("geometry").getSFNode()
        if geom is None or geom.getTypeName() != "Box":
            return None
        size = geom.getField("size").getSFVec3f()
        sx, sy = float(size[0]), float(size[1])
    else:
        return None

    # Posición global del Solid
    px, py, _ = solid_node.getPosition()

    # Rotación yaw del Solid (axis-angle)
    yaw = 0.0
    rot_field = solid_node.getField("rotation")
    if rot_field is not None:
        yaw = _yaw_from_axis_angle(rot_field.getSFRotation())

    # AABB del rectángulo rotado:
    # hx = |cos|*(sx/2) + |sin|*(sy/2)
    # hy = |sin|*(sx/2) + |cos|*(sy/2)
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    hx = c * (sx / 2.0) + s * (sy / 2.0)
    hy = s * (sx / 2.0) + c * (sy / 2.0)

    return (px - hx, py - hy, px + hx, py + hy)

def build_occupancy_grid_from_defs(supervisor, bounds, cell_size, obstacle_defs, r_infl_m):
    occ, origin = make_grid(bounds, cell_size)

    painted = 0
    missing = []
    skipped = []  # existe el DEF pero no tiene Box/Shape->Box en boundingObject

    for d in obstacle_defs:
        node = supervisor.getFromDef(d)
        if node is None:
            missing.append(d)
            continue

        aabb = aabb2d_from_solid_box_bounding_object(node)
        if aabb is None:
            skipped.append(d)
            continue

        rasterize_aabb(occ, aabb, origin, cell_size)
        painted += 1

    inflation_cells = int(math.ceil(r_infl_m / cell_size))
    occ = inflate(occ, inflation_cells)

    return occ, origin, painted, missing, skipped

def save_occupancy_png(occ, bounds, out_path="occ_grid.png"):
    """
    Guarda el occupancy grid como imagen PNG.
    occ: np.array (H,W) con 0 libre, 1 ocupado
    bounds: (x_min, x_max, y_min, y_max)
    """
    import matplotlib.pyplot as plt

    x_min, x_max, y_min, y_max = bounds

    plt.figure()
    # origin="lower" para que el eje Y vaya hacia arriba como un mapa
    plt.imshow(occ, origin="lower", extent=(x_min, x_max, y_min, y_max))
    plt.title("Occupancy Grid")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.savefig(out_path, dpi=200)
    plt.close()

def collect_defs_by_prefix(supervisor, prefixes=("SHELF_", "wall")):
    root = supervisor.getRoot()
    children = root.getField("children")
    out = []
    for i in range(children.getCount()):
        node = children.getMFNode(i)
        try:
            d = node.getDef()
        except Exception:
            d = ""
        if d and any(d.startswith(p) for p in prefixes):
            out.append(d)
    return sorted(out)

def save_occupancy_png_with_all_values(
    occ,
    bounds,
    cell_size,
    out_path="occ_grid_all_values.png",
    fontsize=4
):
    """
    Guarda un PNG escribiendo el valor 0/1 en TODAS las celdas.
    OJO: en grids grandes se verá muy denso.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x_min, x_max, y_min, y_max = bounds
    H, W = occ.shape

    # Figura grande para meter todos los números
    plt.figure(figsize=(20, 20))
    plt.imshow(occ, origin="lower", extent=(x_min, x_max, y_min, y_max), interpolation="nearest")
    plt.title("Occupancy Grid (ALL values)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    # Centros de celdas
    x_centers = x_min + (np.arange(W) + 0.5) * cell_size
    y_centers = y_min + (np.arange(H) + 0.5) * cell_size

    # Escribir TODOS los valores
    for gy in range(H):
        for gx in range(W):
            v = int(occ[gy, gx])
            plt.text(
                x_centers[gx],
                y_centers[gy],
                str(v),
                ha="center",
                va="center",
                fontsize=fontsize
            )

    # Rejilla de celdas (muy fina)
    x_edges = np.linspace(x_min, x_max, W + 1)
    y_edges = np.linspace(y_min, y_max, H + 1)
    plt.gca().set_xticks(x_edges, minor=True)
    plt.gca().set_yticks(y_edges, minor=True)
    plt.grid(which="minor", linewidth=0.2)

    plt.savefig(out_path, dpi=450)
    plt.close()

from collections import deque
import numpy as np

def reachable_mask_from_start(occ, start_cell):
    """
    Devuelve una máscara booleana reachable[H,W] indicando qué celdas libres
    son alcanzables desde start_cell usando conectividad 4 (arriba/abajo/izq/der).
    """
    H, W = occ.shape
    sx, sy = start_cell

    reachable = np.zeros((H, W), dtype=bool)

    # Start fuera de rango o en obstáculo
    if not (0 <= sx < W and 0 <= sy < H):
        return reachable
    if occ[sy, sx] == 1:
        return reachable

    q = deque()
    q.append((sx, sy))
    reachable[sy, sx] = True

    neighbors = [(1,0), (-1,0), (0,1), (0,-1)]
    while q:
        x, y = q.popleft()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                if not reachable[ny, nx] and occ[ny, nx] == 0:
                    reachable[ny, nx] = True
                    q.append((nx, ny))

    return reachable

def save_goals_accessibility_plot(
    occ,
    bounds,
    start_xy,
    goals_xy,
    reachable_mask,
    out_path="goals_accessibility.png"
):
    """
    Dibuja el occupancy grid y marca goals:
      - verdes si están en celda libre y reachable_mask=True
      - rojos si no
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x_min, x_max, y_min, y_max = bounds

    plt.figure(figsize=(8, 8))
    plt.imshow(occ, origin="lower", extent=(x_min, x_max, y_min, y_max), interpolation="nearest")
    plt.title("Goal accessibility on occupancy grid")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    # Start
    plt.scatter([start_xy[0]], [start_xy[1]], marker="x", s=80)

    # Goals
    greens_x, greens_y = [], []
    reds_x, reds_y = [], []

    for (gx, gy, is_ok) in goals_xy:
        if is_ok:
            greens_x.append(gx); greens_y.append(gy)
        else:
            reds_x.append(gx); reds_y.append(gy)

    if greens_x:
        plt.scatter(greens_x, greens_y, s=20, label="reachable")
    if reds_x:
        plt.scatter(reds_x, reds_y, s=20, label="not reachable")

    plt.legend()
    plt.savefig(out_path, dpi=250)
    plt.close()