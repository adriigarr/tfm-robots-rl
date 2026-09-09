import json
import math
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor
from global_planner import (build_grid, plan_path, compute_sth_subgoal,
                            astar, mundo_a_celdas, celdas_a_mundo,
                            subsample_path, find_nearest_free)
from heading_utils import heading_to_webots_rotation


class WebotsEnv(gym.Env):
    """
    Entorno Gymnasium para entrenamiento RL en Webots con STH-WP.

    Soporta 5 etapas de entrenamiento progresivo:
      Stage 1 — Approach puro, un solo goal (goal_00). Episodio termina al llegar.
      Stage 2 — Approach puro, todos los goals con distribución uniforme.
      Stage 3 — Exit puro. Robot teleportado a estantería con heading A* + ruido.
                 _fase_escape activa desde el inicio del episodio.
      Stage 4 — Ciclo completo. 70% approach / 30% exit con curriculum.
      Stage 5 — Retorno puro. Robot spawneado en camino descarga→espera.
                 Aprende el tramo final del ciclo completo.
    """

    metadata = {"render_modes": []}

    MAX_LINEAR_SPEED  = 0.5
    MAX_ANGULAR_SPEED = 1.5
    WHEEL_RADIUS      = 0.0625
    AXLE_LENGTH       = 0.445208
    LIDAR_RAYS        = 36
    MAX_LIDAR_RANGE   = 5.0
    MAX_GOAL_DIST     = 15.0
    TIMESTEP_MS       = 32

    # 0-based indices corregidos (bug original usaba números 1-based como índices)
    # nuevo almacén: bloques 3 y 5 (x≈5–9) son los más alejados de la zona de descarga (x=-11)
    # goal_11→idx10 … goal_16→idx15, goal_23→idx22 … goal_26→idx25
    SHELVES_DIFICILES  = {10, 11, 12, 13, 14, 15, 22, 23, 24, 25}
    PROB_SHELF_DIFICIL = 0.7

    # fracción de episodios approach vs exit en stage 4
    PROB_APPROACH_S4 = 0.7

    # _fase_escape: steps con d_ahead reducido tras llegar a estantería
    # nuevo almacén: salida en 44–135° en espacio abierto → 50 steps suficientes
    ESCAPE_DURATION = 50

    # d_ahead para STH-WP: reducido durante _fase_escape para evitar Causa B
    D_AHEAD_NORMAL = 1.5
    D_AHEAD_ESCAPE = 0.5

    # Replanning dinámico mid-episode (solo activo cuando ped_obs=True)
    REPLAN_DIST_PERP    = 0.6   # m — umbral de distancia perpendicular peatón→segmento
    REPLAN_INFLATE_CELLS = 3    # celdas a bloquear alrededor del peatón (3×0.25m = 0.75m)
    REPLAN_COOLDOWN      = 40   # steps sin replanificar tras cada replanning
    LIDAR_MIN_DYNAMIC    = 2.5  # m — E2.2b: subido de 1.5→2.5 para filtrar paredes de pasillos estrechos
    LIDAR_DYN_MAX        = 4.5  # m — E2.2: margen 0.5m del MAX_LIDAR_RANGE

    def __init__(self, map_path="warehouse_map01.json", stage=1,
                 heading_sigma=0.25, render_mode=None, ped_obs=False,
                 pred_horizon=None, enable_dynamic_replanning=True):
        """
        map_path     : ruta al JSON del almacén
        stage        : etapa de entrenamiento (1-6)
        heading_sigma: desviación estándar del ruido gaussiano en el heading
                       de teleport para stage 3 (radianes, ~0.25 rad ≈ 14°)
        ped_obs      : añadir 8 dims de observación de peatones (pos+vel relativa)
                       y aleatorizar su posición inicial en cada reset (stage 6)
        pred_horizon : lista de horizontes en segundos para predicción de P1.
                       Añade 2 dims por horizonte (dx_pred, dy_pred relativos al robot).
                       Ej: [1.0, 2.0] → +4 dims → obs_size 48→52.
                       Requiere ped_obs=True.
        enable_dynamic_replanning : si False, desactiva por completo el recálculo
                       de full_path mid-episodio disparado por el LIDAR al detectar
                       un obstáculo dinámico (peatón). No afecta a la planificación
                       A* inicial de cada tramo (approach/exit/return), que siempre
                       se ejecuta. Usado para la condición de evaluación E2.1-R0.
        """
        super().__init__()

        assert stage in (1, 2, 3, 4, 5, 6), f"stage debe ser 1-6, recibido: {stage}"

        self.render_mode   = render_mode
        self.stage         = stage
        self.heading_sigma = heading_sigma
        self.map_path      = map_path
        self._ped_obs      = ped_obs
        self._pred_horizon = list(pred_horizon) if pred_horizon else []
        self.enable_dynamic_replanning = enable_dynamic_replanning
        self._replan_attempts  = 0
        self._replan_successes = 0

        self.supervisor = Supervisor()
        self.timestep   = self.TIMESTEP_MS

        with open(map_path, "r") as f:
            warehouse = json.load(f)

        self.goals = [
            (shelf["goal_x"], shelf["goal_y"]) for shelf in warehouse["shelves"]
        ]
        self.goal_ids = [shelf["goal_id"] for shelf in warehouse["shelves"]]

        self.zona_espera = (
            warehouse["zones"]["waiting"]["x"],
            warehouse["zones"]["waiting"]["y"],
        )
        self.zona_descarga = (
            warehouse["zones"]["dropoff"]["x"],
            warehouse["zones"]["dropoff"]["y"],
        )

        self.robot_node = self.supervisor.getFromDef("MIR")
        if self.robot_node is None:
            raise RuntimeError("No se encontró el nodo DEF='MIR' en Webots.")

        pos = self.robot_node.getField("translation").getSFVec3f()
        self._start_x = pos[0]
        self._start_y = pos[1]

        self.left_motor  = self.supervisor.getDevice("middle_left_wheel_joint")
        self.right_motor = self.supervisor.getDevice("middle_right_wheel_joint")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.lidar  = self.supervisor.getDevice("lidar")
        self.gps    = self.supervisor.getDevice("gps")
        self.compass = self.supervisor.getDevice("compass")
        self.bumper  = self.supervisor.getDevice("bumper")
        self.lidar.enable(self.timestep)
        self.gps.enable(self.timestep)
        self.compass.enable(self.timestep)
        self.bumper.enable(self.timestep)

        # Observación de peatones: 2 × (dx_rel, dy_rel, vx, vy) = 8 dims
        # Solo activa cuando ped_obs=True; si no hay nodos en el mundo → ceros
        self._ped_nodes    = []
        self._ped_prev_pos = []
        # Rangos de aleatorización: [(x_range_or_fixed, y_range_or_fixed, z), ...]
        # PEDESTRIAN_1: oscila x∈[-2,4], y=0.3, z=1.27
        # PEDESTRIAN_2: oscila y∈[-5,-0.5], x=-9.5, z=1.27
        self._ped_ranges = [
            {"x": (-4.0, 4.0), "y": 0.3,          "z": 1.27},
            {"x": -9.5,        "y": (-5.0, -0.5),  "z": 1.27},
        ]
        if ped_obs:
            for def_name in ["PEDESTRIAN_1", "PEDESTRIAN_2"]:
                node = self.supervisor.getFromDef(def_name)
                if node is not None:
                    self._ped_nodes.append(node)
                    p = node.getField("translation").getSFVec3f()
                    self._ped_prev_pos.append([p[0], p[1]])
            print(f"[WebotsEnv] ped_obs=True — {len(self._ped_nodes)} peatones detectados")

        obs_size = self.LIDAR_RAYS + 4  # lidar + dist_goal + angle_goal + vlin + vang
        self.observation_space = spaces.Box(
            low=np.full(obs_size, -1.0, dtype=np.float32),
            high=np.full(obs_size, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # precomputar todas las rutas A* al arrancar (una sola vez)
        # _precompute_paths también guarda self._grid_nav y self._map_info para replanning
        print(f"[WebotsEnv] Stage {stage} — precomputando rutas A*...")
        self._astar_cache = self._precompute_paths()
        print(f"[WebotsEnv] Caché A* lista: {len(self._astar_cache)} rutas.")

        # headings de teleport para stage 3 (exit puro con teleport en estantería)
        # Prioridad:
        #   1. JSON con headings reales medidos en inferencia stage 2
        #      (arrival_headings_stage2.json, generado por infer_run002_s524_stage2.py)
        #   2. Fallback: primer segmento del path A* exit (goal → descarga)
        self._headings = {}
        if stage >= 3:
            json_path = os.path.join(os.path.dirname(map_path),
                                     "inferencia_sthwp", "resultados",
                                     "arrival_headings_stage2.json")
            if os.path.exists(json_path):
                print(f"[WebotsEnv] Cargando headings reales desde {json_path}")
                with open(json_path) as _f:
                    raw = json.load(_f)
                for idx, gid in enumerate(self.goal_ids):
                    entry = raw.get(gid)
                    if entry and entry.get("mean_rad") is not None:
                        self._headings[idx] = entry["mean_rad"]
                    else:
                        # fallback individual
                        exit_path = self._astar_cache[("exit", idx)]
                        if exit_path and len(exit_path) >= 2:
                            dx = exit_path[1][0] - exit_path[0][0]
                            dy = exit_path[1][1] - exit_path[0][1]
                            self._headings[idx] = math.atan2(dy, dx)
                        else:
                            self._headings[idx] = 0.0
                        print(f"[WebotsEnv] WARN: {gid} sin datos en JSON, usando exit path")
                print("[WebotsEnv] Headings reales cargados.")
            else:
                print("[WebotsEnv] JSON de headings no encontrado. Usando primer segmento exit path.")
                for idx in range(len(self.goals)):
                    exit_path = self._astar_cache[("exit", idx)]
                    if exit_path and len(exit_path) >= 2:
                        dx = exit_path[1][0] - exit_path[0][0]
                        dy = exit_path[1][1] - exit_path[0][1]
                        self._headings[idx] = math.atan2(dy, dx)
                    else:
                        self._headings[idx] = 0.0
                        print(f"[WebotsEnv] WARN: exit path vacío para goal idx={idx}")

        # estado del episodio
        self.current_goal_idx  = 0
        self.goal              = np.zeros(2, dtype=np.float32)
        self.full_path         = []
        self._step_count       = 0
        self._hacia_descarga   = False
        self._en_retorno       = False
        self._fase_escape      = False
        self._escape_steps     = 0
        self._current_lin_vel  = 0.0
        self._current_ang_vel  = 0.0
        self._prev_dist        = 0.0
        self._replan_cooldown  = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Aleatorización de peatones
    # ─────────────────────────────────────────────────────────────────────────

    def _randomize_pedestrians(self):
        """Teletransporta cada peatón a una posición aleatoria en su corredor."""
        for i, node in enumerate(self._ped_nodes):
            r = self._ped_ranges[i]
            x = float(self.np_random.uniform(*r["x"])) if isinstance(r["x"], tuple) else r["x"]
            y = float(self.np_random.uniform(*r["y"])) if isinstance(r["y"], tuple) else r["y"]
            node.getField("translation").setSFVec3f([x, y, r["z"]])
            self._ped_prev_pos[i] = [x, y]

    # ─────────────────────────────────────────────────────────────────────────
    # Precomputación de rutas
    # ─────────────────────────────────────────────────────────────────────────

    def _precompute_paths(self):
        """Calcula 57 rutas A* (28 approach + 28 exit + 1 return) una sola vez.
        Guarda grid_nav y map_info como atributos para replanning mid-episode."""
        grid_nav,  map_info = build_grid(self.map_path, margin=1.2)
        grid_goal, _        = build_grid(self.map_path, margin=0.3)
        self._grid_nav  = grid_nav
        self._map_info  = map_info

        cache = {}
        for idx, goal_pos in enumerate(self.goals):
            # approach: zona_espera → goal
            path = plan_path(
                self.map_path, self.zona_espera, goal_pos,
                subsample=False, verbose=False,
                grid_nav=grid_nav, grid_goal=grid_goal, map_info=map_info,
            )
            cache[("approach", idx)] = path if path else [goal_pos]

            # exit: goal → zona_descarga
            path = plan_path(
                self.map_path, goal_pos, self.zona_descarga,
                subsample=False, verbose=False,
                grid_nav=grid_nav, grid_goal=grid_goal, map_info=map_info,
            )
            cache[("exit", idx)] = path if path else [self.zona_descarga]

        # return: zona_descarga → zona_espera (ruta compartida, idx=0)
        ret_path = plan_path(
            self.map_path, self.zona_descarga, self.zona_espera,
            subsample=False, verbose=False,
            grid_nav=grid_nav, grid_goal=grid_goal, map_info=map_info,
        )
        cache[("return", 0)] = ret_path if ret_path else [self.zona_espera]

        return cache

    # ─────────────────────────────────────────────────────────────────────────
    # Selección de goals
    # ─────────────────────────────────────────────────────────────────────────

    def _sample_goal_approach(self):
        """Selecciona goal para episodios de approach (stages 2 y 4)."""
        return int(self.np_random.integers(0, len(self.goals)))

    def _sample_goal_exit(self):
        """Selecciona goal para episodios de exit (stages 3 y 4). Muestreo uniforme."""
        return int(self.np_random.integers(0, len(self.goals)))

    # ─────────────────────────────────────────────────────────────────────────
    # Reset helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_approach(self, goal_idx, translation, rotation):
        self.current_goal_idx = goal_idx
        translation.setSFVec3f([self._start_x, self._start_y, 0.2])
        rotation.setSFRotation([0, 0, 1, 0])
        self.full_path       = list(self._astar_cache[("approach", goal_idx)])
        self._hacia_descarga = False
        self._fase_escape    = False
        self._escape_steps   = 0
        print(f"[APPROACH] espera → {self.goal_ids[goal_idx]} ({len(self.full_path)} pts)")

    def _reset_exit(self, goal_idx, translation, rotation):
        self.current_goal_idx = goal_idx
        shelf_pos = self.goals[goal_idx]

        theta       = self._headings.get(goal_idx, 0.0)
        theta_noisy = theta + self.np_random.normal(0, self.heading_sigma)
        rot         = heading_to_webots_rotation(theta_noisy)

        translation.setSFVec3f([shelf_pos[0], shelf_pos[1], 0.2])
        rotation.setSFRotation(rot)

        self.full_path       = list(self._astar_cache[("exit", goal_idx)])
        self._hacia_descarga = True
        self._fase_escape    = True
        self._escape_steps   = 0
        print(
            f"[EXIT] {self.goal_ids[goal_idx]} → descarga, "
            f"θ={math.degrees(theta_noisy):.1f}° ({len(self.full_path)} pts)"
        )

    def _reset_return(self, translation, rotation):
        """Spawn en el camino de retorno descarga→espera, lejos de la pared norte."""
        ret_path = self._astar_cache[("return", 0)]
        zd_x, zd_y = self.zona_descarga

        # Primer nodo a ≥3.0m de zona_descarga: evita penetración en wall3 con heading −108°
        spawn_idx = len(ret_path) - 1
        for i, (px, py) in enumerate(ret_path):
            if math.sqrt((px - zd_x) ** 2 + (py - zd_y) ** 2) >= 3.0:
                spawn_idx = i
                break

        spawn_x, spawn_y = ret_path[spawn_idx]
        next_idx = min(spawn_idx + 1, len(ret_path) - 1)
        dx_h = ret_path[next_idx][0] - spawn_x
        dy_h = ret_path[next_idx][1] - spawn_y
        theta       = math.atan2(dy_h, dx_h)
        theta_noisy = theta + self.np_random.normal(0, self.heading_sigma)
        rot         = heading_to_webots_rotation(theta_noisy)

        translation.setSFVec3f([spawn_x, spawn_y, 0.2])
        rotation.setSFRotation(rot)

        self.full_path       = list(ret_path)
        self._hacia_descarga = False
        self._en_retorno     = True
        self._fase_escape    = False
        self._escape_steps   = 0
        print(
            f"[RETURN] descarga→espera, "
            f"spawn=({spawn_x:.2f},{spawn_y:.2f}), θ={math.degrees(theta):.1f}°, "
            f"path={len(ret_path)} pts"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count      = 0
        self._current_lin_vel = 0.0
        self._current_ang_vel = 0.0
        self._en_retorno      = False
        self._replan_cooldown = 0
        self._replan_attempts  = 0
        self._replan_successes = 0
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        translation = self.robot_node.getField("translation")
        rotation    = self.robot_node.getField("rotation")

        if self.stage == 1:
            self._reset_approach(0, translation, rotation)

        elif self.stage == 2:
            idx = self._sample_goal_approach()
            self._reset_approach(idx, translation, rotation)

        elif self.stage == 3:
            idx = self._sample_goal_exit()
            self._reset_exit(idx, translation, rotation)

        elif self.stage == 4:
            if self.np_random.random() < self.PROB_APPROACH_S4:
                idx = self._sample_goal_approach()
                self._reset_approach(idx, translation, rotation)
            else:
                idx = self._sample_goal_exit()
                self._reset_exit(idx, translation, rotation)

        elif self.stage == 5:
            self._reset_return(translation, rotation)

        elif self.stage == 6:
            # ciclo completo: siempre approach (como stage 4 con PROB=1.0)
            idx = self._sample_goal_approach()
            self._reset_approach(idx, translation, rotation)

        self.supervisor.simulationResetPhysics()
        self.supervisor.step(self.timestep)

        if self._ped_obs and self._ped_nodes:
            self._randomize_pedestrians()

        pos = self.gps.getValues()
        d_actual = self.D_AHEAD_ESCAPE if self._fase_escape else self.D_AHEAD_NORMAL
        subgoal  = compute_sth_subgoal(self.full_path, (pos[0], pos[1]), d_actual)
        self.goal = np.array(subgoal, dtype=np.float32)

        dx = self.goal[0] - pos[0]
        dy = self.goal[1] - pos[1]
        self._prev_dist = math.sqrt(dx * dx + dy * dy)

        return self._get_obs(), {}

    # ─────────────────────────────────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action):
        self._step_count += 1
        truncated = self._step_count >= getattr(self, "_max_steps", 2500)

        velocidad_lineal  = float(action[0]) * self.MAX_LINEAR_SPEED
        velocidad_angular = float(action[1]) * self.MAX_ANGULAR_SPEED

        lidar_raw    = list(self.lidar.getRangeImage())
        lidar_finite = [v for v in lidar_raw if math.isfinite(v)]
        min_dist     = min(lidar_finite) if lidar_finite else 999.0

        # shield: reducir velocidad solo si avanza hacia el obstáculo
        # no aplicar a marcha atrás para permitir maniobras de escape
        if min_dist < 1.5 and velocidad_lineal > 0:
            velocidad_lineal *= min_dist / 1.5

        left_speed  = (velocidad_lineal - velocidad_angular * self.AXLE_LENGTH / 2) / self.WHEEL_RADIUS
        right_speed = (velocidad_lineal + velocidad_angular * self.AXLE_LENGTH / 2) / self.WHEEL_RADIUS
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        self._current_lin_vel = velocidad_lineal
        self._current_ang_vel = velocidad_angular

        self.supervisor.step(self.timestep)

        recompensa = 0.0
        terminated = False
        info       = {}

        pos      = self.gps.getValues()
        rx, ry   = pos[0], pos[1]

        # STH-WP: d_ahead reducido durante _fase_escape para evitar subgoal tras esquina
        d_actual = self.D_AHEAD_ESCAPE if self._fase_escape else self.D_AHEAD_NORMAL
        subgoal  = compute_sth_subgoal(self.full_path, (rx, ry), d_actual)

        # Replanning dinámico: si un peatón intercepta el segmento robot→subgoal,
        # recalcular la ruta A* con el peatón como obstáculo temporal
        # (desactivable con enable_dynamic_replanning=False — condición E2.1-R0)
        if self.enable_dynamic_replanning:
            if self._replan_cooldown == 0:
                # E2.2: replanning basado en LIDAR (sin supervisor)
                if self._try_replan_lidar(rx, ry, lidar_raw, subgoal[0], subgoal[1]):
                    subgoal = compute_sth_subgoal(self.full_path, (rx, ry), d_actual)
                    self._replan_cooldown = self.REPLAN_COOLDOWN
            if self._replan_cooldown > 0:
                self._replan_cooldown -= 1

        self.goal = np.array(subgoal, dtype=np.float32)
        dx        = self.goal[0] - rx
        dy        = self.goal[1] - ry
        dist_actual = math.sqrt(dx * dx + dy * dy)

        # ── colisión ─────────────────────────────────────────────────────────
        if self.bumper.getValue() > 0:
            recompensa -= 150.0
            terminated  = True
            info["colision"]         = True
            info["llego_estanteria"] = self._hacia_descarga or self._en_retorno
            print("[COLISIÓN]")

        else:
            # progreso hacia el subgoal
            recompensa += (self._prev_dist - dist_actual) * 3.0

            # proximidad continua (exponencial) — obstáculos estáticos/LIDAR
            if min_dist < 1.5:
                recompensa -= 1.5 * math.exp(-2.5 * min_dist)

            # penalización por proximidad específica a peatones (B1)
            # usa la posición real del nodo (no LIDAR) → señal diferenciada del obstáculo estático
            # solo activa cuando ped_obs=True; sin efecto en entrenamiento sin peatones
            if self._ped_obs and self._ped_nodes:
                for ped_node in self._ped_nodes:
                    p = ped_node.getField("translation").getSFVec3f()
                    dist_ped = math.sqrt((rx - p[0]) ** 2 + (ry - p[1]) ** 2)
                    if dist_ped < 4.0:
                        # E2.1: umbral ampliado a 4m para coherencia con LIDAR 5m
                        recompensa -= 1.5 * math.exp(-1.5 * dist_ped)

            # penalización angular base
            recompensa -= 0.05 * abs(velocidad_angular)

            # _fase_escape: solo gestiona el contador; d_ahead se reduce en _compute_subgoal
            # nuevo almacén: salida en espacio abierto → sin penalizaciones extra ni bonus marcha atrás
            if self._fase_escape:
                self._escape_steps += 1
                if self._escape_steps >= self.ESCAPE_DURATION:
                    self._fase_escape = False

            # bonus de alineación movimiento-subgoal
            # Premia que el vector de movimiento apunte hacia el subgoal,
            # ya sea avanzando (vel>0) o retrocediendo (vel<0).
            # Fórmula: sign(vel_lineal) * cos(angulo_rel)
            #   · Avanzar hacia subgoal:   sign=+1, cos≈+1  → +0.15 ✓
            #   · Retroceder hacia subgoal: sign=-1, cos≈-1  → +0.15 ✓  (marcha atrás útil)
            #   · Avanzar alejándose:       sign=+1, cos≈-1  → -0.15 ✓
            #   · Retroceder alejándose:    sign=-1, cos≈+1  → -0.15 ✓
            compass_val   = self.compass.getValues()
            robot_head    = math.atan2(compass_val[0], compass_val[1])
            angle_to_goal = math.atan2(self.goal[1] - ry, self.goal[0] - rx)
            angulo_rel    = (angle_to_goal - robot_head + math.pi) % (2 * math.pi) - math.pi
            vel_sign      = math.copysign(1.0, velocidad_lineal) if abs(velocidad_lineal) > 0.02 else 1.0
            recompensa   += 0.15 * vel_sign * math.cos(angulo_rel)

            # E1.5: exit-corridor reward — esperar cuando P1 está en el cono de salida
            # Solo activo durante la fase de exit (hacia_descarga o fase_escape inicial)
            # Detecta P1 dentro de 2.5m en cono de 45° delante del robot
            if self._ped_obs and self._ped_nodes and (self._hacia_descarga or self._fase_escape):
                for ped_node in self._ped_nodes:
                    p        = ped_node.getField("translation").getSFVec3f()
                    px_rel   = p[0] - rx
                    py_rel   = p[1] - ry
                    dist_ped = math.sqrt(px_rel ** 2 + py_rel ** 2)
                    if dist_ped < 4.0:
                        angle_to_ped = math.atan2(py_rel, px_rel)
                        angle_diff   = abs((angle_to_ped - robot_head + math.pi) % (2 * math.pi) - math.pi)
                        if angle_diff < math.pi / 4 and abs(velocidad_lineal) < 0.05:
                            recompensa += 0.6

            # ── comprobación de llegada ───────────────────────────────────────
            gfx, gfy   = self.full_path[-1]
            dist_final = math.sqrt((rx - gfx) ** 2 + (ry - gfy) ** 2)

            if dist_final < 0.5:
                if self._en_retorno:
                    # llegó a zona_espera = éxito retorno (stage 5)
                    recompensa += 100.0
                    terminated  = True
                    info["exito"]        = True
                    info["llego_espera"] = True
                    print("[ÉXITO RETORNO] zona de espera alcanzada")

                elif not self._hacia_descarga:
                    # llegó a la estantería
                    if self.stage <= 2:
                        # stages 1-2: approach completo = éxito del episodio
                        recompensa += 50.0
                        terminated  = True
                        info["exito"]           = True
                        info["llego_estanteria"] = True
                        print(f"[ÉXITO APPROACH] {self.goal_ids[self.current_goal_idx]}")
                    else:
                        # stage 4: continuar hacia descarga
                        self._hacia_descarga = True
                        self._fase_escape    = True
                        self._escape_steps   = 0
                        recompensa          += 50.0
                        self.full_path = list(
                            self._astar_cache[("exit", self.current_goal_idx)]
                        )
                        subgoal   = compute_sth_subgoal(self.full_path, (rx, ry), self.D_AHEAD_ESCAPE)
                        self.goal = np.array(subgoal, dtype=np.float32)
                        dx        = self.goal[0] - rx
                        dy        = self.goal[1] - ry
                        dist_actual = math.sqrt(dx * dx + dy * dy)
                        print(f"[ESTANTERÍA → DESCARGA] {self.goal_ids[self.current_goal_idx]}")
                else:
                    # llegó a la descarga
                    recompensa += 100.0
                    if self.stage == 6:
                        # ciclo completo: switch a retorno sin terminar el episodio
                        self._hacia_descarga = False
                        self._en_retorno     = True
                        self._fase_escape    = False
                        self._escape_steps   = 0
                        self.full_path = list(self._astar_cache[("return", 0)])
                        subgoal   = compute_sth_subgoal(self.full_path, (rx, ry), self.D_AHEAD_NORMAL)
                        self.goal = np.array(subgoal, dtype=np.float32)
                        dx        = self.goal[0] - rx
                        dy        = self.goal[1] - ry
                        dist_actual = math.sqrt(dx * dx + dy * dy)
                        print(f"[DESCARGA → RETORNO] {self.goal_ids[self.current_goal_idx]}")
                    else:
                        terminated  = True
                        info["exito"]           = True
                        info["llego_estanteria"] = True
                        print(f"[ÉXITO COMPLETO] {self.goal_ids[self.current_goal_idx]}")

        self._prev_dist = dist_actual
        recompensa -= 0.001

        if truncated and not terminated:
            recompensa -= 20.0
            info["truncado"]         = True
            info["llego_estanteria"] = self._hacia_descarga or self._en_retorno

        return self._get_obs(), recompensa, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # Replanning dinámico mid-episode
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dist_perp(px, py, ax, ay, bx, by):
        """Distancia perpendicular de (px,py) al segmento (ax,ay)→(bx,by)."""
        dx, dy = bx - ax, by - ay
        len2 = dx * dx + dy * dy
        if len2 < 1e-9:
            return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
        cx, cy = ax + t * dx, ay + t * dy
        return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)

    def _try_replan(self, rx, ry, sx, sy):
        """Devuelve True y actualiza full_path si algún peatón bloquea robot→subgoal."""
        if not self.enable_dynamic_replanning:
            raise RuntimeError(
                "_try_replan() invocado con enable_dynamic_replanning=False. "
                "Este método no debería llamarse nunca en la condición E2.1-R0."
            )
        for node in self._ped_nodes:
            p = node.getField("translation").getSFVec3f()
            if self._dist_perp(p[0], p[1], rx, ry, sx, sy) < self.REPLAN_DIST_PERP:
                return self._replanificar(rx, ry)
        return False

    def _replanificar(self, rx, ry):
        """Añade los peatones (inflados) a la cuadrícula y recalcula ruta A* mid-episode."""
        if not self.enable_dynamic_replanning:
            raise RuntimeError(
                "_replanificar() invocado con enable_dynamic_replanning=False. "
                "Este método no debería llamarse nunca en la condición E2.1-R0."
            )
        goal_pos   = self.full_path[-1]
        origen     = self._map_info["origin"]
        resolution = self._map_info["resolution"]
        rows, cols = self._grid_nav.shape

        grid_tmp = self._grid_nav.copy()
        for node in self._ped_nodes:
            p = node.getField("translation").getSFVec3f()
            pr, pc = mundo_a_celdas(p[0], p[1], origen, resolution)
            for dr in range(-self.REPLAN_INFLATE_CELLS, self.REPLAN_INFLATE_CELLS + 1):
                for dc in range(-self.REPLAN_INFLATE_CELLS, self.REPLAN_INFLATE_CELLS + 1):
                    nr, nc = pr + dr, pc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        grid_tmp[nr, nc] = 1

        start_cell = mundo_a_celdas(rx, ry, origen, resolution)
        goal_cell  = mundo_a_celdas(goal_pos[0], goal_pos[1], origen, resolution)
        if grid_tmp[start_cell[0]][start_cell[1]] == 1:
            start_cell = find_nearest_free(grid_tmp, start_cell)
        if grid_tmp[goal_cell[0]][goal_cell[1]] == 1:
            goal_cell = find_nearest_free(grid_tmp, goal_cell)
        if start_cell is None or goal_cell is None:
            return False

        path_cells = astar(grid_tmp, start_cell, goal_cell)
        if path_cells is None:
            return False

        path_world = [celdas_a_mundo(r, c, origen, resolution) for r, c in path_cells]
        if not path_world:
            return False
        path_world[-1] = goal_pos
        self.full_path = path_world
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Replanning dinámico basado en LIDAR (E2.2) — sin supervisor
    # ─────────────────────────────────────────────────────────────────────────

    def _lidar_to_obstacles(self, rx, ry, lidar_raw):
        """Proyecta rayos LIDAR a coordenadas mundo. Solo incluye lecturas en el
        rango dinámico [LIDAR_MIN_DYNAMIC, LIDAR_DYN_MAX] cuya celda en el mapa
        estático esté libre (celda==0) — filtra paredes ya conocidas."""
        compass_val = self.compass.getValues()
        robot_head  = math.atan2(compass_val[0], compass_val[1])
        n           = len(lidar_raw)
        angle_step  = 2 * math.pi / n
        origen      = self._map_info["origin"]
        resolution  = self._map_info["resolution"]
        rows, cols  = self._grid_nav.shape
        obstacles   = []
        for i, d in enumerate(lidar_raw):
            if not math.isfinite(d):
                continue
            if self.LIDAR_MIN_DYNAMIC <= d <= self.LIDAR_DYN_MAX:
                ray_angle = robot_head + i * angle_step
                ox = rx + d * math.cos(ray_angle)
                oy = ry + d * math.sin(ray_angle)
                pr, pc = mundo_a_celdas(ox, oy, origen, resolution)
                if 0 <= pr < rows and 0 <= pc < cols and self._grid_nav[pr][pc] == 0:
                    obstacles.append((ox, oy))
        return obstacles

    def _try_replan_lidar(self, rx, ry, lidar_raw, sx, sy):
        """Versión LIDAR de _try_replan. Devuelve True si algún obstáculo dinámico
        bloquea el segmento robot→subgoal y se ha replanificado con éxito."""
        if not self.enable_dynamic_replanning:
            raise RuntimeError(
                "_try_replan_lidar() invocado con enable_dynamic_replanning=False. "
                "Este método no debería llamarse nunca en la condición E2.1-R0."
            )
        obstacles = self._lidar_to_obstacles(rx, ry, lidar_raw)
        for (ox, oy) in obstacles:
            if self._dist_perp(ox, oy, rx, ry, sx, sy) < self.REPLAN_DIST_PERP:
                self._replan_attempts += 1
                if self._replanificar_lidar(rx, ry, obstacles):
                    self._replan_successes += 1
                    return True
                return False
        return False

    def _replanificar_lidar(self, rx, ry, obstacles):
        """Infla las posiciones LIDAR en el grid y recalcula ruta A*."""
        if not self.enable_dynamic_replanning:
            raise RuntimeError(
                "_replanificar_lidar() invocado con enable_dynamic_replanning=False. "
                "Este método no debería llamarse nunca en la condición E2.1-R0."
            )
        goal_pos   = self.full_path[-1]
        origen     = self._map_info["origin"]
        resolution = self._map_info["resolution"]
        rows, cols = self._grid_nav.shape

        grid_tmp = self._grid_nav.copy()
        for (ox, oy) in obstacles:
            pr, pc = mundo_a_celdas(ox, oy, origen, resolution)
            for dr in range(-self.REPLAN_INFLATE_CELLS, self.REPLAN_INFLATE_CELLS + 1):
                for dc in range(-self.REPLAN_INFLATE_CELLS, self.REPLAN_INFLATE_CELLS + 1):
                    nr, nc = pr + dr, pc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        grid_tmp[nr, nc] = 1

        start_cell = mundo_a_celdas(rx, ry, origen, resolution)
        goal_cell  = mundo_a_celdas(goal_pos[0], goal_pos[1], origen, resolution)
        if grid_tmp[start_cell[0]][start_cell[1]] == 1:
            start_cell = find_nearest_free(grid_tmp, start_cell)
        if grid_tmp[goal_cell[0]][goal_cell[1]] == 1:
            goal_cell  = find_nearest_free(grid_tmp, goal_cell)
        if start_cell is None or goal_cell is None:
            return False

        path_cells = astar(grid_tmp, start_cell, goal_cell)
        if path_cells is None:
            return False

        path_world = [celdas_a_mundo(r, c, origen, resolution) for r, c in path_cells]
        if not path_world:
            return False
        path_world[-1] = goal_pos
        self.full_path = path_world
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Observación
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        lidar = self.lidar.getRangeImage()
        lidar_norm = [
            min(v if math.isfinite(v) else self.MAX_LIDAR_RANGE, self.MAX_LIDAR_RANGE)
            / self.MAX_LIDAR_RANGE
            for v in lidar
        ]

        pos     = self.gps.getValues()
        robot_x = pos[0]
        robot_y = pos[1]

        dx = self.goal[0] - robot_x
        dy = self.goal[1] - robot_y
        dist_to_goal = math.sqrt(dx * dx + dy * dy)

        compass_val   = self.compass.getValues()
        robot_head    = math.atan2(compass_val[0], compass_val[1])
        angle_to_goal = math.atan2(dy, dx)
        angulo_rel    = (angle_to_goal - robot_head + math.pi) % (2 * math.pi) - math.pi

        dist_norm  = min(dist_to_goal, self.MAX_GOAL_DIST) / self.MAX_GOAL_DIST
        angle_norm = angulo_rel / math.pi
        vlin_norm  = self._current_lin_vel / self.MAX_LINEAR_SPEED
        vang_norm  = self._current_ang_vel / self.MAX_ANGULAR_SPEED

        base_obs = lidar_norm + [dist_norm, angle_norm, vlin_norm, vang_norm]

        return np.array(base_obs, dtype=np.float32)

    def render(self):
        pass

    def close(self):
        pass
