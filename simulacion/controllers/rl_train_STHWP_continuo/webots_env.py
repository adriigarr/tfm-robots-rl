# conectar webots con SB3
# SB3 <---> WebotsEnnv <---> Webots

# librerias
import json
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor
from global_planner import build_grid, plan_path, compute_sth_subgoal


# se crea la clase WebotsEnv que hereda de gym.Env
class WebotsEnv(gym.Env):
    metadata = {'render_modes': []}
    
    # Parametros físicos del robot
    MAX_LINEAR_SPEED  = 0.5
    MAX_ANGULAR_SPEED = 1.5
    WHEEL_RADIUS      = 0.0625
    AXLE_LENGTH       = 0.445208
    LIDAR_RAYS        = 36
    MAX_LIDAR_RANGE   = 3.5
    MAX_GOAL_DIST     = 15.0
    TIMESTEP_MS       = 32

    # curriculum learning: probabilidad de empezar desde estantería → descarga
    PROB_CURRICULUM   = 0.6

    # índices (0-based) de las estanterías con goal_y = 1.5 o posiciones
    # interiores difíciles — son las que fallan en la fase estantería → descarga
    # goal_05→idx4, goal_09→idx8, goal_10→idx9, goal_13→idx12,
    # goal_17→idx16, goal_18→idx17, goal_19→idx18,
    # goal_23→idx22, goal_24→idx23, goal_25→idx24
    SHELVES_DIFICILES = {5, 9, 10, 13, 17, 18, 19, 23, 24, 25}

    # probabilidad de que un episodio curriculum use una estantería difícil
    PROB_SHELF_DIFICIL = 0.7

    
    def __init__(self, map_path = 'warehouse_map.json', render_mode = None):
        super().__init__()
        
        self.render_mode = render_mode
        
        self.supervisor = Supervisor()
        self.timestep = self.TIMESTEP_MS
        
        # cargar el mapa del almacén
        with open(map_path, "r") as f:
            warehouse = json.load(f)
        
        # extraer los goals de las estanterias
        self.goals = [
            (shelf["goal_x"], shelf["goal_y"])
            for shelf in warehouse["shelves"]
        ]
        
        self.goal_ids = [
            shelf["goal_id"]
            for shelf in warehouse["shelves"]
        ]
        
        # extraer la posicion de la zona de espera
        self.zona_espera = (
            warehouse["zones"]["waiting"]["x"],
            warehouse["zones"]["waiting"]["y"]
        )
        
        # extraer la posición de la zona de descarga
        self.zona_descarga = (
            warehouse["zones"]["dropoff"]["x"],
            warehouse["zones"]["dropoff"]["y"]
        )
        
        # extraer los waypoints del pasillo de retorno
        self.waypoints_pasillo = [
            (wp["x"], wp["y"])
            for wp in warehouse["return_corridor"]["waypoints"]
        ]
        
        # Goal activo (empieza en la primera estantería)
        self.current_goal_idx = 0
        self.goal = np.array(self.goals[0], dtype=np.float32)
        
        # nodo del robot
        self.robot_node = self.supervisor.getFromDef("MIR")
        if self.robot_node is None:
            raise RuntimeError(
                "No se encontró el nodo DEF = ROBOT en Webots"
                "Asegurate de que el robot tiene ese DEF en el fichero .wbt"
            )
            
        # posicion inicial del robot
        pos = self.robot_node.getField("translation").getSFVec3f()
        self._start_x = pos[0]
        self._start_y = pos[1]
        
        # Motores
        self.left_motor = self.supervisor.getDevice("middle_left_wheel_joint")
        self.right_motor = self.supervisor.getDevice("middle_right_wheel_joint")
        
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # LIDAR
        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)
        
        # GPS
        self.gps = self.supervisor.getDevice("gps")
        self.gps.enable(self.timestep)
        
        # compass
        self.compass = self.supervisor.getDevice("compass")
        self.compass.enable(self.timestep)
        
        # bumper
        self.bumper = self.supervisor.getDevice("bumper")
        self.bumper.enable(self.timestep)
        
        # espacio de observaciones
            # 36 valores del lidar
            # distancia al goal
            # angulo al goal
            # velocidad lineal
            # velocidad angular
        obs_size = self.LIDAR_RAYS + 4
        self.observation_space = spaces.Box(
            low = np.full(obs_size, -1.0, dtype=np.float32),
            high = np.full(obs_size, 1.0, dtype=np.float32),
            dtype = np.float32
        )
        
        # espacio de acciones
            # 2 números continuos entre -1 y 1
            # [0] -> velocidad lineal
            # [1] -> velocidad angular
        self.action_space = spaces.Box(
            low = np.array([-1.0, -1.0], dtype=np.float32),
            high = np.array([1.0, 1.0], dtype=np.float32),
            dtype = np.float32
        )
        
        # estado interno del episodio
        self._current_lin_vel = 0.0
        self._current_ang_vel = 0.0
        self._prev_dist = None
        self._hacia_descarga = False
        
        # precargar el grid una sola vez (costoso)
        self.grid, self.map_info = build_grid(map_path)
        self.full_path = []   # ruta A* completa (sin subsamplear)
        self.d_ahead   = 1.5  # radio STH-WP en metros
        
        self.map_path = map_path
        
        self._step_count = 0
    
    def reset(self, seed=None, options=None):
        # reinicia el episodio
        # con probabilidad PROB_CURRICULUM el robot empieza directamente en una
        # estantería y tiene que llegar a la zona de descarga (curriculum)
        #   · con probabilidad PROB_SHELF_DIFICIL elige una estantería difícil
        #   · el resto elige aleatoriamente
        # el resto de episodios sigue el flujo normal: espera → estantería aleatoria
        # devuelve (obs, info)
        super().reset(seed=seed)
        self._step_count     = 0
        self._hacia_descarga = False
        
        # parar los motores
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        translation = self.robot_node.getField("translation")
        rotation    = self.robot_node.getField("rotation")

        if self.np_random.random() < self.PROB_CURRICULUM:
            # ── modo curriculum: teletransportar a la estantería → descarga ──────
            self._hacia_descarga = True

            # elegir estantería: difícil o aleatoria según probabilidad
            if self.np_random.random() < self.PROB_SHELF_DIFICIL:
                idx_lista = list(self.SHELVES_DIFICILES)
                self.current_goal_idx = idx_lista[
                    self.np_random.integers(0, len(idx_lista))
                ]
            else:
                self.current_goal_idx = self.np_random.integers(0, len(self.goals))

            goal_id     = self.goal_ids[self.current_goal_idx]
            start_shelf = self.goals[self.current_goal_idx]
            translation.setSFVec3f([start_shelf[0], start_shelf[1], 0.2])
            rotation.setSFRotation([0, 1, 0, 0])
            self.supervisor.simulationResetPhysics()
            self.supervisor.step(self.timestep)

            pos = self.gps.getValues()
            start_pos = (pos[0], pos[1])
            self.full_path = plan_path(self.map_path, start_pos, self.zona_descarga, subsample=False)
            if self.full_path is None:
                self.full_path = [self.zona_descarga]
            print(f"[CURRICULUM] {goal_id} → descarga ({len(self.full_path)} puntos)")

        else:
            # ── modo normal: teletransportar a zona de espera → estantería ───────
            self.current_goal_idx = self.np_random.integers(0, len(self.goals))
            goal_id    = self.goal_ids[self.current_goal_idx]
            goal_final = self.goals[self.current_goal_idx]

            translation.setSFVec3f([self._start_x, self._start_y, 0.2])
            rotation.setSFRotation([0, 1, 0, 0])
            self.supervisor.simulationResetPhysics()
            self.supervisor.step(self.timestep)

            pos = self.gps.getValues()
            start_pos = (pos[0], pos[1])
            self.full_path = plan_path(self.map_path, start_pos, goal_final, subsample=False)
            if self.full_path is None:
                self.full_path = [goal_final]
            print(f"[EPISODIO] espera → {goal_id} ({len(self.full_path)} puntos ruta STH-WP)")

        # subgoal inicial con STH-WP
        pos = self.gps.getValues()
        subgoal = compute_sth_subgoal(self.full_path, (pos[0], pos[1]), self.d_ahead)
        self.goal = np.array(subgoal, dtype=np.float32)
        
        # resetear el estado interno
        self._current_lin_vel = 0.0
        self._current_ang_vel = 0.0
        
        # calcular distancia inicial al goal
        pos = self.gps.getValues()
        dx = self.goal[0] - pos[0]
        dy = self.goal[1] - pos[1]
        self._prev_dist = math.sqrt(dx*dx + dy*dy)
        
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        self._step_count += 1
        truncated = self._step_count >= 2500

        velocidad_lineal  = float(action[0]) * self.MAX_LINEAR_SPEED
        velocidad_angular = float(action[1]) * self.MAX_ANGULAR_SPEED

        lidar_raw    = list(self.lidar.getRangeImage())
        lidar_finite = [v for v in lidar_raw if math.isfinite(v)]
        min_dist     = min(lidar_finite) if lidar_finite else 999.0

        # velocidad adaptativa por proximidad
        if min_dist < 1.5:
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

        # ── posición actual ───────────────────────────────────────────────────────
        pos = self.gps.getValues()
        rx, ry = pos[0], pos[1]

        # ── actualizar subgoal STH-WP ─────────────────────────────────────────────
        subgoal    = compute_sth_subgoal(self.full_path, (rx, ry), self.d_ahead)
        self.goal  = np.array(subgoal, dtype=np.float32)
        dx         = self.goal[0] - rx
        dy         = self.goal[1] - ry
        dist_actual = math.sqrt(dx*dx + dy*dy)

        # ── colisión ──────────────────────────────────────────────────────────────
        if self.bumper.getValue() > 0:
            recompensa -= 150.0
            terminated  = True
            info["colision"] = True
            info["llego_estanteria"] = self._hacia_descarga
            print("[COLISIÓN] Contacto físico detectado")

        else:
            # ── progreso hacia el subgoal ─────────────────────────────────────────
            recompensa += (self._prev_dist - dist_actual) * 3.0

            # ── proximidad continua (exponencial) ────────────────────────────────
            if min_dist < 1.5:
                recompensa -= 1.5 * math.exp(-2.5 * min_dist)

            # ── penalización por velocidad angular excesiva ───────────────────────
            recompensa -= 0.05 * abs(velocidad_angular)

            # ── bonus por orientarse hacia el goal ───────────────────────────────
            compass_val   = self.compass.getValues()
            robot_head    = math.atan2(compass_val[0], compass_val[1])
            angle_to_goal = math.atan2(self.goal[1] - ry, self.goal[0] - rx)
            angulo_rel    = (angle_to_goal - robot_head + math.pi) % (2 * math.pi) - math.pi
            recompensa   += 0.15 * math.cos(angulo_rel)

            # ── llegada al GOAL FINAL ─────────────────────────────────────────────
            gfx, gfy   = self.full_path[-1]
            dist_final = math.sqrt((rx - gfx)**2 + (ry - gfy)**2)

            if dist_final < 0.5:
                if not self._hacia_descarga:
                    # ── llegó a la estantería → replanificar hacia descarga ───────
                    self._hacia_descarga = True
                    recompensa += 50.0
                    pos_actual = self.gps.getValues()
                    self.full_path = plan_path(
                        self.map_path,
                        (pos_actual[0], pos_actual[1]),
                        self.zona_descarga,
                        subsample=False
                    )
                    if self.full_path is None:
                        self.full_path = [self.zona_descarga]
                    subgoal    = compute_sth_subgoal(self.full_path, (pos_actual[0], pos_actual[1]), self.d_ahead)
                    self.goal  = np.array(subgoal, dtype=np.float32)
                    dx         = self.goal[0] - pos_actual[0]
                    dy         = self.goal[1] - pos_actual[1]
                    dist_actual = math.sqrt(dx*dx + dy*dy)
                    print("[ESTANTERÍA] Iniciando navegación a descarga")
                else:
                    # ── llegó a la descarga → éxito ───────────────────────────────
                    recompensa += 100.0
                    terminated  = True
                    info["exito"] = True
                    info["llego_estanteria"] = True
                    print(f"[ÉXITO] Robot llegó a la zona de descarga")

        # ── actualizar distancia previa (siempre, al final) ───────────────────────
        self._prev_dist = dist_actual

        # ── penalización por paso ─────────────────────────────────────────────────
        recompensa -= 0.001

        # ── penalización por truncado ─────────────────────────────────────────────
        if truncated and not terminated:
            recompensa -= 20.0
            info["truncado"] = True
            info["llego_estanteria"] = self._hacia_descarga

        obs = self._get_obs()
        return obs, recompensa, terminated, truncated, info
    
    def _get_obs(self):
        # lee los sensores
        # construye el vector de observación
        # np.array de shape (26,) con dtype float32
        
        # lidar (normalizado con la formula de max range)
        lidar = self.lidar.getRangeImage()
        # print(f"[DEBUG] lidar len: {len(lidar)}")
        
        lidar_norm = [
            min(v if math.isfinite(v) else self.MAX_LIDAR_RANGE, self.MAX_LIDAR_RANGE) / self.MAX_LIDAR_RANGE
            for v in lidar
        ]
        
        # posición y orientación del robot
        pos = self.gps.getValues()
        robot_x = pos[0]
        robot_y = pos[1]
        
        # distancia y ángulo al goal
        dx = self.goal[0] - robot_x
        dy = self.goal[1] - robot_y
        dist_to_goal = math.sqrt(dx*dx + dy*dy)
        
        compass_val = self.compass.getValues()
        robot_head = math.atan2(compass_val[0], compass_val[1])
        angle_to_goal = math.atan2(dy, dx) 
        angulo_relativo = angle_to_goal - robot_head
        angulo_relativo = (angulo_relativo + math.pi) % (2 * math.pi) - math.pi
        
        dist_norm = min(dist_to_goal, self.MAX_GOAL_DIST) / self.MAX_GOAL_DIST
        angle_norm = angulo_relativo / math.pi
        
        # velocidad lineal y angular
        velocidad_lineal_norm = self._current_lin_vel / self.MAX_LINEAR_SPEED
        velocidad_angular_norm = self._current_ang_vel / self.MAX_ANGULAR_SPEED
        
        # concatenar y devolver np.array de shape (26)
        return np.array(
            lidar_norm + [dist_norm, angle_norm, velocidad_lineal_norm, velocidad_angular_norm],
            dtype=np.float32
        )
    
    def render(self):
        # webots tiene su propia visualización
        pass
    
    def close(self):
        # cerrar la conexión con webots
        pass