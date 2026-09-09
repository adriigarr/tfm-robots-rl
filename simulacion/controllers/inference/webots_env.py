# conectar webots con SB3
# SB3 <---> WebotsEnnv <---> Webots

# librerias
import json
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor
from global_planner import build_grid, plan_path


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

    
    def __init__(self, map_path = 'warehouse_map.json', render_mode = None):
        super().__init__()
        
        self.render_mode = render_mode
        
        self.supervisor = Supervisor()
        self.timestep = self.TIMESTEP_MS
        
        # cargar el mapa del almacén
        with open(map_path, "r") as f:
            warehouse = json.load(f)
        
        # obtener las referencias del json
        
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
        
        #espacio de acciones
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
        
        # precargar el grid una sola vez (costoso)
        self.grid, self.map_info = build_grid(map_path)
        self.waypoints = []
        self.wp_idx = 0
        
        self.map_path = map_path
        
        self._step_count = 0
    
    def reset(self, seed=None, options=None):
        # reinicia el episodio
        # devuelve al robot a su posicion inicial
        # elige un nuevo goal aleatorio
        # devuelve (obs, info)
        super().reset(seed=seed)
        self._step_count = 0
        
        # parar los motores
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # teletransportar el robot a su posición inicial
        translation = self.robot_node.getField("translation")
        rotation    = self.robot_node.getField("rotation")

        translation.setSFVec3f([self._start_x, self._start_y, 0.2])
        rotation.setSFRotation([0, 1, 0, 0])

        # resetear física: velocidades, fuerzas acumuladas, inercia
        self.supervisor.simulationResetPhysics()

        # avanzar un paso para que los sensores tengan valores válidos
        self.supervisor.step(self.timestep)
        
        # elegir goal aleatorio
        self.current_goal_idx = self.np_random.integers(0, len(self.goals))
        goal_final = self.goals[self.current_goal_idx]

        # calcular ruta con A*
        pos = self.gps.getValues()
        start_pos = (pos[0], pos[1])
        self.waypoints = plan_path(self.map_path, start_pos, goal_final)
        self.wp_idx = 0

        # si A* no encuentra ruta, usar goal directo como fallback
        if self.waypoints is None:
            self.waypoints = [goal_final]

        # el goal activo es el primer waypoint
        self.goal = np.array(self.waypoints[0], dtype=np.float32)

        goal_id = self.goal_ids[self.current_goal_idx]
        print(f"[EPISODIO] Goal: {goal_id} → {len(self.waypoints)} waypoints")
        
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
        # recibe la accion del agente
        # avanza la simulación un paso
        # calcular la recompensa
        # devuelve (obs, reward, terminated, truncated, info)
        
        self._step_count += 1
        truncated = self._step_count >= 2500
        
        #convertir la accion a velocidades y mver el robot
        velocidad_lineal = float(action[0]) * self.MAX_LINEAR_SPEED
        velocidad_angular = float(action[1]) * self.MAX_ANGULAR_SPEED
        
        # ── velocidad adaptativa por proximidad ────────────────────
        lidar_raw = list(self.lidar.getRangeImage())
        lidar_finite = [v for v in lidar_raw if math.isfinite(v)]
        min_dist = min(lidar_finite) if lidar_finite else 999.0

        if min_dist < 1.5:
            factor = min_dist / 1.5  # 0.0 (pegado) → 1.0 (lejos)
            velocidad_lineal *= factor
        
        left_speed = (velocidad_lineal - velocidad_angular * self.AXLE_LENGTH / 2) / self.WHEEL_RADIUS
        right_speed = (velocidad_lineal + velocidad_angular * self.AXLE_LENGTH / 2) / self.WHEEL_RADIUS
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        
        self._current_lin_vel = velocidad_lineal
        self._current_ang_vel = velocidad_angular
        
        # avanzar la simulación
        self.supervisor.step(self.timestep)
        
        # calcular la recompensa
        recompensa = 0.0
        terminated = False
        info = {}
        
            # progreso hacia el goal
        pos = self.gps.getValues()
        dx = self.goal[0] - pos[0]
        dy = self.goal[1] - pos[1]
        dist_actual = math.sqrt(dx*dx + dy*dy)
        
        recompensa = (self._prev_dist - dist_actual) * 5.0
        self._prev_dist = dist_actual
        
            # añadir la penalización suave por proximidad
        lidar_raw = list(self.lidar.getRangeImage())
        lidar_finite = [v for v in lidar_raw if math.isfinite(v)]
        
            # penalización por colisiones
        if self.bumper.getValue() > 0:
            recompensa -= 10.0
            terminated = True
            info["colision"] = True
            print("[COLISIÓN] Contacto físico detectado")
            
            # recompensa por alcanzar el goal
        elif dist_actual < 0.5:
            self.wp_idx += 1
            if self.wp_idx >= len(self.waypoints):
                # llegó al goal final
                recompensa += 15.0
                terminated = True
                info["exito"] = True
                print(f"[ÉXITO] Robot llegó al goal final")
            else:
                # avanza al siguiente waypoint
                self.goal = np.array(self.waypoints[self.wp_idx], dtype=np.float32)
                recompensa += 2.0
                pos_actual = self.gps.getValues()
                dx = self.goal[0] - pos_actual[0]
                dy = self.goal[1] - pos_actual[1]
                self._prev_dist = math.sqrt(dx*dx + dy*dy)
                print(f"[WP] Waypoint {self.wp_idx}/{len(self.waypoints)-1}")
        elif lidar_finite and min(lidar_finite) < 1.0:
            recompensa -= 0.2

            # penalización por paso
        recompensa -= 0.005
        
        # construir la observación
        obs = self._get_obs()
        return obs, recompensa, terminated, truncated, info
    
    def _get_obs(self):
        # lee los sensores
        # construye el vector de observación
        # np.array de shape (40,) con dtype float32
        
        # lidar (normalizado con la formula de max range)
        lidar = self.lidar.getRangeImage()

        lidar_norm = [
            min(v, self.MAX_LIDAR_RANGE) / self.MAX_LIDAR_RANGE
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
        
        # concatenar y devolver np.array de shape (40)
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