# inference.py
# Script de inferencia — ejecuta el modelo entrenado en todos los goals
# sin entrenamiento, solo evaluación

import sys
import time
import numpy as np

sys.path.insert(0, '../rl_train')

from webots_env import WebotsEnv
from stable_baselines3 import PPO
from global_planner import plan_path

# ── configuración ────────────────────────────────────────────────────────────
MAP_PATH   = "warehouse_map01.json"
MODEL_PATH = "ppo_warehouse_10"   # sin .zip
PAUSA_ENTRE_GOALS = 1.5           # segundos entre goals
MAX_STEPS  = 3000                 # límite de steps por goal
# ─────────────────────────────────────────────────────────────────────────────

# cargar entorno y modelo
env   = WebotsEnv(map_path=MAP_PATH)
model = PPO.load(MODEL_PATH, env=env)

print(f"\n{'='*50}")
print(f"  INFERENCIA — {len(env.goals)} goals")
print(f"  Modelo: {MODEL_PATH}")
print(f"{'='*50}\n")

resultados = []

for goal_idx in range(len(env.goals)):
    goal_id    = env.goal_ids[goal_idx]
    goal_final = env.goals[goal_idx]

    # resetear entorno
    obs, _ = env.reset()

    # forzar el goal específico y recalcular ruta A*
    env.current_goal_idx = goal_idx
    pos = env.gps.getValues()
    start_pos = (pos[0], pos[1])

    env.waypoints = plan_path(MAP_PATH, start_pos, goal_final)
    if env.waypoints is None:
        env.waypoints = [goal_final]
    env.wp_idx = 0
    env.goal   = np.array(env.waypoints[0], dtype=np.float32)

    # recalcular distancia inicial
    dx = env.goal[0] - pos[0]
    dy = env.goal[1] - pos[1]
    import math
    env._prev_dist = math.sqrt(dx*dx + dy*dy)

    print(f"[{goal_idx+1:02d}/28] {goal_id} → {len(env.waypoints)} waypoints", end=" ... ", flush=True)

    terminated = False
    truncated  = False
    steps      = 0
    info       = {}

    while not terminated and not truncated and steps < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    if info.get("exito"):
        estado = "ÉXITO"
        simbolo = "✓"
    elif info.get("colision"):
        estado = "COLISIÓN"
        simbolo = "✗"
    else:
        estado = "TRUNCADO"
        simbolo = "~"

    print(f"{simbolo} {estado} ({steps} steps)")
    resultados.append((goal_id, simbolo, estado, steps))

    time.sleep(PAUSA_ENTRE_GOALS)

# ── resumen final ─────────────────────────────────────────────────────────────
exitos    = sum(1 for _, s, _, _ in resultados if s == "✓")
colisiones = sum(1 for _, s, _, _ in resultados if s == "✗")
truncados  = sum(1 for _, s, _, _ in resultados if s == "~")

print(f"\n{'='*50}")
print(f"  RESUMEN FINAL")
print(f"{'='*50}")
for goal_id, simbolo, estado, steps in resultados:
    print(f"  {simbolo} {goal_id}: {estado} ({steps} steps)")

print(f"\n  Éxitos:     {exitos}/28  ({exitos/28*100:.1f}%)")
print(f"  Colisiones: {colisiones}/28")
print(f"  Truncados:  {truncados}/28")
print(f"{'='*50}\n")

env.supervisor.simulationSetMode(env.supervisor.SIMULATION_MODE_PAUSE)