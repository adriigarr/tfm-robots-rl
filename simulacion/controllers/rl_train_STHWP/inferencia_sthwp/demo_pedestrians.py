"""
Demo de movimiento de peatones — robot estático, N pasos observación.

Propósito: visualizar la trayectoria de PEDESTRIAN_1 y PEDESTRIAN_2 en el mundo.
El robot no se mueve (velocidad = 0). Solo se avanza la simulación y se
imprimen las posiciones de cada peatón en cada step.

Lanzar Webots SIN --no-rendering para verlo visualmente:
  echo "demo_pedestrians" > rl_train_STHWP/current_stage.txt
  /Applications/Webots.app/Contents/MacOS/webots worlds/warehouse_1.wbt

Salida en consola: step | ped1=(x, y) | ped2=(x, y)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv

MAP_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "warehouse_map01.json")
N_STEPS    = 500   # ~500 × 64ms ≈ 32 s de simulación (una oscilación completa)

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.0, ped_obs=True)
env._max_steps = N_STEPS

# Detectar nodos de peatones (ya los carga WebotsEnv si ped_obs=True)
ped_nodes = env._ped_nodes
n_peds = len(ped_nodes)
print(f"\n{'='*60}")
print(f" Demo movimiento peatones | {n_peds} peatón(es) detectado(s)")
print(f" {N_STEPS} steps × {env.timestep}ms = {N_STEPS * env.timestep / 1000:.1f}s de simulación")
print(f"{'='*60}")
print(f"{'step':>5}  " + "  ".join(f"ped{i+1}=(   x,    y)" for i in range(n_peds)))
print("─" * 60)

# Reset: coloca peatones en posición aleatoria (comportamiento por defecto)
obs, _ = env.reset()

# Acción nula: robot completamente parado
zero_action = [0.0, 0.0]

for step in range(N_STEPS):
    # Leer posiciones antes del step
    positions = []
    for node in ped_nodes:
        p = node.getField("translation").getSFVec3f()
        positions.append((round(p[0], 2), round(p[1], 2)))

    if step % 10 == 0:   # imprimir cada 10 steps para no saturar la consola
        pos_str = "  ".join(f"({x:>5}, {y:>5})" for x, y in positions)
        print(f"  {step:>4}  {pos_str}")

    obs, reward, done, truncated, info = env.step(zero_action)
    if done or truncated:
        break

# Resumen final
print("\n  Posiciones finales:")
for i, node in enumerate(ped_nodes):
    p = node.getField("translation").getSFVec3f()
    print(f"    PEDESTRIAN_{i+1}: x={p[0]:.2f}  y={p[1]:.2f}  z={p[2]:.2f}")

print(f"\n{'='*60}")
print(f" Demo completada — {step+1} steps ejecutados")
print(f"{'='*60}\n")

env.supervisor.simulationQuit(0)
