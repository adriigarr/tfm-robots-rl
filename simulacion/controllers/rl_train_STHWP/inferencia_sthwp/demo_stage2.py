"""
Demo visual stage 2 — recorre los 28 goals en orden, 1 episodio por goal.

Lanzar Webots SIN --no-rendering para ver el robot moverse.
Al terminar los 28 goals Webots queda abierto para inspección.

Cambiar RUN_ID / MODEL_PATH para probar un seed distinto.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webots_env import WebotsEnv
from stable_baselines3 import PPO

# ── configuración ─────────────────────────────────────────────────────────────
MAP_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "warehouse_map01.json")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pruebas", "run001_s123_stage2_final")
RUN_ID     = "run001_s123"

# ── cargar entorno y modelo ───────────────────────────────────────────────────
env   = WebotsEnv(map_path=MAP_PATH, stage=2)
model = PPO.load(MODEL_PATH, env=env)

N_GOALS = len(env.goal_ids)

print(f"\n{'='*55}")
print(f" DEMO VISUAL stage 2 | {RUN_ID} | {N_GOALS} goals")
print(f"{'='*55}\n")

resultados = []

for goal_idx in range(N_GOALS):
    goal_id = env.goal_ids[goal_idx]
    env._sample_goal_approach = lambda gi=goal_idx: gi

    obs, _    = env.reset()
    done      = False
    truncated = False
    pasos     = 0
    reward_ep = 0.0

    print(f"[{goal_idx+1:>2}/{N_GOALS}] → {goal_id} ...", end="", flush=True)

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        pasos     += 1
        reward_ep += reward

    es_exito    = bool(info.get("exito",    False))
    es_colision = bool(info.get("colision", False))

    if es_exito:
        resultado = "ÉXITO    ✓"
    elif es_colision:
        resultado = "COLISIÓN ✗"
    else:
        resultado = "TRUNCADO ?"

    print(f"  {resultado}   pasos={pasos:<5}  reward={reward_ep:.1f}")
    resultados.append((goal_id, es_exito, es_colision))

# ── resumen ───────────────────────────────────────────────────────────────────
n_ok  = sum(1 for _, ok, _ in resultados if ok)
n_col = sum(1 for _, _, col in resultados if col)
n_tru = N_GOALS - n_ok - n_col

print(f"\n{'='*55}")
print(f" RESUMEN FINAL")
print(f"{'='*55}")
print(f"  Éxito:     {n_ok}/{N_GOALS}")
print(f"  Colisión:  {n_col}/{N_GOALS}")
print(f"  Truncado:  {n_tru}/{N_GOALS}")

if n_col > 0 or n_tru > 0:
    print(f"\n  Goals con problema:")
    for gid, ok, col in resultados:
        if not ok:
            tipo = "COLISIÓN" if col else "TRUNCADO"
            print(f"    {gid}: {tipo}")

print(f"\n  Webots queda abierto para inspección.")
print(f"{'='*55}\n")
