"""
ETAPA 6 DIN v4 — Replanning integrado en entorno | SEED=123

Continúa desde run003_s123_stage6_din_v3_final (48 dims, ~6M steps previos).
Novedad respecto a v3: webots_env.py ejecuta replanificación A* dinámica mid-episode
cuando un peatón intercepta el segmento robot→subgoal (dist_perp < 0.6 m).
El agente aprende a navegar rutas recalculadas durante el entrenamiento, eliminando
el distribution shift que causó 0% de éxito en la replanificación externa (§25).

Parámetros de replanning (class constants en WebotsEnv):
  REPLAN_DIST_PERP     = 0.6 m
  REPLAN_INFLATE_CELLS = 3 celdas (0.75 m)
  REPLAN_COOLDOWN      = 40 pasos

World: warehouse_1.wbt (PEDESTRIAN_1 + PEDESTRIAN_2 activos)
Salida: pruebas/run003_s123_stage6_din_v4_final.zip
"""

import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH = "warehouse_map01.json"
LOAD_ID  = "run003_s123"
RUN_ID   = "run003_s123"
SEED     = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 7000

model = PPO.load(f"./pruebas/{LOAD_ID}_stage6_din_v3_final", env=env)
model.learning_rate = 5e-5
model.ent_coef      = 0.005

SAVE_PATH = f"./pruebas/{RUN_ID}_stage6_din_v4_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_stage6_din_v4"

model.learn(
    total_timesteps     = 2_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id=f"{RUN_ID}_din_v4"),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_stage6_din_v4",
        ),
    ],
    tb_log_name         = f"stage6_din_v4_{RUN_ID}",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 6 DIN v4 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
