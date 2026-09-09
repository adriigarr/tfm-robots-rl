"""
ETAPA 6 DIN v3 — Recompensa B1 activa | SEED=123

Continúa desde el checkpoint pico de v2_cont (paso 2501472), que fue el punto
de máximo rendimiento antes del colapso por sobre-exploración.
Novedad: webots_env.py incluye penalización B1 (dist real al peatón, umbral 2m).

ent_coef reducido a 0.005 (vs 0.01 en v2_cont) para mitigar la tendencia al
colapso post-pico observada en esta seed (train/std creciente monotónicamente).

World: warehouse_1.wbt (PEDESTRIAN_1 + PEDESTRIAN_2 activos)
Salida: pruebas/run003_s123_stage6_din_v3_final.zip
"""

import os
import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH = "warehouse_map01.json"
RUN_ID   = "run003_s123"
SEED     = 123

CKPT_PATH = os.path.join(
    ".", "pruebas",
    "checkpoints_run003_s123_stage6_din_v2_cont",
    "run003_s123_stage6_din_v2_cont_2501472_steps",
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 7000

model = PPO.load(CKPT_PATH, env=env)
model.learning_rate = 5e-5
model.ent_coef      = 0.005

SAVE_PATH = f"./pruebas/{RUN_ID}_stage6_din_v3_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_stage6_din_v3"

model.learn(
    total_timesteps     = 2_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id=f"{RUN_ID}_din_v3"),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_stage6_din_v3",
        ),
    ],
    tb_log_name         = f"stage6_din_v3_{RUN_ID}",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 6 DIN v3 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
