"""
ETAPA 5 — Return puro.   ~2M steps  |  SEED=123  | v3

Robot teleportado a ret_path[2]=(0.0,7.25). Heading hacia ret_path[3] (+ ruido 0.2 rad).
v3: corrección de heading_to_webots_rotation — eje Z (vertical) en lugar de eje Y (horizontal).
El bug v1/v2 causaba pitch de 108° → bumper penetraba suelo → colisión en paso 1.

Requiere: pruebas/subwp_s123_wp75_stage4_final.zip
Salida:   pruebas/subwp_s123_wp75_stage5_final.zip
"""

import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH      = "warehouse_map01.json"
RUN_ID        = "subwp_s123"
SEED          = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=5)

model = PPO.load(
    f"./pruebas/{RUN_ID}_wp75_stage4_final",
    env             = env,
    learning_rate   = 3e-4,
    ent_coef        = 0.02,
    seed            = SEED,
    tensorboard_log = "./tensorboard_logs/",
)

SAVE_PATH = f"./pruebas/{RUN_ID}_wp75_stage5_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_wp75_stage5"

model.learn(
    total_timesteps = 2_000_000,
    callback        = [
        StatsCallback(goal_ids=env.goal_ids, stage=5, run_id=RUN_ID),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_wp75_stage5",
        ),
    ],
    tb_log_name         = f"stage5_{RUN_ID}_wp75_v3",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 5 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
