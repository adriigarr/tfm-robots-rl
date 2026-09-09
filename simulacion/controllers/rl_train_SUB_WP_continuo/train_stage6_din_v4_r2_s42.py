"""
ETAPA 6 DIN v4 r2 — Replanning integrado | SEED=42 | dropoff corregido

Continúa desde stage6_din_v3_r2_final con replanning A* integrado en webots_env.py.

Requiere: pruebas/subwp_s42_wp75_r2_stage6_din_v3_final.zip
Salida:   pruebas/subwp_s42_wp75_r2_stage6_din_v4_final.zip
"""

import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH = "warehouse_map01.json"
RUN_ID   = "subwp_s42_wp75"
SEED     = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 6000

model = PPO.load(
    f"./pruebas/{RUN_ID}_r2_stage6_din_v3_final",
    env             = env,
    tensorboard_log = "./tensorboard_logs/",
)
model.learning_rate = 5e-5
model.ent_coef      = 0.005

SAVE_PATH = f"./pruebas/{RUN_ID}_r2_stage6_din_v4_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_r2_stage6_din_v4"

model.learn(
    total_timesteps     = 2_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id=f"{RUN_ID}_r2_din_v4"),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_r2_stage6_din_v4",
        ),
    ],
    tb_log_name         = f"stage6_din_v4_r2_{RUN_ID}",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 6 DIN v4 r2 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
