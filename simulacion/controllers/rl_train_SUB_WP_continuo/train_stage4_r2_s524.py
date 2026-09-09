"""
ETAPA 4 r2 — Ciclo approach+exit (70%/30%) | SEED=524 | dropoff corregido

Requiere: pruebas/subwp_s524_wp75_r2_stage3_final.zip
Salida:   pruebas/subwp_s524_wp75_r2_stage4_final.zip
"""

import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH      = "warehouse_map01.json"
RUN_ID        = "subwp_s524"
SEED          = 524
HEADING_SIGMA = 0.25

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=4, heading_sigma=HEADING_SIGMA)

model = PPO.load(
    f"./pruebas/{RUN_ID}_wp75_r2_stage3_final",
    env             = env,
    learning_rate   = 1e-4,
    ent_coef        = 0.01,
    seed            = SEED,
    tensorboard_log = "./tensorboard_logs/",
)

SAVE_PATH = f"./pruebas/{RUN_ID}_wp75_r2_stage4_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_wp75_r2_stage4"

model.learn(
    total_timesteps     = 4_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=4, run_id=f"{RUN_ID}_r2"),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_wp75_r2_stage4",
        ),
    ],
    tb_log_name         = f"stage4_r2_{RUN_ID}_wp75_noped",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 4 r2 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
