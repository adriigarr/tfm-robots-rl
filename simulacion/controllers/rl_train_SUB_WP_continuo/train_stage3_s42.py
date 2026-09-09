"""
ETAPA 3 — Exit puro.   ~4M steps  |  SEED=42

Requiere: pruebas/subwp_s42_stage2_final.zip
          inferencia_subwp/resultados/arrival_headings_stage2.json

Teleporta el robot a la estantería con heading real medido en inferencia stage 2
(+ ruido gaussiano heading_sigma=0.25 rad ≈ 14°).

Salida: pruebas/subwp_s42_stage3_final.zip
"""

import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH      = "warehouse_map01.json"
RUN_ID        = "subwp_s42"
SEED          = 42
HEADING_SIGMA = 0.25

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=3, heading_sigma=HEADING_SIGMA)

model = PPO.load(
    f"./pruebas/{RUN_ID}_stage2_final",
    env             = env,
    learning_rate   = 2e-4,
    ent_coef        = 0.01,
    seed            = SEED,
    tensorboard_log = "./tensorboard_logs/",
)

SAVE_PATH = f"./pruebas/{RUN_ID}_wp75_stage3_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_wp75_stage3"

model.learn(
    total_timesteps = 4_000_000,
    callback        = [
        StatsCallback(goal_ids=env.goal_ids, stage=3, run_id=RUN_ID),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_wp75_stage3",
        ),
    ],
    tb_log_name         = f"stage3_{RUN_ID}_wp75",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 3 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
