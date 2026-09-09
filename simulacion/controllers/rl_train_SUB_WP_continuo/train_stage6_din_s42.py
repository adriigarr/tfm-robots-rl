"""
ETAPA 6 DIN — Ciclo completo con peatones (fine-tuning dinámico).  1M steps  |  SEED=42

Requiere: pruebas/subwp_s42_wp75_stage5_final.zip

Fine-tuning del modelo estático con obstáculos dinámicos activos (PEDESTRIAN_1 + PEDESTRIAN_2).
World: warehouse_1_subwp.wbt (con peatones).
LR reducida (5e-5) para preservar el conocimiento estático.

Salida: pruebas/subwp_s42_wp75_stage6_din_final.zip
"""

import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH = "warehouse_map01.json"
LOAD_ID  = "subwp_s42_wp75"
RUN_ID   = "subwp_s42"
SEED     = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=6)
env._max_steps = 6000

model = PPO.load(
    f"./pruebas/{LOAD_ID}_stage5_final",
    env             = env,
    learning_rate   = 5e-5,
    ent_coef        = 0.01,
    seed            = SEED,
    tensorboard_log = "./tensorboard_logs/",
)

SAVE_PATH = f"./pruebas/{LOAD_ID}_stage6_din_final"
CKPT_DIR  = f"./pruebas/checkpoints_{LOAD_ID}_stage6_din"

model.learn(
    total_timesteps = 1_000_000,
    callback        = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id=f"{RUN_ID}_din"),
        CheckpointCallback(
            save_freq   = 50_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{LOAD_ID}_stage6_din",
        ),
    ],
    tb_log_name         = f"stage6_din_{RUN_ID}_wp75",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 6 DIN | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
