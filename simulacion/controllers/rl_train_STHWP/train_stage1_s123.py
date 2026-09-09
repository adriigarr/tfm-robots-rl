"""
ETAPA 1 — Approach puro, goal único (goal_00).   ~500k steps  |  SEED=123

Criterio de paso a stage 2: exito_ult100_% ≥ 95% en TensorBoard.
Si al terminar los 500k el éxito es < 70%, subir total_timesteps a 800_000
y relanzar desde cero.

Salida: pruebas/run002_s123_stage1_final.zip
"""

import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH = "warehouse_map01.json"
RUN_ID   = "run002_s123"
SEED     = 123

# ── reproducibilidad ──────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=1)

model = PPO(
    policy          = "MlpPolicy",
    env             = env,
    learning_rate   = 3e-4,
    n_steps         = 2048,
    batch_size      = 64,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    ent_coef        = 0.01,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    seed            = SEED,
    verbose         = 1,
    tensorboard_log = "./tensorboard_logs/",
)

SAVE_PATH = f"./pruebas/{RUN_ID}_stage1_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_stage1"

model.learn(
    total_timesteps = 500_000,
    callback        = [
        StatsCallback(goal_ids=env.goal_ids, stage=1, run_id=RUN_ID),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_stage1",
        ),
    ],
    tb_log_name         = f"stage1_{RUN_ID}",
    reset_num_timesteps = True,
)

model.save(SAVE_PATH)
print(f"[ETAPA 1 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)  # cierra Webots para permitir automatización
