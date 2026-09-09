"""
ETAPA 2 — Approach puro, todos los goals.   ~4M steps  |  SEED=123

Requiere: pruebas/run002_s123_stage1_final.zip

Objetivo: generalizar el approach a los 28 goals del almacén con sesgo
PROB_SHELF_DIFICIL=0.7 hacia goals de bloques 3 y 5 (más alejados de zona espera).
Episodio termina al llegar a la estantería. Sin exit, sin _fase_escape.

Salida: pruebas/run002_s123_stage2_final.zip
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

env = WebotsEnv(map_path=MAP_PATH, stage=2)

model = PPO.load(
    f"./pruebas/{RUN_ID}_stage1_final",
    env             = env,
    learning_rate   = 2e-4,
    ent_coef        = 0.01,
    seed            = SEED,
    tensorboard_log = "./tensorboard_logs/",
)

SAVE_PATH = f"./pruebas/{RUN_ID}_stage2_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_stage2"

model.learn(
    total_timesteps = 4_000_000,
    callback        = [
        StatsCallback(goal_ids=env.goal_ids, stage=2, run_id=RUN_ID),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_stage2",
        ),
    ],
    tb_log_name         = f"stage2_{RUN_ID}",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 2 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
