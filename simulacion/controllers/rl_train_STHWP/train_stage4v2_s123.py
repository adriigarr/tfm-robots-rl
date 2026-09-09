"""
ETAPA 4 v2 — Ciclo completo integrado (70% approach / 30% exit).   ~4M steps  |  SEED=123

Requiere: pruebas/run002_s123_stage3v2_final.zip
          inferencia_sthwp/resultados/arrival_headings_stage2.json

Base: stage 3 v2 (headings reales + reward marcha atrás).
Objetivo: ciclo approach→exit encadenado sobre los 28 goals.

Salida: pruebas/run002_s123_stage4v2_final.zip
"""

import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH      = "warehouse_map01.json"
RUN_ID        = "run002_s123"
SEED          = 123
HEADING_SIGMA = 0.25

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=4, heading_sigma=HEADING_SIGMA)

model = PPO.load(
    f"./pruebas/{RUN_ID}_stage3v2_final",
    env             = env,
    learning_rate   = 1e-4,
    ent_coef        = 0.01,
    seed            = SEED,
    tensorboard_log = "./tensorboard_logs/",
)

SAVE_PATH = f"./pruebas/{RUN_ID}_stage4v2_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_stage4v2"

model.learn(
    total_timesteps = 4_000_000,
    callback        = [
        StatsCallback(goal_ids=env.goal_ids, stage=4, run_id=RUN_ID),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_stage4v2",
        ),
    ],
    tb_log_name         = f"stage4v2_{RUN_ID}",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 4 v2 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
