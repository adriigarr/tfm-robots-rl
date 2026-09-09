"""
ETAPA 3 v2 — Exit puro, iteración 6.   ~4M steps  |  SEED=123

Requiere: pruebas/run002_s123_stage2_final.zip
          inferencia_sthwp/resultados/arrival_headings_stage2.json

Cambios respecto a i5:
  - Heading de teleport: headings REALES medidos en inferencia stage 2
    (robot empieza con la orientación con la que llega físicamente a la estantería).
  - Reward de orientación: premia marcha atrás hacia el subgoal
    (vel_sign * cos(angulo_rel)), eliminando el sesgo que hacía el retroceso inviable.

Salida: pruebas/run002_s123_stage3v2_final.zip
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
ENT_COEF      = 0.01

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=3, heading_sigma=HEADING_SIGMA)

model = PPO.load(
    f"./pruebas/{RUN_ID}_stage2_final",
    env             = env,
    learning_rate   = 2e-4,
    ent_coef        = ENT_COEF,
    seed            = SEED,
    tensorboard_log = "./tensorboard_logs/",
)

SAVE_PATH = f"./pruebas/{RUN_ID}_stage3v2_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_stage3_i6"

model.learn(
    total_timesteps = 4_000_000,
    callback        = [
        StatsCallback(goal_ids=env.goal_ids, stage=3, run_id=RUN_ID),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_stage3_i6",
        ),
    ],
    tb_log_name         = f"stage3_i6_{RUN_ID}",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 3 v2 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
