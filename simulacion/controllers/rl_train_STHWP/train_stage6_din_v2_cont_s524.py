"""
ETAPA 6 DIN v2 CONT — Continuación entrenamiento esquiva real | SEED=524

Continúa desde run003_s524_stage6_din_v2_final (48 dims, 1M steps previos).
Sin expansión de pesos — obs space ya es 48 dims en modelo y env.
3M steps adicionales (total acumulado: ~4M desde baseline estático).

World: warehouse_1.wbt (PEDESTRIAN_1 + PEDESTRIAN_2 activos)
Salida: pruebas/run003_s524_stage6_din_v2_cont_final.zip
"""

import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH = "warehouse_map01.json"
LOAD_ID  = "run003_s524"
RUN_ID   = "run003_s524"
SEED     = 524

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 7000

model = PPO.load(f"./pruebas/{LOAD_ID}_stage6_din_v2_final", env=env)
model.learning_rate = 5e-5
model.ent_coef      = 0.01

SAVE_PATH = f"./pruebas/{RUN_ID}_stage6_din_v2_cont_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_stage6_din_v2_cont"

model.learn(
    total_timesteps     = 3_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id=f"{RUN_ID}_din_v2_cont"),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_stage6_din_v2_cont",
        ),
    ],
    tb_log_name         = f"stage6_din_v2_cont_{RUN_ID}",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 6 DIN v2 CONT | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
