"""
ETAPA 3 r2 — Exit puro | SEED=524 | dropoff corregido (-11.0, 0.0)

Reentrenamiento desde stage2_final con la zona de descarga corregida.
La primera ejecución usó dropoff=(0.0, 10.5) en lugar de (-11.0, 0.0),
invalidando las fases de exit y retorno de todos los modelos posteriores.

Requiere: pruebas/subwp_s524_stage2_final.zip
Salida:   pruebas/subwp_s524_wp75_r2_stage3_final.zip
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

env = WebotsEnv(map_path=MAP_PATH, stage=3, heading_sigma=HEADING_SIGMA)

model = PPO.load(
    f"./pruebas/{RUN_ID}_stage2_final",
    env             = env,
    learning_rate   = 2e-4,
    ent_coef        = 0.01,
    seed            = SEED,
    tensorboard_log = "./tensorboard_logs/",
)

SAVE_PATH = f"./pruebas/{RUN_ID}_wp75_r2_stage3_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_wp75_r2_stage3"

model.learn(
    total_timesteps     = 4_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=3, run_id=f"{RUN_ID}_r2"),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_wp75_r2_stage3",
        ),
    ],
    tb_log_name         = f"stage3_r2_{RUN_ID}_wp75_noped",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 3 r2 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
