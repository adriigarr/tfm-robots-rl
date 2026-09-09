"""
ETAPA 5 — Retorno puro (descarga→espera).   ~2M steps  |  SEED=524

Requiere: pruebas/run002_s524_stage4v2_final.zip

Base: stage 4 v2 (ciclo approach+exit consolidado).
Objetivo: aprender el tramo descarga→zona_espera para completar el ciclo completo.
El robot spawna en el camino de retorno a ≥3m de zona_descarga (safe desde wall3).
La subgoal se calcula con compute_sth_subgoal(ret_path, pos, D_AHEAD_NORMAL=1.5m).

Fix aplicado: heading_utils ahora usa eje Z [0,0,1,-θ] (antes Y [0,1,0,-θ]).
Para θ≈-108° (heading SW) el eje Y producía pitch que hundía el bumper en el suelo.

Salida: pruebas/run002_s524_stage5_final.zip
"""

import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH      = "warehouse_map01.json"
RUN_ID        = "run002_s524"
SEED          = 524
HEADING_SIGMA = 0.25

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=5, heading_sigma=HEADING_SIGMA)
env._max_steps = 3000

model = PPO.load(
    f"./pruebas/{RUN_ID}_stage4v2_final",
    env             = env,
    learning_rate   = 3e-4,
    ent_coef        = 0.02,
    seed            = SEED,
    tensorboard_log = "./tensorboard_logs/",
)

SAVE_PATH = f"./pruebas/{RUN_ID}_stage5_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_stage5"

model.learn(
    total_timesteps = 2_000_000,
    callback        = [
        StatsCallback(goal_ids=env.goal_ids, stage=5, run_id=RUN_ID),
        CheckpointCallback(
            save_freq   = 50_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_stage5",
        ),
    ],
    tb_log_name         = f"stage5_{RUN_ID}",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 5 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
