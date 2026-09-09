"""
ETAPA 6 — Ciclo completo estático (approach→exit→return).  1M steps  |  SEED=524

Requiere: pruebas/run002_s524_stage4v2_final.zip

Base: stage 4 v2 (ciclo approach+exit consolidado).
Objetivo: encadenar los tres tramos en un único episodio sin reinicios.
  - approach: zona_espera → estantería
  - exit:     estantería → zona_descarga
  - return:   zona_descarga → zona_espera  (nuevo respecto a stage 4v2)

El episodio no termina al llegar a zona_descarga — el env hace switch interno
al tramo de retorno (stage==6 en webots_env.py step()).

LR reducida (1e-4) para preservar el conocimiento de approach+exit.
World: warehouse_1_static.wbt (sin peatones).

Salida: pruebas/run002_s524_stage6_final.zip
"""

import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH      = "warehouse_map01.json"
LOAD_ID       = "run002_s524"
RUN_ID        = "run003_s524"
SEED          = 524
HEADING_SIGMA = 0.25

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=HEADING_SIGMA)
env._max_steps = 7000  # 4000 ap+exit + 3000 return

model = PPO.load(
    f"./pruebas/{LOAD_ID}_stage4v2_final",
    env             = env,
    learning_rate   = 1e-4,
    ent_coef        = 0.02,
    seed            = SEED,
    tensorboard_log = "./tensorboard_logs/",
)

SAVE_PATH = f"./pruebas/{RUN_ID}_stage6_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_stage6"

model.learn(
    total_timesteps = 1_000_000,
    callback        = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id=RUN_ID),
        CheckpointCallback(
            save_freq   = 50_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_stage6",
        ),
    ],
    tb_log_name         = f"stage6_{RUN_ID}",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 6 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
