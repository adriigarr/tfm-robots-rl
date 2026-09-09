"""
ETAPA 6 DIN v4 — Replanning integrado en entorno | SEED=42

Continúa desde subwp_s42_wp75_stage6_din_v3_final (48 dims, ~6M steps previos).
Novedad respecto a v3: webots_env.py ejecuta replanificación A* dinámica mid-episode
cuando un peatón intercepta el segmento robot→wp_actual (dist_perp < 0.6 m).
Tras el replanning, _wp_idx se resetea a 0 para recorrer la nueva ruta desde el inicio.
El submuestreo es fase-consciente: WP_STEP_EXIT=0.75m en exit, WP_STEP=1.5m en el resto.

Parámetros de replanning (class constants en WebotsEnv):
  REPLAN_DIST_PERP     = 0.6 m
  REPLAN_INFLATE_CELLS = 3 celdas (0.75 m)
  REPLAN_COOLDOWN      = 40 pasos

World: warehouse_1_subwp.wbt (PEDESTRIAN_1 + PEDESTRIAN_2 activos)
Salida: pruebas/subwp_s42_wp75_stage6_din_v4_final.zip
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
RUN_ID   = "subwp_s42_wp75"
SEED     = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 6000

model = PPO.load(f"./pruebas/{LOAD_ID}_stage6_din_v3_final", env=env)
model.learning_rate = 5e-5
model.ent_coef      = 0.005

SAVE_PATH = f"./pruebas/{RUN_ID}_stage6_din_v4_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_stage6_din_v4"

model.learn(
    total_timesteps     = 2_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id=f"{RUN_ID}_din_v4"),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_stage6_din_v4",
        ),
    ],
    tb_log_name         = f"stage6_din_v4_{RUN_ID}",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[ETAPA 6 DIN v4 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
