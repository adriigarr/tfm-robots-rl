"""
EXPERIMENTO E1 — 1 Peatón (solo PEDESTRIAN_1) | SUB-WP | SEED=524

Base: subwp_s524_wp75_r2_stage6_din_v3_final
      (48 dims, dropoff correcto en (-11,0))
World: warehouse_1_subwp_1ped.wbt  (solo PEDESTRIAN_1)
Salida: pruebas/subwp_e1_1ped_s524_final.zip
"""

import os, sys, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

CONTROLLER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MAP_PATH  = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
LOAD_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_s524_wp75_r2_stage6_din_v3_final")
SAVE_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_e1_1ped_s524_final")
CKPT_DIR  = os.path.join(CONTROLLER_DIR, "pruebas", "checkpoints_subwp_e1_1ped_s524")

SEED = 524
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 6000

model = PPO.load(LOAD_PATH, env=env)
model.learning_rate = 5e-5
model.ent_coef      = 0.005

model.learn(
    total_timesteps     = 3_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id="subwp_e1_1ped_s524"),
        CheckpointCallback(save_freq=100_000, save_path=CKPT_DIR,
                           name_prefix="subwp_e1_1ped_s524"),
    ],
    tb_log_name         = "subwp_e1_1ped_s524",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[E1 SUB-WP s524] Guardado en {SAVE_PATH}.zip")
env.supervisor.simulationQuit(0)
