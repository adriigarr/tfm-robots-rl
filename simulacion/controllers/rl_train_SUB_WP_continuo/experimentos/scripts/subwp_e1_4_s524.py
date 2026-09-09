"""
EXPERIMENTO E1.4 — Patience reward + proximidad reforzada + replan agresivo | SUBWP | SEED=524

Cambios respecto a E1.3:
  - Penalización proximidad P1: 0.8*exp(-2*d) → 1.5*exp(-3*d)
  - Patience reward: +0.4 cuando dist_P1 < 0.8m y vel_lineal < 0.05 m/s
  - REPLAN_DIST_PERP: 0.6 → 1.0 m  (anticipar replanning antes)
  - REPLAN_INFLATE_CELLS: 3 → 4    (margen 1.0m alrededor de P1)
  - REPLAN_COOLDOWN: 40 → 20 steps  (replanning más frecuente)
  Objetivo: reducir col_exit en goals 18, 19, 21, 25 y mejorar cluster A

Base: subwp_e1_3_s524_final (48 dims, mejor SUBWP actual 81.6%)
Steps: 2M fine-tune | lr=1e-5
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
LOAD_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_e1_3_s524_final")
SAVE_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_e1_4_s524_final")
CKPT_DIR  = os.path.join(CONTROLLER_DIR, "pruebas", "checkpoints_subwp_e1_4_s524")

SEED = 524
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"\n[E1.4 SUBWP s524] Patience reward + replan agresivo")
print(f"  Base: {LOAD_PATH}")

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 6000

model = PPO.load(LOAD_PATH, env=env)
model.tensorboard_log = os.path.join(CONTROLLER_DIR, "tensorboard_logs")
model.learning_rate = 1e-5
model.ent_coef      = 0.005

model.learn(
    total_timesteps     = 2_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id="subwp_e1_4_s524"),
        CheckpointCallback(save_freq=100_000, save_path=CKPT_DIR,
                           name_prefix="subwp_e1_4_s524"),
    ],
    tb_log_name         = "subwp_e1_4_s524",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[E1.4 SUBWP s524] Guardado en {SAVE_PATH}.zip")
env.supervisor.simulationQuit(0)
