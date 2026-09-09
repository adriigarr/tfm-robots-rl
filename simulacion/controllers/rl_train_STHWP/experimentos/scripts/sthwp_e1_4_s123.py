"""
EXPERIMENTO E1.4 — Patience reward + proximidad reforzada | STHWP | SEED=123

Cambios respecto a E1.3:
  - Penalización proximidad P1: 0.8*exp(-2*d) → 1.5*exp(-3*d)  [más agresiva a distancias cortas]
  - Patience reward: +0.4 cuando dist_P1 < 0.8m y vel_lineal < 0.05 m/s
  Objetivo: que el robot aprenda a esperar cuando P1 bloquea la salida (col_exit)

Base: sthwp_e1_3_s123_final (52 dims, pred_horizon=[1.0,2.0], goal_28 resuelto)
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
LOAD_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "sthwp_e1_3_s123_final")
SAVE_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "sthwp_e1_4_s123_final")
CKPT_DIR  = os.path.join(CONTROLLER_DIR, "pruebas", "checkpoints_sthwp_e1_4_s123")

SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"\n[E1.4 STHWP s123] Patience reward + proximidad reforzada")
print(f"  Base: {LOAD_PATH}")

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25,
                ped_obs=True, pred_horizon=[1.0, 2.0])
env._max_steps = 7000

model = PPO.load(LOAD_PATH, env=env)
model.tensorboard_log = os.path.join(CONTROLLER_DIR, "tensorboard_logs")
model.learning_rate = 1e-5
model.ent_coef      = 0.005

model.learn(
    total_timesteps     = 2_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id="sthwp_e1_4_s123"),
        CheckpointCallback(save_freq=100_000, save_path=CKPT_DIR,
                           name_prefix="sthwp_e1_4_s123"),
    ],
    tb_log_name         = "sthwp_e1_4_s123",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[E1.4 STHWP s123] Guardado en {SAVE_PATH}.zip")
env.supervisor.simulationQuit(0)
