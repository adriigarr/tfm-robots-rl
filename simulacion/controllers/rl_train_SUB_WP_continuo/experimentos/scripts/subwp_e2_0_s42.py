"""
EXPERIMENTO E2.0 — Pre-entrenamiento stage6 sin peatón | SUBWP | SEED=42

Objetivo: obtener un modelo SUBWP con el ciclo completo (approach→exit→return)
dominado antes de añadir el peatón en E2.1. Equivalente al run003_sXX_stage6_final
que existe en STHWP pero no en SUBWP (SUBWP pasó directamente de stage5 a stage6_din).

El peatón está presente en el mundo (LIDAR lo detecta como obstáculo dinámico)
pero ped_obs=False → sin rewards/penalizaciones específicas de peatón.

Base: subwp_s42_wp75_r2_stage5_final (40 dims, stage5 — return)
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
LOAD_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_s42_wp75_r2_stage5_final")
SAVE_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_e2_0_s42_final")
CKPT_DIR  = os.path.join(CONTROLLER_DIR, "pruebas", "checkpoints_subwp_e2_0_s42")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"\n[E2.0 SUBWP s42] Stage6 sin peatón — ciclo completo")
print(f"  Base: {LOAD_PATH}")

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=False)
env._max_steps = 6000

model = PPO.load(LOAD_PATH, env=env)
model.tensorboard_log = os.path.join(CONTROLLER_DIR, "tensorboard_logs")
model.learning_rate = 5e-5
model.ent_coef      = 0.01

model.learn(
    total_timesteps     = 2_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id="subwp_e2_0_s42"),
        CheckpointCallback(save_freq=200_000, save_path=CKPT_DIR,
                           name_prefix="subwp_e2_0_s42"),
    ],
    tb_log_name         = "subwp_e2_0_s42",
    reset_num_timesteps = True,
)

model.save(SAVE_PATH)
print(f"[E2.0 SUBWP s42] Guardado en {SAVE_PATH}.zip")
env.supervisor.simulationQuit(0)
