"""
EXPERIMENTO E1 — 1 Peatón (solo PEDESTRIAN_1) | STHWP | SEED=42

Propuesta E1: curriculum por peatones. Entrena con un solo peatón (P1: oscila
x∈[-2,4], y=0.3 a 0.5 m/s) antes de introducir ambos peatones. Objetivo:
conseguir 100% de tasa de éxito con 1 peatón para luego escalar a 2.

Base: run003_s42_stage6_din_v3_final (48 dims, ~6M steps acumulados)
World: warehouse_1_1ped.wbt  (solo PEDESTRIAN_1; PEDESTRIAN_2 eliminado)
      → _ped_nodes tiene 1 elemento; dims P2 en obs = [0,0,0,0] siempre.

Salida: pruebas/sthwp_e1_1ped_s42_final.zip
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
LOAD_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "run003_s42_stage6_din_v3_final")
SAVE_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "sthwp_e1_1ped_s42_final")
CKPT_DIR  = os.path.join(CONTROLLER_DIR, "pruebas", "checkpoints_sthwp_e1_1ped_s42")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 7000

model = PPO.load(LOAD_PATH, env=env)
model.learning_rate = 5e-5
model.ent_coef      = 0.005

model.learn(
    total_timesteps     = 3_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id="sthwp_e1_1ped_s42"),
        CheckpointCallback(save_freq=100_000, save_path=CKPT_DIR,
                           name_prefix="sthwp_e1_1ped_s42"),
    ],
    tb_log_name         = "sthwp_e1_1ped_s42",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[E1 STHWP s42] Guardado en {SAVE_PATH}.zip")
env.supervisor.simulationQuit(0)
