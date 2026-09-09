"""
EXPERIMENTO E2.1 — Observación realista (solo LIDAR 5m, sin supervisor) | STHWP | SEED=524

Cambios respecto a E1.x:
  - Se elimina la observación por supervisor (posición+velocidad del peatón)
  - Se elimina pred_horizon (también dependía del supervisor)
  - LIDAR ampliado de 3.5m a 5.0m para detección anticipada por sensor físico
  - Obs space: 40 dims (36 LIDAR + 4 estado), misma arquitectura que stage6 pre-peatón
  - Reentrenamiento desde stage6_final (no fine-tune — obs space incompatible con E1.x)

Base: run003_s524_stage6_final (40 dims, stage 6 sin peatón)
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
LOAD_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "run003_s524_stage6_final")
SAVE_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "sthwp_e2_1_s524_final")
CKPT_DIR  = os.path.join(CONTROLLER_DIR, "pruebas", "checkpoints_sthwp_e2_1_s524")

SEED = 524
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"\n[E2.1 STHWP s524] Obs realista: solo LIDAR 5m (sin supervisor)")
print(f"  Base: {LOAD_PATH}")

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 7000

# Curriculum ponderado: goals problemáticos con 3× probabilidad
HARD_GOALS = {"goal_14", "goal_15", "goal_16", "goal_23", "goal_24",
              "goal_25", "goal_26", "goal_27", "goal_28"}
_weights = np.array([3.0 if gid in HARD_GOALS else 1.0 for gid in env.goal_ids])
_weights /= _weights.sum()

def _sample_weighted():
    return int(env.np_random.choice(len(env.goal_ids), p=_weights))

env._sample_goal_approach = _sample_weighted

model = PPO.load(LOAD_PATH, env=env)
model.tensorboard_log = os.path.join(CONTROLLER_DIR, "tensorboard_logs")
model.learning_rate = 1e-4
model.ent_coef      = 0.01

model.learn(
    total_timesteps     = 6_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id="sthwp_e2_1_s524"),
        CheckpointCallback(save_freq=200_000, save_path=CKPT_DIR,
                           name_prefix="sthwp_e2_1_s524"),
    ],
    tb_log_name         = "sthwp_e2_1_s524",
    reset_num_timesteps = True,
)

model.save(SAVE_PATH)
print(f"[E2.1 STHWP s524] Guardado en {SAVE_PATH}.zip")
env.supervisor.simulationQuit(0)
