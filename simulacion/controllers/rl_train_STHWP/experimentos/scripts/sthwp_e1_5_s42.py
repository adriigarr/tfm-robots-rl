"""
EXPERIMENTO E1.5 — Exit-corridor reward + curriculum ponderado | STHWP | SEED=42

Cambios respecto a E1.4:
  - Patience reward eliminado (demasiado genérico, no ayudó en col_exit)
  - Exit-corridor reward: +0.6/paso cuando P1 está en cono 45° delante del robot
    durante la fase de exit (hacia_descarga o fase_escape), dist < 2.5m, vel < 0.05
  - Curriculum ponderado: goals 14-16, 23-28 con peso 3× (60% del tiempo)
  - 4M steps fine-tune (vs 2M en E1.4 — convergencia insuficiente entre seeds)

Base: sthwp_e1_4_s42_final (52 dims, pred_horizon=[1.0,2.0])
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
LOAD_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "sthwp_e1_4_s42_final")
SAVE_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "sthwp_e1_5_s42_final")
CKPT_DIR  = os.path.join(CONTROLLER_DIR, "pruebas", "checkpoints_sthwp_e1_5_s42")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"\n[E1.5 STHWP s42] Exit-corridor reward + curriculum ponderado")
print(f"  Base: {LOAD_PATH}")

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25,
                ped_obs=True, pred_horizon=[1.0, 2.0])
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
model.learning_rate = 1e-5
model.ent_coef      = 0.005

model.learn(
    total_timesteps     = 4_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id="sthwp_e1_5_s42"),
        CheckpointCallback(save_freq=200_000, save_path=CKPT_DIR,
                           name_prefix="sthwp_e1_5_s42"),
    ],
    tb_log_name         = "sthwp_e1_5_s42",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[E1.5 STHWP s42] Guardado en {SAVE_PATH}.zip")
env.supervisor.simulationQuit(0)
