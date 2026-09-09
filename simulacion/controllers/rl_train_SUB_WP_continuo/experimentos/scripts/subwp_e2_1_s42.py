"""
EXPERIMENTO E2.1 — Observación realista (solo LIDAR 5m, sin supervisor) | SUBWP | SEED=42

Cambios respecto a E1.x:
  - Se elimina la observación por supervisor (posición+velocidad del peatón)
  - LIDAR ampliado de 3.5m a 5.0m para detección anticipada por sensor físico
  - Obs space: 40 dims (36 LIDAR + 4 estado), misma arquitectura que pre-peatón
  - Reentrenamiento desde E2.0 (stage6 sin peatón, ciclo completo dominado)

Base: subwp_e2_0_sXX_final (40 dims, stage6 sin peatón))
Nota: 6M steps con curriculum en goals difíciles
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
LOAD_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_e2_0_s42_final")
SAVE_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_e2_1_s42_final")
CKPT_DIR  = os.path.join(CONTROLLER_DIR, "pruebas", "checkpoints_subwp_e2_1_s42")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"\n[E2.1 SUBWP s42] Obs realista: solo LIDAR 5m (sin supervisor)")
print(f"  Base: {LOAD_PATH}")

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 6000

# Curriculum ponderado: goals problemáticos con 3× probabilidad
HARD_GOALS = {"goal_17", "goal_18", "goal_19", "goal_20", "goal_21",
              "goal_24", "goal_25", "goal_26", "goal_27", "goal_28"}
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
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id="subwp_e2_1_s42"),
        CheckpointCallback(save_freq=200_000, save_path=CKPT_DIR,
                           name_prefix="subwp_e2_1_s42"),
    ],
    tb_log_name         = "subwp_e2_1_s42",
    reset_num_timesteps = True,
)

model.save(SAVE_PATH)
print(f"[E2.1 SUBWP s42] Guardado en {SAVE_PATH}.zip")
env.supervisor.simulationQuit(0)
