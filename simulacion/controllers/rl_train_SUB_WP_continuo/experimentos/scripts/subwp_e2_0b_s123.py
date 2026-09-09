"""
EXPERIMENTO E2.0b — Pre-entrenamiento stage6 sin peatón | SUBWP | SEED=123 (corrección)

Por qué E2.0 s123 falló:
  subwp_s123_wp75_r2_stage5_final sufrió olvido catastrófico del approach durante
  el entrenamiento de stage5 (return-only). En stage6, el reset siempre empieza con
  approach, por lo que el modelo s123 producía ep_len=6000 (timeout) desde el primer
  batch — reward nunca positivo en 2M steps.

Corrección:
  Cargar desde subwp_s123_wp75_r2_stage4_final que sí domina approach+exit
  (stage4 entrena mezcla approach/exit). Mismo objetivo: aprender el ciclo
  completo (approach→exit→return) en stage6 sin peatón.

Base: subwp_s123_wp75_r2_stage4_final (40 dims, stage4 approach+exit)
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
LOAD_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_s123_wp75_r2_stage4_final")
SAVE_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_e2_0_s123_final")
CKPT_DIR  = os.path.join(CONTROLLER_DIR, "pruebas", "checkpoints_subwp_e2_0b_s123")

SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"\n[E2.0b SUBWP s123] Stage6 sin peatón — corrección (base: stage4)")
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
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id="subwp_e2_0b_s123"),
        CheckpointCallback(save_freq=200_000, save_path=CKPT_DIR,
                           name_prefix="subwp_e2_0b_s123"),
    ],
    tb_log_name         = "subwp_e2_0b_s123",
    reset_num_timesteps = True,
)

model.save(SAVE_PATH)
print(f"[E2.0b SUBWP s123] Guardado en {SAVE_PATH}.zip")
env.supervisor.simulationQuit(0)
