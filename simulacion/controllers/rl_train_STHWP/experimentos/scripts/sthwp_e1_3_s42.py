"""
EXPERIMENTO E1.3 — Trayectoria P1 extendida x∈[-4,4] | STHWP | SEED=42

P1 pasa de oscilar x∈[-2,4] a x∈[-4,4]. El tramo extra [-4,-2] aleja a P1
del corredor de approach de goals 24-28 durante más tiempo, creando ventanas
más largas para que el robot aprenda el timing correcto.

Base: sthwp_e1_pred_s42_final (52 dims — mejor modelo s42 con obs predictiva)
NOTA: se mantiene pred_horizon=[1.0,2.0] para aprovechar la obs predictiva ya aprendida.
World: warehouse_1_1ped.wbt (PEDESTRIAN_1 con trajectory=-4 0.3, 4 0.3)
Steps: 2M (fine-tune desde E1_pred, lr=1e-5)
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
LOAD_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "sthwp_e1_pred_s42_final")
SAVE_PATH = os.path.join(CONTROLLER_DIR, "pruebas", "sthwp_e1_3_s42_final")
CKPT_DIR  = os.path.join(CONTROLLER_DIR, "pruebas", "checkpoints_sthwp_e1_3_s42")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"\n[E1.3 STHWP s42] Trayectoria P1 extendida x∈[-4,4]")
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
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id="sthwp_e1_3_s42"),
        CheckpointCallback(save_freq=100_000, save_path=CKPT_DIR,
                           name_prefix="sthwp_e1_3_s42"),
    ],
    tb_log_name         = "sthwp_e1_3_s42",
    reset_num_timesteps = False,
)

model.save(SAVE_PATH)
print(f"[E1.3 STHWP s42] Guardado en {SAVE_PATH}.zip")
env.supervisor.simulationQuit(0)
