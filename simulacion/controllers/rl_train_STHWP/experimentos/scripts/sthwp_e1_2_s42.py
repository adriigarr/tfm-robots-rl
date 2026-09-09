"""
EXPERIMENTO E1.2 — Sesgo goals problemáticos | STHWP | SEED=42

Base:   sthwp_e1_1ped_s42_final (E1, 88.1% inferencia)
World:  warehouse_1_1ped.wbt (solo PEDESTRIAN_1)
Sesgo:  70% goals 24-28 (sector P1) · 30% resto
Steps:  1 500 000 adicionales
LR:     2e-5 · ent_coef=0.005
Salida: pruebas/sthwp_e1_2_s42_final.zip
"""

import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from webots_env import WebotsEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

SEED          = 42
CONTROLLER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MAP_PATH      = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
LOAD_PATH     = os.path.join(CONTROLLER_DIR, "pruebas", "sthwp_e1_1ped_s42_final")
SAVE_PATH     = os.path.join(CONTROLLER_DIR, "pruebas", "sthwp_e1_2_s42_final")
CKPT_DIR      = os.path.join(CONTROLLER_DIR, "pruebas", "checkpoints_sthwp_e1_2_s42")

# goals 24-28 (índices 0-based: 23-27)
HARD_INDICES = [23, 24, 25, 26, 27]
PROB_HARD    = 0.70

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 7000

_rng = np.random.default_rng(SEED)
def _biased_goal():
    if _rng.random() < PROB_HARD:
        return int(HARD_INDICES[_rng.integers(0, len(HARD_INDICES))])
    return int(env.np_random.integers(0, len(env.goals)))
env._sample_goal_approach = _biased_goal

model = PPO.load(LOAD_PATH, env=env)
model.learning_rate = 2e-5
model.ent_coef      = 0.005

checkpoint_cb = CheckpointCallback(
    save_freq=100_000, save_path=CKPT_DIR,
    name_prefix="sthwp_e1_2_s42")

print(f"\n[E1.2 STHWP s42] sesgo {PROB_HARD*100:.0f}% goals 24-28 | 1.5M steps")
model.learn(
    total_timesteps=1_500_000,
    reset_num_timesteps=False,
    callback=checkpoint_cb,
    tb_log_name="sthwp_e1_2_s42",
    progress_bar=False,
)
model.save(SAVE_PATH)
print(f"[OK] guardado en {SAVE_PATH}")
env.supervisor.simulationQuit(0)
