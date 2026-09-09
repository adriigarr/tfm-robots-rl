"""
EXPERIMENTO E1.2 — Sesgo goals problemáticos | SUB-WP | SEED=123

Base:   subwp_e1_1ped_s123_final (E1, 73.8% inferencia)
World:  warehouse_1_subwp_1ped.wbt (solo PEDESTRIAN_1)
Sesgo:  70% goals 19-22 + 24-28 (clusters A y B) · 30% resto
Steps:  2 000 000 adicionales
LR:     2e-5 · ent_coef=0.005
Salida: pruebas/subwp_e1_2_s123_final.zip

Cluster A (goals 19-22): fallo SUBWP-específico, col_approach masivo
Cluster B (goals 24-28): sector PEDESTRIAN_1, compartido con STHWP
"""

import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from webots_env import WebotsEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

SEED           = 123
CONTROLLER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MAP_PATH       = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
LOAD_PATH      = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_e1_1ped_s123_final")
SAVE_PATH      = os.path.join(CONTROLLER_DIR, "pruebas", "subwp_e1_2_s123_final")
CKPT_DIR       = os.path.join(CONTROLLER_DIR, "pruebas", "checkpoints_subwp_e1_2_s123")

# Cluster A: goals 19-22 (índices 18-21) — fallo approach SUBWP
# Cluster B: goals 24-28 (índices 23-27) — sector P1
HARD_INDICES = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
PROB_HARD    = 0.70

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 6000

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
    name_prefix="subwp_e1_2_s123")

print(f"\n[E1.2 SUBWP s42] sesgo {PROB_HARD*100:.0f}% goals 19-22+24-28 | 2M steps")
model.learn(
    total_timesteps=2_000_000,
    reset_num_timesteps=False,
    callback=checkpoint_cb,
    tb_log_name="subwp_e1_2_s123",
    progress_bar=False,
)
model.save(SAVE_PATH)
print(f"[OK] guardado en {SAVE_PATH}")
env.supervisor.simulationQuit(0)
