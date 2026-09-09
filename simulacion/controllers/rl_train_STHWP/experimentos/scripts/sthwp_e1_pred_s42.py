"""
EXPERIMENTO E1_pred — Observación predictiva P1 | STHWP | SEED=42

Añade 4 dims de predicción de posición de P1 en t+1s y t+2s a la observación
(48→52 dims). Se parte de los pesos de E1 (s42) mediante weight transplant:
  - Capas con shape independiente de obs_dim: se copian directamente.
  - Primera capa (Linear 48→64): se extiende a Linear 52→64 inicializando
    las 4 nuevas columnas a cero. El modelo "ignora" las predicciones al
    principio y aprende gradualmente a usarlas.

Base: sthwp_e1_1ped_s42_final (48 dims, 3M steps)
World: warehouse_1_1ped.wbt (solo PEDESTRIAN_1 activo)
Total: 3M steps fine-tune con obs 52 dims
"""

import os, sys, random, io, zipfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

CONTROLLER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MAP_PATH   = os.path.join(CONTROLLER_DIR, "warehouse_map01.json")
LOAD_PATH  = os.path.join(CONTROLLER_DIR, "pruebas", "sthwp_e1_1ped_s42_final")
SAVE_PATH  = os.path.join(CONTROLLER_DIR, "pruebas", "sthwp_e1_pred_s42_final")
CKPT_DIR   = os.path.join(CONTROLLER_DIR, "pruebas", "checkpoints_sthwp_e1_pred_s42")

SEED       = 42
OLD_OBS    = 48
NEW_OBS    = 52
PRED_HORIZON = [1.0, 2.0]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"\n[E1_pred STHWP s42] Weight transplant {OLD_OBS}→{NEW_OBS} dims")
print(f"  Base: {LOAD_PATH}")

# Crear env con obs extendida (52 dims)
env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25,
                ped_obs=True, pred_horizon=PRED_HORIZON)
env._max_steps = 7000
print(f"  Obs space: {env.observation_space.shape[0]} dims")

# Cargar pesos del modelo E1 sin env (evita chequeo de obs space)
old_model = PPO.load(LOAD_PATH, env=None, device="cpu")
old_sd = old_model.policy.state_dict()

# Crear nuevo modelo desde cero con obs 52 dims
new_model = PPO(
    "MlpPolicy", env,
    learning_rate = 2e-5,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    ent_coef      = 0.005,
    verbose       = 1,
    tensorboard_log = os.path.join(CONTROLLER_DIR, "tensorboard_logs"),
    seed          = SEED,
    device        = "cpu",
)

# Weight transplant
with torch.no_grad():
    new_sd = new_model.policy.state_dict()
    for key, old_val in old_sd.items():
        if key not in new_sd:
            continue
        new_val = new_sd[key]
        if new_val.shape == old_val.shape:
            new_sd[key] = old_val.clone()
        elif (new_val.shape[0] == old_val.shape[0] and
              new_val.dim() == 2 and old_val.dim() == 2 and
              new_val.shape[1] == NEW_OBS and old_val.shape[1] == OLD_OBS):
            # Primera capa: (64, 48) → (64, 52); nuevas columnas a cero
            extended = torch.zeros_like(new_val)
            extended[:, :OLD_OBS] = old_val
            new_sd[key] = extended
        else:
            print(f"  [WARN] skip '{key}': {old_val.shape} → {new_val.shape}")
    new_model.policy.load_state_dict(new_sd)

print(f"  Transplante completado. Iniciando entrenamiento...")

new_model.learn(
    total_timesteps     = 3_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id="sthwp_e1_pred_s42"),
        CheckpointCallback(save_freq=100_000, save_path=CKPT_DIR,
                           name_prefix="sthwp_e1_pred_s42"),
    ],
    tb_log_name         = "sthwp_e1_pred_s42",
    reset_num_timesteps = False,
)

new_model.save(SAVE_PATH)
print(f"[E1_pred STHWP s42] Guardado en {SAVE_PATH}.zip")
env.supervisor.simulationQuit(0)
