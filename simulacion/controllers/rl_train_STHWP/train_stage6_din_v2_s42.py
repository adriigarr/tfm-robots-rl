"""
ETAPA 6 DIN v2 — Ciclo completo + esquiva real (Opción 1 + Opción A) | SEED=42

Mejoras sobre v1:
  - Opción 1: peatones aleatorizados en posición inicial cada reset (ped_obs=True)
  - Opción A: observación de peatones (pos relativa + vel) → 48 dims obs
  - Transferencia de pesos: run003_s42_stage6_final (40d) → red expandida (48d)
    Las 8 dims nuevas se inicializan a 0 (el modelo ignora peatones al inicio
    y aprende a usarlos progresivamente durante el entrenamiento).

World: warehouse_1.wbt (PEDESTRIAN_1 + PEDESTRIAN_2 activos)
Salida: pruebas/run003_s42_stage6_din_v2_final.zip
"""

import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH  = "warehouse_map01.json"
LOAD_ID   = "run003_s42"
RUN_ID    = "run003_s42"
SEED      = 42
OLD_OBS_DIM = 40
NEW_OBS_DIM = 48

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 7000

# ── Crear nuevo modelo con obs=48 ─────────────────────────────────────────────
new_model = PPO(
    "MlpPolicy", env,
    learning_rate   = 5e-5,
    ent_coef        = 0.01,
    seed            = SEED,
    tensorboard_log = "./tensorboard_logs/",
    verbose         = 1,
)

# ── Cargar modelo base (40 dims) y expandir pesos ────────────────────────────
old_model = PPO.load(f"./pruebas/{LOAD_ID}_stage6_final")

old_state = old_model.policy.state_dict()
new_state = new_model.policy.state_dict()

transferred, expanded, skipped = 0, 0, 0
for key in new_state:
    if key not in old_state:
        skipped += 1
        continue
    old_w = old_state[key]
    new_w = new_state[key]
    if old_w.shape == new_w.shape:
        new_state[key] = old_w.clone()
        transferred += 1
    elif (len(old_w.shape) == 2 and
          old_w.shape[1] == OLD_OBS_DIM and
          new_w.shape[1] == NEW_OBS_DIM):
        expanded_w = torch.zeros_like(new_w)
        expanded_w[:, :OLD_OBS_DIM] = old_w
        new_state[key] = expanded_w
        expanded += 1
    else:
        skipped += 1

new_model.policy.load_state_dict(new_state)
print(f"[EXPAND] Transferidos={transferred} | Expandidos={expanded} | Skipped={skipped}")
print(f"[EXPAND] {OLD_OBS_DIM}d → {NEW_OBS_DIM}d OK")

SAVE_PATH = f"./pruebas/{RUN_ID}_stage6_din_v2_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_stage6_din_v2"

new_model.learn(
    total_timesteps     = 1_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id=f"{RUN_ID}_din_v2"),
        CheckpointCallback(
            save_freq   = 50_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_stage6_din_v2",
        ),
    ],
    tb_log_name         = f"stage6_din_v2_{RUN_ID}",
    reset_num_timesteps = True,
)

new_model.save(SAVE_PATH)
print(f"[ETAPA 6 DIN v2 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
