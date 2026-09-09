"""
ETAPA 6 DIN v3 r2 — Peatones + recompensa B1 | SEED=42 | dropoff corregido

Salta directamente de stage5_r2 (40-dim obs) a v3 (ped_obs=True, 48-dim obs).
Crea un modelo nuevo con la arquitectura 48-dim y transfiere los pesos del
stage5_r2 expandiendo la capa de entrada (8 dims de peatones → 0.0 al inicio).

Requiere: pruebas/subwp_s42_wp75_r2_stage5_final.zip
Salida:   pruebas/subwp_s42_wp75_r2_stage6_din_v3_final.zip
"""

import numpy as np
import random
import torch
from webots_env import WebotsEnv
from callbacks import StatsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MAP_PATH    = "warehouse_map01.json"
RUN_ID      = "subwp_s42_wp75"
SEED        = 42
OLD_OBS_DIM = 40
NEW_OBS_DIM = 48

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = WebotsEnv(map_path=MAP_PATH, stage=6, heading_sigma=0.25, ped_obs=True)
env._max_steps = 6000

# ── Crear modelo nuevo con obs=48 ─────────────────────────────────────────────
new_model = PPO(
    "MlpPolicy", env,
    learning_rate   = 5e-5,
    ent_coef        = 0.005,
    seed            = SEED,
    tensorboard_log = "./tensorboard_logs/",
    verbose         = 1,
)

# ── Cargar stage5_r2 (40 dims) y expandir pesos ───────────────────────────────
old_model = PPO.load(f"./pruebas/{RUN_ID}_r2_stage5_final")

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

SAVE_PATH = f"./pruebas/{RUN_ID}_r2_stage6_din_v3_final"
CKPT_DIR  = f"./pruebas/checkpoints_{RUN_ID}_r2_stage6_din_v3"

new_model.learn(
    total_timesteps     = 2_000_000,
    callback            = [
        StatsCallback(goal_ids=env.goal_ids, stage=6, run_id=f"{RUN_ID}_r2_din_v3"),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = CKPT_DIR,
            name_prefix = f"{RUN_ID}_r2_stage6_din_v3",
        ),
    ],
    tb_log_name         = f"stage6_din_v3_r2_{RUN_ID}",
    reset_num_timesteps = True,
)

new_model.save(SAVE_PATH)
print(f"[ETAPA 6 DIN v3 r2 | {RUN_ID}] Modelo guardado en {SAVE_PATH}.zip")

env.supervisor.simulationQuit(0)
