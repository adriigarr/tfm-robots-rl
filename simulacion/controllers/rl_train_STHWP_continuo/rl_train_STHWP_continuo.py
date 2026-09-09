"""
    Este script se encarga de arrancar el entrenamiento. 
    Se centra en hacer 4 cosas:
    1. Instanciar el entorno
    2. Verificar que el entorno es compatible con SB3
    3. Crear el modelo de RL
    4. Lanzar el entrenamiento
"""

# librerias
from webots_env import WebotsEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from callbacks import StatsCallback
from stable_baselines3.common.callbacks import CheckpointCallback

# instanciar el entorno
env = WebotsEnv(map_path='warehouse_map01.json')

# verificar que el entorno es compatible con SB3
# check_env(env, warn=True)

stats_cb = StatsCallback(goal_ids=env.goal_ids)

""" # crear el modelo de RL desde cero
model = PPO(
    policy = "MlpPolicy",
    env = env,
    learning_rate= 0.0003,
    n_steps = 2048,
    batch_size = 64,
    n_epochs = 10,
    gamma = 0.99,
    verbose = 1,
    tensorboard_log= "./tensorboard_logs/"
) """

# cargar PPO_15 y continuar con curriculum dirigido a estanterías difíciles
model = PPO.load(
    "./pruebas/ppo_sthwp_15",
    env=env,
    learning_rate=0.0001,
    tensorboard_log="./tensorboard_logs/"
)

checkpoint_cb = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = "./pruebas/checkpoints_sthwp_17/",
    name_prefix = "ppo_sthwp_17"
)

model.learn(
    total_timesteps=1_000_000,
    callback = [stats_cb, checkpoint_cb]
)

model.save("./pruebas/ppo_sthwp_17")

# pausar la simulacion
env.supervisor.simulationSetMode(env.supervisor.SIMULATION_MODE_PAUSE)