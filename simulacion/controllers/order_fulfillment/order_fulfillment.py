"""order_fulfillment.py

Controlador Webots que ejecuta UN pedido de la interfaz (`interfaz/backend`)
en el almacén de referencia WH01 (mapa `warehouse_map01`). Es el contrapunto
real de `interfaz/backend/execution.py::MockExecutionService`: en vez de
simular los estados con temporizadores, ejecuta un episodio real de
aproximación → recogida → entrega → retorno con la política SUB-WP entrenada.

Archivo NUEVO — no modifica ningún controlador de entrenamiento/evaluación
existente. Reutiliza `WebotsEnv`/`global_planner`/`heading_utils` de
`rl_train_SUB_WP_continuo/` importándolos (mismo patrón que ya usa
`controllers/inference/inference.py` con `rl_train`), sin tocar esos
archivos.

Parametrización (variables de entorno, fijadas por
`WebotsExecutionService.run()` antes de lanzar Webots):
  ORDER_GOAL_ID     — goal_id del pedido (p.ej. "goal_03"), obligatorio.
  ORDER_STATUS_FILE — ruta absoluta del archivo JSON de estado a escribir,
                      obligatorio. Se escribe de forma atómica
                      (fichero temporal + os.replace) para que un lector
                      externo (el backend, sondeando) nunca vea un JSON a
                      medio escribir.
  ORDER_MODEL_PATH  — ruta al checkpoint PPO (.zip) a cargar, obligatorio.
  ORDER_MAX_STEPS   — límite de pasos del episodio (por defecto 6000, igual
                      que la evaluación de generalización SUBWP).

Los valores de "status" escritos son literales de
`interfaz/backend/models.py::OrderStatus` (en_cola/iniciando_simulacion los
pone el backend antes de que este proceso exista; aquí solo se escriben los
que ocurren dentro de la simulación). Se duplican como cadenas planas en
vez de importar el paquete `interfaz` para no acoplar este controlador
(que corre con el intérprete/venv de Webots) al entorno virtual del
backend — si cambian los nombres de estado en `models.py`, hay que
actualizarlos aquí también.
"""

import json
import os
import sys
import time
import traceback

CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
SUBWP_DIR = os.path.abspath(os.path.join(CONTROLLER_DIR, "..", "rl_train_SUB_WP_continuo"))
sys.path.insert(0, SUBWP_DIR)

from webots_env import WebotsEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

MAP_PATH = os.path.join(SUBWP_DIR, "warehouse_map01.json")

# Pausa real (segundos) en los puntos de recogida/entrega: además de dar
# tiempo a que el backend (que sondea el archivo de estado) capture cada
# estado intermedio sin perderlo entre dos escrituras seguidas, representa
# de forma simple el tiempo de manipulación del producto.
PICKUP_PAUSE_S = 1.0

STATUS_FILE = os.environ["ORDER_STATUS_FILE"]


def write_status(status: str, progress: int) -> None:
    tmp_path = STATUS_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"status": status, "progress": progress}, f)
    os.replace(tmp_path, STATUS_FILE)


def run_order():
    """Ejecuta el episodio del pedido. Devuelve el `WebotsEnv` construido (para
    que el llamador pueda cerrar Webots con `env.supervisor.simulationQuit()`),
    o `None` si no se llegó a construir el entorno."""
    goal_id = os.environ["ORDER_GOAL_ID"]
    model_path = os.environ["ORDER_MODEL_PATH"]
    max_steps = int(os.environ.get("ORDER_MAX_STEPS", 6000))

    env = WebotsEnv(map_path=MAP_PATH, stage=6, ped_obs=False)
    env._max_steps = max_steps

    if goal_id not in env.goal_ids:
        write_status("error_ejecucion", 0)
        return env

    goal_idx = env.goal_ids.index(goal_id)
    # Fuerza el episodio a ESTE goal en vez de uno aleatorio (mismo mecanismo
    # que usa experimentos/scripts/subwp_eval_generalizacion.py:97 para la
    # evaluación de generalización).
    env._sample_goal_approach = lambda gi=goal_idx: gi

    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    write_status("hacia_estanteria", 15)

    prev_hacia_descarga = False
    prev_en_retorno = False
    terminated = truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)

        if terminated and info.get("colision"):
            write_status("colision", 0)
            return env
        if terminated and info.get("exito"):
            write_status("completado", 100)
            return env
        if truncated and not terminated:
            write_status("truncado", 0)
            return env

        if env._hacia_descarga and not prev_hacia_descarga:
            write_status("recogiendo", 45)
            time.sleep(PICKUP_PAUSE_S)
            write_status("hacia_entrega", 55)
        if env._en_retorno and not prev_en_retorno:
            write_status("entregado", 80)
            time.sleep(PICKUP_PAUSE_S)
            write_status("retornando", 90)
        prev_hacia_descarga = env._hacia_descarga
        prev_en_retorno = env._en_retorno

    # Bucle terminado sin colisión/éxito/truncamiento explícitos: no debería
    # ocurrir dado el contrato de WebotsEnv.step(), pero se cubre por si acaso.
    write_status("error_ejecucion", 0)
    return env


if __name__ == "__main__":
    _env = None
    try:
        _env = run_order()
    except Exception:
        traceback.print_exc()
        try:
            write_status("error_ejecucion", 0)
        except OSError:
            pass
    finally:
        # Cierra Webots al terminar, igual que el resto de controladores del
        # repo (p.ej. subwp_eval_generalizacion.py:129) — el proceso lanzado
        # por WebotsExecutionService.run() debe terminar por sí solo. Si
        # WebotsEnv() falla a mitad de construir (antes de tener `_env`), el
        # proceso Webots queda abierto; lo cubre el timeout/kill de
        # WebotsExecutionService en el backend.
        if _env is not None:
            _env.supervisor.simulationQuit(0)
