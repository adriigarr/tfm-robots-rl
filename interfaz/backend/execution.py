"""Servicio de ejecución desacoplado del backend de pedidos.

`ExecutionService` es la frontera entre la API de pedidos y "quien mueve al
robot de verdad". `MockExecutionService` simula los estados en segundo
plano; `WebotsExecutionService` lanza una simulación real de Webots para el
almacén WH01. Ninguna de las dos toca la API, el modelo de pedido ni el
frontend.
"""

import asyncio
import json
import os
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Awaitable, Callable

from models import TERMINAL_STATES, OrderStatus

UpdateCallback = Callable[[OrderStatus, int], Awaitable[None]]

# Secuencia de progreso del trayecto simulado tras "en_cola".
MOCK_STEP_SEQUENCE: list[tuple[OrderStatus, int]] = [
    (OrderStatus.INICIANDO_SIMULACION, 10),
    (OrderStatus.HACIA_ESTANTERIA, 30),
    (OrderStatus.RECOGIENDO, 50),
    (OrderStatus.HACIA_ENTREGA, 70),
    (OrderStatus.ENTREGADO, 85),
    (OrderStatus.RETORNANDO, 95),
    (OrderStatus.COMPLETADO, 100),
]


class ExecutionService(ABC):
    """Ejecuta el trayecto de un pedido y notifica cada cambio de estado."""

    @abstractmethod
    async def run(self, order_id: str, goal_id: str, on_update: UpdateCallback) -> None:
        """Recorre los estados del pedido llamando a `on_update(status, progress)`
        por cada uno. No debe bloquear el hilo que la invoca (se lanza como
        tarea en segundo plano)."""
        raise NotImplementedError


class MockExecutionService(ExecutionService):
    """Simula el ciclo completo del robot sin tocar Webots ni hardware real.

    `fail_at`: si se indica un estado de `MOCK_STEP_SEQUENCE`, el trayecto se
    interrumpe justo antes de alcanzarlo y termina en `fail_status`. Pensado
    para pruebas del camino de error.
    """

    def __init__(
        self,
        step_delay: float = 0.4,
        fail_at: OrderStatus | None = None,
        fail_status: OrderStatus = OrderStatus.ERROR_EJECUCION,
    ):
        self.step_delay = step_delay
        self.fail_at = fail_at
        self.fail_status = fail_status

    async def run(self, order_id: str, goal_id: str, on_update: UpdateCallback) -> None:
        for status, progress in MOCK_STEP_SEQUENCE:
            await asyncio.sleep(self.step_delay)
            if self.fail_at is not None and status == self.fail_at:
                await on_update(self.fail_status, progress)
                return
            await on_update(status, progress)


class WebotsExecutionServiceDisabledError(RuntimeError):
    """La configuración de la integración con Webots es inválida o incompleta
    (binario, mundo o modelo inexistentes) — no se ha llegado a lanzar nada."""


# Rutas por defecto, relativas a la raíz del repo (interfaz/backend/execution.py
# está dos niveles por debajo de ella).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SIMULACION_DIR = _REPO_ROOT / "simulacion"
DEFAULT_WEBOTS_BIN = "/Applications/Webots.app/Contents/MacOS/webots"
DEFAULT_WORLD_PATH = _SIMULACION_DIR / "worlds" / "warehouse_wh01_order.wbt"
DEFAULT_MODEL_PATH = (
    _SIMULACION_DIR
    / "controllers"
    / "rl_train_SUB_WP_continuo"
    / "pruebas"
    / "subwp_e2_1_s42_final.zip"
)

# Traduce el "status" que escribe el controlador Webots
# (simulacion/controllers/order_fulfillment/order_fulfillment.py) al enum
# OrderStatus. Se mantiene como diccionario explícito (en vez de
# OrderStatus(status_str)) para no aceptar a ciegas cualquier cadena que
# aparezca en el archivo de estado.
_STATUS_FILE_TO_ORDER_STATUS: dict[str, OrderStatus] = {
    "hacia_estanteria": OrderStatus.HACIA_ESTANTERIA,
    "recogiendo": OrderStatus.RECOGIENDO,
    "hacia_entrega": OrderStatus.HACIA_ENTREGA,
    "entregado": OrderStatus.ENTREGADO,
    "retornando": OrderStatus.RETORNANDO,
    "completado": OrderStatus.COMPLETADO,
    "colision": OrderStatus.COLISION,
    "truncado": OrderStatus.TRUNCADO,
    "error_ejecucion": OrderStatus.ERROR_EJECUCION,
}


class WebotsExecutionService(ExecutionService):
    """Lanza una simulación Webots real (un proceso por pedido, igual que el
    resto de scripts de evaluación del repo: `webots --mode=fast
    --no-rendering --minimize <mundo>`) para el almacén WH01. El controlador
    `simulacion/controllers/order_fulfillment/order_fulfillment.py` ejecuta
    el episodio con la política SUB-WP y reporta el progreso escribiendo un
    archivo JSON de estado, que este servicio sondea.

    No lanza nada por sí sola: solo se instancia cuando `EXECUTION_MODE=webots`
    (ver `get_execution_service()`), que nunca es el valor por defecto.
    """

    def __init__(
        self,
        webots_bin: str | None = None,
        world_path: str | Path | None = None,
        model_path: str | Path | None = None,
        max_steps: int | None = None,
        poll_interval: float | None = None,
        timeout: float | None = None,
        render: bool | None = None,
    ):
        self.webots_bin = webots_bin or os.environ.get("WEBOTS_BIN", DEFAULT_WEBOTS_BIN)
        self.world_path = Path(world_path or os.environ.get("WEBOTS_WORLD_WH01", DEFAULT_WORLD_PATH))
        self.model_path = Path(model_path or os.environ.get("WEBOTS_MODEL_PATH", DEFAULT_MODEL_PATH))
        self.max_steps = int(max_steps or os.environ.get("WEBOTS_MAX_STEPS", 6000))
        self.poll_interval = float(poll_interval or os.environ.get("WEBOTS_STATUS_POLL_INTERVAL", 0.3))
        self.timeout = float(timeout or os.environ.get("WEBOTS_LAUNCH_TIMEOUT", 600.0))
        # Solo para depuración manual: con rendering se ve el robot moverse,
        # pero la simulación va bastante más lenta que en --mode=fast
        # --no-rendering (el modo pensado para uso real de la interfaz).
        self.render = render if render is not None else os.environ.get("WEBOTS_RENDER", "0") == "1"
        self._check_config()

    def _check_config(self) -> None:
        if not os.path.exists(self.webots_bin):
            raise WebotsExecutionServiceDisabledError(f"No se encuentra el binario de Webots: {self.webots_bin}")
        if not self.world_path.exists():
            raise WebotsExecutionServiceDisabledError(f"No se encuentra el mundo de WH01: {self.world_path}")
        if not self.model_path.exists():
            raise WebotsExecutionServiceDisabledError(f"No se encuentra el modelo PPO: {self.model_path}")

    async def run(self, order_id: str, goal_id: str, on_update: UpdateCallback) -> None:
        # Cubre el arranque de Webots (carga del mundo, del modelo PPO, del
        # primer reset) antes de que el controlador escriba su primer
        # estado. Necesario también para que la máquina de estados sea
        # coherente: EN_COLA solo admite pasar a INICIANDO_SIMULACION, nunca
        # directamente a HACIA_ESTANTERIA (que es lo primero que escribe
        # order_fulfillment.py).
        await on_update(OrderStatus.INICIANDO_SIMULACION, 5)

        with tempfile.TemporaryDirectory(prefix=f"order_{order_id}_") as tmp_dir:
            status_path = Path(tmp_dir) / "status.json"
            env = {
                **os.environ,
                "ORDER_GOAL_ID": goal_id,
                "ORDER_STATUS_FILE": str(status_path),
                "ORDER_MODEL_PATH": str(self.model_path),
                "ORDER_MAX_STEPS": str(self.max_steps),
            }

            args = [self.webots_bin, "--mode=fast"]
            if not self.render:
                args += ["--no-rendering", "--minimize"]
            args.append(str(self.world_path))

            process = await asyncio.create_subprocess_exec(
                *args,
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await self._poll_until_terminal(status_path, process, on_update)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

    async def _poll_until_terminal(self, status_path: Path, process, on_update: UpdateCallback) -> None:
        last_status_str: str | None = None
        deadline = time.monotonic() + self.timeout

        while True:
            status_str, progress = self._read_status_file(status_path)
            if status_str is not None and status_str != last_status_str:
                order_status = _STATUS_FILE_TO_ORDER_STATUS.get(status_str)
                if order_status is not None:
                    await on_update(order_status, progress)
                    last_status_str = status_str
                    if order_status in TERMINAL_STATES:
                        return

            if process.returncode is not None:
                # Webots terminó sin dejar escrito un estado final: se trata
                # como fallo de ejecución en vez de dejar el pedido colgado.
                await on_update(OrderStatus.ERROR_EJECUCION, progress or 0)
                return

            if time.monotonic() > deadline:
                process.kill()
                await on_update(OrderStatus.ERROR_EJECUCION, progress or 0)
                return

            await asyncio.sleep(self.poll_interval)

    @staticmethod
    def _read_status_file(status_path: Path) -> tuple[str | None, int]:
        try:
            data = json.loads(status_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None, 0
        return data.get("status"), int(data.get("progress", 0))


def get_execution_service() -> ExecutionService:
    """Fábrica controlada por la variable de entorno EXECUTION_MODE.
    Por defecto siempre es 'mock': nada abre Webots automáticamente."""
    mode = os.environ.get("EXECUTION_MODE", "mock").strip().lower()
    if mode == "mock":
        return MockExecutionService()
    if mode == "webots":
        return WebotsExecutionService()
    raise ValueError(f"EXECUTION_MODE desconocido: {mode!r} (usa 'mock' o 'webots')")
