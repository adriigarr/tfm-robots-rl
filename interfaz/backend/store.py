"""Estado en memoria: catálogo cargado de disco, pedidos activos/históricos y
estado operativo del robot.

Deliberadamente simple (dict en memoria) porque solo hay un robot y como
mucho un pedido activo a la vez; no hace falta una base de datos para esto.
"""

import json
import uuid
from pathlib import Path

from models import STATES_REQUIRING_REVIEW, TERMINAL_STATES, TRANSITIONS, OrderStatus, RobotStatus


class InvalidTransitionError(Exception):
    pass


class ProductNotFoundError(Exception):
    pass


class OutOfStockError(Exception):
    pass


class RobotBusyError(Exception):
    """El robot ya tiene un pedido activo (`RobotStatus.OCUPADO`)."""


class RobotNeedsReviewError(Exception):
    """El robot está bloqueado tras un incidente (`RobotStatus.REQUIERE_REVISION`)."""


class RobotNotInReviewError(Exception):
    """Se intentó reiniciar el robot sin que esté en `requiere_revision`."""


class OrderNotFoundError(Exception):
    pass


class OrderStore:
    def __init__(self, catalog_path: Path):
        data = json.loads(Path(catalog_path).read_text())
        self.productos: dict[str, dict] = {p["sku"]: p for p in data["productos"]}
        self.orders: dict[str, dict] = {}
        self.robot_status: RobotStatus = RobotStatus.DISPONIBLE

    def get_producto(self, sku: str) -> dict | None:
        return self.productos.get(sku)

    def listar_productos(self) -> list[dict]:
        return list(self.productos.values())

    def get_order(self, order_id: str) -> dict:
        order = self.orders.get(order_id)
        if order is None:
            raise OrderNotFoundError(order_id)
        return order

    def crear_pedido(self, sku: str) -> dict:
        producto = self.get_producto(sku)
        if producto is None:
            raise ProductNotFoundError(sku)
        if producto["stock"] <= 0:
            raise OutOfStockError(sku)
        if self.robot_status == RobotStatus.REQUIERE_REVISION:
            raise RobotNeedsReviewError()
        if self.robot_status == RobotStatus.OCUPADO:
            raise RobotBusyError()

        order_id = str(uuid.uuid4())
        order = {
            "order_id": order_id,
            "sku": producto["sku"],
            "nombre": producto["nombre"],
            "goal_id": producto["goal_id"],
            "status": OrderStatus.EN_COLA,
            "progress": 0,
        }
        self.orders[order_id] = order
        self.robot_status = RobotStatus.OCUPADO
        return order

    def transition(self, order_id: str, new_status: OrderStatus, progress: int) -> dict:
        order = self.get_order(order_id)
        current = OrderStatus(order["status"])
        if new_status not in TRANSITIONS.get(current, set()):
            raise InvalidTransitionError(f"{current} -> {new_status} no permitido")
        order["status"] = new_status
        order["progress"] = progress
        self._sync_robot_status(new_status)
        return order

    def force_error(self, order_id: str, status: OrderStatus = OrderStatus.ERROR_EJECUCION) -> dict:
        """Vía de escape para fallos no anticipados por la máquina de estados
        (p.ej. una excepción inesperada del servicio de ejecución)."""
        order = self.get_order(order_id)
        order["status"] = status
        self._sync_robot_status(status)
        return order

    def reset_robot(self) -> RobotStatus:
        """Confirmación explícita del operador tras una revisión. No implica
        ninguna recuperación física: solo desbloquea el estado lógico del
        prototipo para volver a aceptar pedidos."""
        if self.robot_status != RobotStatus.REQUIERE_REVISION:
            raise RobotNotInReviewError()
        self.robot_status = RobotStatus.DISPONIBLE
        return self.robot_status

    def _sync_robot_status(self, order_status: OrderStatus) -> None:
        if order_status in STATES_REQUIRING_REVIEW:
            self.robot_status = RobotStatus.REQUIERE_REVISION
        elif order_status in TERMINAL_STATES:
            self.robot_status = RobotStatus.DISPONIBLE
