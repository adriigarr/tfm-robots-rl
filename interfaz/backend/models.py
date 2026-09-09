from enum import Enum

from pydantic import BaseModel, ConfigDict


class OrderStatus(str, Enum):
    EN_COLA = "en_cola"
    INICIANDO_SIMULACION = "iniciando_simulacion"
    HACIA_ESTANTERIA = "hacia_estanteria"
    RECOGIENDO = "recogiendo"
    HACIA_ENTREGA = "hacia_entrega"
    ENTREGADO = "entregado"
    RETORNANDO = "retornando"
    COMPLETADO = "completado"
    ERROR_EJECUCION = "error_ejecucion"
    COLISION = "colision"
    TRUNCADO = "truncado"


TERMINAL_STATES = {
    OrderStatus.COMPLETADO,
    OrderStatus.ERROR_EJECUCION,
    OrderStatus.COLISION,
    OrderStatus.TRUNCADO,
}

# Estados que, al alcanzarse, dejan el robot bloqueado hasta una revisión
# explícita del operador (ver store.py::OrderStore.reset_robot). No implican
# ninguna recuperación física real: solo bloquean el estado lógico del
# prototipo para evitar encolar pedidos sobre un robot en un estado dudoso.
STATES_REQUIRING_REVIEW = {OrderStatus.COLISION, OrderStatus.TRUNCADO}


class RobotStatus(str, Enum):
    DISPONIBLE = "disponible"
    OCUPADO = "ocupado"
    REQUIERE_REVISION = "requiere_revision"

# Transiciones válidas desde cada estado. Cualquier salto que no figure aquí se rechaza.
TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.EN_COLA: {OrderStatus.INICIANDO_SIMULACION, OrderStatus.ERROR_EJECUCION},
    OrderStatus.INICIANDO_SIMULACION: {OrderStatus.HACIA_ESTANTERIA, OrderStatus.ERROR_EJECUCION},
    OrderStatus.HACIA_ESTANTERIA: {
        OrderStatus.RECOGIENDO,
        OrderStatus.COLISION,
        OrderStatus.ERROR_EJECUCION,
        OrderStatus.TRUNCADO,
    },
    OrderStatus.RECOGIENDO: {OrderStatus.HACIA_ENTREGA, OrderStatus.ERROR_EJECUCION, OrderStatus.TRUNCADO},
    OrderStatus.HACIA_ENTREGA: {
        OrderStatus.ENTREGADO,
        OrderStatus.COLISION,
        OrderStatus.ERROR_EJECUCION,
        OrderStatus.TRUNCADO,
    },
    OrderStatus.ENTREGADO: {OrderStatus.RETORNANDO},
    OrderStatus.RETORNANDO: {
        OrderStatus.COMPLETADO,
        OrderStatus.COLISION,
        OrderStatus.ERROR_EJECUCION,
        OrderStatus.TRUNCADO,
    },
    OrderStatus.COMPLETADO: set(),
    OrderStatus.ERROR_EJECUCION: set(),
    OrderStatus.COLISION: set(),
    OrderStatus.TRUNCADO: set(),
}


class Producto(BaseModel):
    sku: str
    nombre: str
    categoria: str
    icono: str
    stock: int
    goal_id: str
    ubicacion: str
    desc: str


class OrderRequest(BaseModel):
    """Único dato que el frontend puede enviar. `extra='forbid'` impide colar
    goal_id, coordenadas, rutas o comandos arbitrarios desde el cliente."""

    model_config = ConfigDict(extra="forbid")

    sku: str


class OrderResponse(BaseModel):
    order_id: str
    sku: str
    nombre: str
    goal_id: str
    status: OrderStatus
    progress: int


class RobotStatusResponse(BaseModel):
    status: RobotStatus
