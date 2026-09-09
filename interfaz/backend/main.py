import asyncio
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from execution import ExecutionService, get_execution_service
from models import OrderRequest, OrderResponse, OrderStatus, Producto, RobotStatusResponse
from store import (
    OrderNotFoundError,
    OrderStore,
    OutOfStockError,
    ProductNotFoundError,
    RobotBusyError,
    RobotNeedsReviewError,
    RobotNotInReviewError,
)

CATALOG_PATH = Path(__file__).parent / "catalog.json"


def create_app(store: OrderStore | None = None, execution_service: ExecutionService | None = None) -> FastAPI:
    app = FastAPI(title="Almacén - Catálogo de pedidos")
    app.state.store = store or OrderStore(CATALOG_PATH)
    app.state.execution_service = execution_service or get_execution_service()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_store() -> OrderStore:
        return app.state.store

    def get_service() -> ExecutionService:
        return app.state.execution_service

    @app.get("/api/products", response_model=list[Producto])
    def get_products():
        return get_store().listar_productos()

    @app.post("/api/orders", response_model=OrderResponse, status_code=201)
    async def create_order(req: OrderRequest):
        store = get_store()
        try:
            order = store.crear_pedido(req.sku)
        except ProductNotFoundError:
            raise HTTPException(status_code=404, detail="Producto no encontrado en el catálogo")
        except OutOfStockError:
            raise HTTPException(status_code=409, detail="Producto sin existencias")
        except RobotBusyError:
            raise HTTPException(status_code=409, detail="El robot ya tiene un pedido activo")
        except RobotNeedsReviewError:
            raise HTTPException(
                status_code=409,
                detail="El robot requiere revisión tras un incidente. Reinícialo antes de aceptar nuevos pedidos.",
            )

        asyncio.create_task(_run_execution(order["order_id"], order["goal_id"], store, get_service()))
        return order

    @app.get("/api/orders/{order_id}", response_model=OrderResponse)
    def get_order(order_id: str):
        try:
            return get_store().get_order(order_id)
        except OrderNotFoundError:
            raise HTTPException(status_code=404, detail="Pedido no encontrado")

    @app.get("/api/robot/status", response_model=RobotStatusResponse)
    def get_robot_status():
        return {"status": get_store().robot_status}

    @app.post("/api/robot/reset", response_model=RobotStatusResponse)
    def reset_robot():
        try:
            status = get_store().reset_robot()
        except RobotNotInReviewError:
            raise HTTPException(status_code=409, detail="El robot no requiere revisión actualmente")
        return {"status": status}

    return app


async def _run_execution(order_id: str, goal_id: str, store: OrderStore, service: ExecutionService) -> None:
    async def on_update(status: OrderStatus, progress: int) -> None:
        store.transition(order_id, status, progress)

    try:
        await service.run(order_id, goal_id, on_update)
    except Exception:
        # Cualquier fallo no anticipado por la máquina de estados (incluida
        # una integración Webots deshabilitada) deja el pedido en error y
        # libera el robot, en lugar de dejarlo bloqueado indefinidamente.
        # Se registra siempre: sin esto, un ExecutionService que lanza una
        # excepción (p.ej. una transición inválida) se convertía en
        # "error_ejecucion" sin dejar ningún rastro en los logs.
        traceback.print_exc()
        store.force_error(order_id, OrderStatus.ERROR_EJECUCION)


app = create_app()
