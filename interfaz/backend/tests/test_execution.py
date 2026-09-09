import asyncio

from execution import MockExecutionService
from models import TRANSITIONS, OrderStatus


def test_mock_execution_transitions_son_validas_y_llegan_a_completado():
    service = MockExecutionService(step_delay=0)
    updates: list[OrderStatus] = []

    async def on_update(status, progress):
        updates.append(status)

    asyncio.run(service.run("o1", "goal_1", on_update))

    assert updates[-1] == OrderStatus.COMPLETADO
    assert OrderStatus.ENTREGADO in updates
    assert updates.index(OrderStatus.ENTREGADO) < updates.index(OrderStatus.RETORNANDO)

    current = OrderStatus.EN_COLA
    for status in updates:
        assert status in TRANSITIONS[current], f"{current} -> {status} no es una transición válida"
        current = status


def test_mock_execution_simula_fallo_y_se_detiene():
    service = MockExecutionService(
        step_delay=0,
        fail_at=OrderStatus.HACIA_ENTREGA,
        fail_status=OrderStatus.COLISION,
    )
    updates: list[OrderStatus] = []

    async def on_update(status, progress):
        updates.append(status)

    asyncio.run(service.run("o1", "goal_1", on_update))

    assert updates[-1] == OrderStatus.COLISION
    assert OrderStatus.HACIA_ENTREGA not in updates
    assert OrderStatus.COMPLETADO not in updates


# Las pruebas de WebotsExecutionService (validación de configuración,
# sondeo del archivo de estado, timeout) están en tests/test_webots_execution.py.
