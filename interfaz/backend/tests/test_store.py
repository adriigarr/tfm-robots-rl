import os

import pytest

from execution import MockExecutionService, WebotsExecutionService, get_execution_service
from models import OrderStatus
from store import InvalidTransitionError, OutOfStockError, ProductNotFoundError, RobotBusyError


def test_crear_pedido_sku_inexistente(store):
    with pytest.raises(ProductNotFoundError):
        store.crear_pedido("NO_EXISTE")


def test_crear_pedido_sin_stock(store):
    with pytest.raises(OutOfStockError):
        store.crear_pedido("T02")


def test_crear_pedido_robot_ocupado(store):
    store.crear_pedido("T01")
    with pytest.raises(RobotBusyError):
        store.crear_pedido("T01")


def test_transition_rechaza_salto_incoherente(store):
    order = store.crear_pedido("T01")
    with pytest.raises(InvalidTransitionError):
        store.transition(order["order_id"], OrderStatus.COMPLETADO, 100)


def test_transition_valida_avanza_estado(store):
    order = store.crear_pedido("T01")
    updated = store.transition(order["order_id"], OrderStatus.INICIANDO_SIMULACION, 10)
    assert updated["status"] == OrderStatus.INICIANDO_SIMULACION


def test_execution_mode_por_defecto_es_mock(monkeypatch):
    monkeypatch.delenv("EXECUTION_MODE", raising=False)
    assert isinstance(get_execution_service(), MockExecutionService)


def test_execution_mode_webots_selecciona_servicio_real(monkeypatch, tmp_path):
    webots_bin = tmp_path / "webots"
    world_path = tmp_path / "warehouse_wh01_order.wbt"
    model_path = tmp_path / "model.zip"
    for p in (webots_bin, world_path, model_path):
        p.write_text("")

    monkeypatch.setenv("EXECUTION_MODE", "webots")
    monkeypatch.setenv("WEBOTS_BIN", str(webots_bin))
    monkeypatch.setenv("WEBOTS_WORLD_WH01", str(world_path))
    monkeypatch.setenv("WEBOTS_MODEL_PATH", str(model_path))
    assert isinstance(get_execution_service(), WebotsExecutionService)
