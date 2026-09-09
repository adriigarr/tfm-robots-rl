"""Pruebas de WebotsExecutionService SIN lanzar Webots real: se mockea
`asyncio.create_subprocess_exec` para aislar la lógica de sondeo del
archivo de estado y su traducción a OrderStatus."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from execution import WebotsExecutionService, WebotsExecutionServiceDisabledError
from models import OrderStatus


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.killed = False

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


@pytest.fixture
def webots_paths(tmp_path):
    webots_bin = tmp_path / "webots"
    world_path = tmp_path / "warehouse_wh01_order.wbt"
    model_path = tmp_path / "model.zip"
    for p in (webots_bin, world_path, model_path):
        p.write_text("")
    return webots_bin, world_path, model_path


def _write_status_atomic(path: Path, status: str, progress: int) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"status": status, "progress": progress}, f)
    os.replace(tmp, path)


def test_check_config_rechaza_binario_webots_inexistente(webots_paths, tmp_path):
    _, world_path, model_path = webots_paths
    with pytest.raises(WebotsExecutionServiceDisabledError):
        WebotsExecutionService(
            webots_bin=str(tmp_path / "no_existe"), world_path=world_path, model_path=model_path
        )


def test_check_config_rechaza_mundo_inexistente(webots_paths, tmp_path):
    webots_bin, _, model_path = webots_paths
    with pytest.raises(WebotsExecutionServiceDisabledError):
        WebotsExecutionService(
            webots_bin=webots_bin, world_path=tmp_path / "no_existe.wbt", model_path=model_path
        )


def test_check_config_rechaza_modelo_inexistente(webots_paths, tmp_path):
    webots_bin, world_path, _ = webots_paths
    with pytest.raises(WebotsExecutionServiceDisabledError):
        WebotsExecutionService(
            webots_bin=webots_bin, world_path=world_path, model_path=tmp_path / "no_existe.zip"
        )


def test_run_traduce_la_secuencia_completa_hasta_completado(monkeypatch, webots_paths):
    webots_bin, world_path, model_path = webots_paths
    secuencia = [
        ("hacia_estanteria", 15),
        ("recogiendo", 45),
        ("hacia_entrega", 55),
        ("entregado", 80),
        ("retornando", 90),
        ("completado", 100),
    ]

    async def fake_create_subprocess_exec(*args, **kwargs):
        status_path = Path(kwargs["env"]["ORDER_STATUS_FILE"])

        async def writer():
            for status, progress in secuencia:
                await asyncio.sleep(0.01)
                _write_status_atomic(status_path, status, progress)

        asyncio.create_task(writer())
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    service = WebotsExecutionService(
        webots_bin=str(webots_bin), world_path=world_path, model_path=model_path,
        poll_interval=0.005, timeout=5,
    )

    updates = []

    async def on_update(status, progress):
        updates.append(status)

    asyncio.run(service.run("order-1", "goal_01", on_update))

    assert updates == [
        OrderStatus.INICIANDO_SIMULACION,
        OrderStatus.HACIA_ESTANTERIA,
        OrderStatus.RECOGIENDO,
        OrderStatus.HACIA_ENTREGA,
        OrderStatus.ENTREGADO,
        OrderStatus.RETORNANDO,
        OrderStatus.COMPLETADO,
    ]


def test_run_colision_detiene_el_sondeo(monkeypatch, webots_paths):
    webots_bin, world_path, model_path = webots_paths

    async def fake_create_subprocess_exec(*args, **kwargs):
        status_path = Path(kwargs["env"]["ORDER_STATUS_FILE"])

        async def writer():
            await asyncio.sleep(0.01)
            _write_status_atomic(status_path, "hacia_estanteria", 15)
            await asyncio.sleep(0.01)
            _write_status_atomic(status_path, "colision", 0)

        asyncio.create_task(writer())
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    service = WebotsExecutionService(
        webots_bin=str(webots_bin), world_path=world_path, model_path=model_path,
        poll_interval=0.005, timeout=5,
    )

    updates = []

    async def on_update(status, progress):
        updates.append(status)

    asyncio.run(service.run("order-1", "goal_01", on_update))

    assert updates == [OrderStatus.INICIANDO_SIMULACION, OrderStatus.HACIA_ESTANTERIA, OrderStatus.COLISION]


def test_run_proceso_termina_sin_estado_final_es_error_ejecucion(monkeypatch, webots_paths):
    webots_bin, world_path, model_path = webots_paths

    async def fake_create_subprocess_exec(*args, **kwargs):
        process = FakeProcess()

        async def crash_soon():
            await asyncio.sleep(0.01)
            process.returncode = 1  # el proceso Webots murió sin escribir nada

        asyncio.create_task(crash_soon())
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    service = WebotsExecutionService(
        webots_bin=str(webots_bin), world_path=world_path, model_path=model_path,
        poll_interval=0.005, timeout=5,
    )

    updates = []

    async def on_update(status, progress):
        updates.append(status)

    asyncio.run(service.run("order-1", "goal_01", on_update))

    assert updates == [OrderStatus.INICIANDO_SIMULACION, OrderStatus.ERROR_EJECUCION]


def test_run_supera_el_timeout_mata_el_proceso(monkeypatch, webots_paths):
    webots_bin, world_path, model_path = webots_paths
    fake_process_holder = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        process = FakeProcess()
        fake_process_holder["process"] = process
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    service = WebotsExecutionService(
        webots_bin=str(webots_bin), world_path=world_path, model_path=model_path,
        poll_interval=0.005, timeout=0.02,
    )

    updates = []

    async def on_update(status, progress):
        updates.append(status)

    asyncio.run(service.run("order-1", "goal_01", on_update))

    assert updates == [OrderStatus.INICIANDO_SIMULACION, OrderStatus.ERROR_EJECUCION]
    assert fake_process_holder["process"].killed
