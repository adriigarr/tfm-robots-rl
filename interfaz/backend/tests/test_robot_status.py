import time

from fastapi.testclient import TestClient

from models import OrderStatus, RobotStatus


def _esperar_estado(client, order_id, estado_final, timeout=5):
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/orders/{order_id}").json()["status"]
        if status == estado_final:
            return status
        time.sleep(0.02)
    raise AssertionError(f"El pedido no alcanzó '{estado_final}' (último estado: {status})")


def test_colision_bloquea_el_robot(app_factory):
    app = app_factory(fail_at=OrderStatus.RECOGIENDO, fail_status=OrderStatus.COLISION)
    with TestClient(app) as client:
        order = client.post("/api/orders", json={"sku": "T01"}).json()
        _esperar_estado(client, order["order_id"], OrderStatus.COLISION.value)

        status = client.get("/api/robot/status").json()
        assert status["status"] == RobotStatus.REQUIERE_REVISION.value


def test_truncado_bloquea_el_robot(app_factory):
    app = app_factory(fail_at=OrderStatus.RECOGIENDO, fail_status=OrderStatus.TRUNCADO)
    with TestClient(app) as client:
        order = client.post("/api/orders", json={"sku": "T01"}).json()
        _esperar_estado(client, order["order_id"], OrderStatus.TRUNCADO.value)

        status = client.get("/api/robot/status").json()
        assert status["status"] == RobotStatus.REQUIERE_REVISION.value


def test_rechaza_nuevos_pedidos_mientras_requiere_revision(app_factory):
    app = app_factory(fail_at=OrderStatus.RECOGIENDO, fail_status=OrderStatus.COLISION)
    with TestClient(app) as client:
        order = client.post("/api/orders", json={"sku": "T01"}).json()
        _esperar_estado(client, order["order_id"], OrderStatus.COLISION.value)

        res = client.post("/api/orders", json={"sku": "T01"})
        assert res.status_code == 409


def test_reset_recupera_disponibilidad_del_robot(app_factory):
    app = app_factory(fail_at=OrderStatus.RECOGIENDO, fail_status=OrderStatus.COLISION)
    with TestClient(app) as client:
        order = client.post("/api/orders", json={"sku": "T01"}).json()
        _esperar_estado(client, order["order_id"], OrderStatus.COLISION.value)

        reset_res = client.post("/api/robot/reset")
        assert reset_res.status_code == 200
        assert reset_res.json()["status"] == RobotStatus.DISPONIBLE.value

        nuevo = client.post("/api/orders", json={"sku": "T01"})
        assert nuevo.status_code == 201


def test_reset_sin_revision_pendiente_devuelve_409(app):
    with TestClient(app) as client:
        res = client.post("/api/robot/reset")
        assert res.status_code == 409


def test_pedido_completado_deja_robot_disponible_sin_necesidad_de_reset(app):
    with TestClient(app) as client:
        order = client.post("/api/orders", json={"sku": "T01"}).json()
        _esperar_estado(client, order["order_id"], OrderStatus.COMPLETADO.value)

        status = client.get("/api/robot/status").json()
        assert status["status"] == RobotStatus.DISPONIBLE.value

        nuevo = client.post("/api/orders", json={"sku": "T01"})
        assert nuevo.status_code == 201
