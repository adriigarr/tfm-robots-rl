import time

from fastapi.testclient import TestClient

from models import OrderStatus


def test_get_products_returns_catalog(app):
    with TestClient(app) as client:
        res = client.get("/api/products")
        assert res.status_code == 200
        skus = {p["sku"] for p in res.json()}
        assert skus == {"T01", "T02"}


def test_get_products_incluye_ubicacion(app):
    with TestClient(app) as client:
        res = client.get("/api/products")
        assert res.status_code == 200
        for producto in res.json():
            assert "ubicacion" in producto
            assert isinstance(producto["ubicacion"], str) and producto["ubicacion"]


def test_create_order_valid_maps_sku_to_goal_id(app):
    with TestClient(app) as client:
        res = client.post("/api/orders", json={"sku": "T01"})
        assert res.status_code == 201
        body = res.json()
        assert body["sku"] == "T01"
        assert body["goal_id"] == "goal_t01"
        assert body["status"] == "en_cola"
        assert "order_id" in body


def test_create_order_sku_inexistente(app):
    with TestClient(app) as client:
        res = client.post("/api/orders", json={"sku": "NO_EXISTE"})
        assert res.status_code == 404


def test_create_order_sin_stock(app):
    with TestClient(app) as client:
        res = client.post("/api/orders", json={"sku": "T02"})
        assert res.status_code == 409


def test_create_order_rechaza_segundo_pedido_activo(app):
    with TestClient(app) as client:
        first = client.post("/api/orders", json={"sku": "T01"})
        assert first.status_code == 201
        second = client.post("/api/orders", json={"sku": "T01"})
        assert second.status_code == 409


def test_create_order_rechaza_goal_id_arbitrario(app):
    with TestClient(app) as client:
        res = client.post("/api/orders", json={"sku": "T01", "goal_id": "goal_falso"})
        assert res.status_code == 422


def test_create_order_rechaza_coordenadas_o_comandos(app):
    with TestClient(app) as client:
        res = client.post(
            "/api/orders",
            json={"sku": "T01", "x": 1.0, "y": 2.0, "command": "goto"},
        )
        assert res.status_code == 422


def test_get_order_inexistente(app):
    with TestClient(app) as client:
        res = client.get("/api/orders/no-existe")
        assert res.status_code == 404


def test_pedido_avanza_hasta_completado_y_libera_robot(app):
    with TestClient(app) as client:
        created = client.post("/api/orders", json={"sku": "T01"}).json()
        order_id = created["order_id"]

        deadline = time.time() + 5
        status = created["status"]
        while status != OrderStatus.COMPLETADO.value and time.time() < deadline:
            time.sleep(0.02)
            status = client.get(f"/api/orders/{order_id}").json()["status"]

        assert status == OrderStatus.COMPLETADO.value

        # con el pedido completado, el robot vuelve a estar disponible
        second = client.post("/api/orders", json={"sku": "T01"})
        assert second.status_code == 201
