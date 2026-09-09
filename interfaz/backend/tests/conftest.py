import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution import MockExecutionService  # noqa: E402
from main import create_app  # noqa: E402
from store import OrderStore  # noqa: E402

TEST_CATALOG = {
    "map_id": "warehouse_test",
    "productos": [
        {
            "sku": "T01",
            "nombre": "Producto con stock",
            "categoria": "test",
            "icono": "📦",
            "stock": 10,
            "goal_id": "goal_t01",
            "ubicacion": "A-01-01",
            "desc": "Producto de prueba con existencias.",
        },
        {
            "sku": "T02",
            "nombre": "Producto sin stock",
            "categoria": "test",
            "icono": "📦",
            "stock": 0,
            "goal_id": "goal_t02",
            "ubicacion": "A-01-02",
            "desc": "Producto de prueba agotado.",
        },
    ],
}


@pytest.fixture
def catalog_path(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(TEST_CATALOG))
    return path


@pytest.fixture
def store(catalog_path):
    return OrderStore(catalog_path)


@pytest.fixture
def app(catalog_path):
    return create_app(
        store=OrderStore(catalog_path),
        execution_service=MockExecutionService(step_delay=0.01),
    )


@pytest.fixture
def app_factory(catalog_path):
    """Fábrica de apps con un MockExecutionService configurable, para poder
    forzar el camino de fallo (colision/truncado) en las pruebas."""

    def _make(**mock_kwargs):
        mock_kwargs.setdefault("step_delay", 0.01)
        return create_app(
            store=OrderStore(catalog_path),
            execution_service=MockExecutionService(**mock_kwargs),
        )

    return _make
