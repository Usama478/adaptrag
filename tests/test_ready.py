import logging
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from adaptrag.config import Settings, get_settings
from adaptrag.main import create_app

UNREACHABLE_DB = "postgresql+psycopg://nobody:nothing@localhost:1/nothing"


def client_with_database(database_url: str) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(database_url=database_url)
    return TestClient(app)


def test_ready_returns_200_when_database_is_up(make_client: Callable[..., TestClient]) -> None:
    response = make_client().get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": "ok"}


def test_ready_returns_503_when_database_is_down() -> None:
    response = client_with_database(UNREACHABLE_DB).get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "Database is not reachable."


def test_ready_logs_the_reason_when_database_is_down(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="adaptrag"):
        client_with_database(UNREACHABLE_DB).get("/ready")
    assert "Readiness check failed" in caplog.text
