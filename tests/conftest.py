from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text
from sqlalchemy.engine import make_url

from adaptrag.config import Settings, get_settings
from adaptrag.db import get_engine
from adaptrag.embedding.base import Embedder
from adaptrag.embedding.fake import FakeEmbedder
from adaptrag.embedding.provider import get_app_embedder
from adaptrag.main import create_app
from adaptrag.models import EMBEDDING_DIMENSION, Base

TEST_DB_NAME = "adaptrag_test"


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """The app's database URL with the database name swapped to the test database."""
    url = make_url(get_settings().database_url).set(database=TEST_DB_NAME)
    return url.render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def migrated_engine(test_database_url: str) -> Engine:
    """Rebuild the test database with Alembic, once per test run."""
    config = Config("alembic.ini")
    config.attributes["database_url"] = test_database_url
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    return get_engine(test_database_url)


@pytest.fixture
def db_engine(migrated_engine: Engine) -> Iterator[Engine]:
    """Hand the test a migrated database, then empty every table afterwards."""
    yield migrated_engine
    table_names = ", ".join(table.name for table in Base.metadata.sorted_tables)
    with migrated_engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {table_names} RESTART IDENTITY CASCADE"))


@pytest.fixture
def make_client(
    db_engine: Engine, test_database_url: str, tmp_path: Path
) -> Callable[..., TestClient]:
    """Build an app that uses the test database, a temporary data folder and a fake embedder.

    Uploads are saved under tmp_path / "uploads" (data_dir / "uploads", as in
    routes/documents.py). The real model is never loaded here.
    """

    def _make(max_upload_mb: int = 20, embedder: Embedder | None = None) -> TestClient:
        app = create_app()
        app.dependency_overrides[get_settings] = lambda: Settings(
            database_url=test_database_url,
            data_dir=tmp_path,
            max_upload_mb=max_upload_mb,
        )
        chosen = embedder if embedder is not None else FakeEmbedder(dimension=EMBEDDING_DIMENSION)
        app.dependency_overrides[get_app_embedder] = lambda: chosen
        return TestClient(app)

    return _make
