import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from adaptrag.models import Document

HASH = "a" * 64
ROUTING_COLUMNS = {"profile", "strategy", "route_rule", "route_reason"}
REVISION_BEFORE_ROUTING = "5c8e1a7d4f2b"


def test_file_hash_must_be_unique(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        session.add(Document(file_hash=HASH, filename="a.pdf", size_bytes=1))
        session.commit()
        session.add(Document(file_hash=HASH, filename="b.pdf", size_bytes=1))
        with pytest.raises(IntegrityError):
            session.commit()


def document_columns(engine: Engine) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns("documents")}


def test_routing_columns_return_after_downgrade_then_upgrade(
    db_engine: Engine, test_database_url: str
) -> None:
    config = Config("alembic.ini")
    config.attributes["database_url"] = test_database_url
    assert ROUTING_COLUMNS <= document_columns(db_engine)

    try:
        command.downgrade(config, REVISION_BEFORE_ROUTING)
        assert not ROUTING_COLUMNS & document_columns(db_engine)
    finally:
        # Always go back to head, or every later test would run on the old schema.
        command.upgrade(config, "head")

    assert ROUTING_COLUMNS <= document_columns(db_engine)
