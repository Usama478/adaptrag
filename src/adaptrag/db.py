from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from adaptrag.config import Settings, get_settings


@lru_cache
def get_engine(database_url: str) -> Engine:
    """Create one engine (one connection pool) per database URL, and reuse it."""
    return create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )


def get_session(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Iterator[Session]:
    """Give each request its own session, and close it when the request ends."""
    engine = get_engine(settings.database_url)
    with Session(engine) as session:
        yield session
