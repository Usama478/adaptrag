import pytest
from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from adaptrag.models import Document

HASH = "a" * 64


def test_file_hash_must_be_unique(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        session.add(Document(file_hash=HASH, filename="a.pdf", size_bytes=1))
        session.commit()
        session.add(Document(file_hash=HASH, filename="b.pdf", size_bytes=1))
        with pytest.raises(IntegrityError):
            session.commit()
