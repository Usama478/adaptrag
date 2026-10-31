"""Tests for storing chunk sets and chunks in Postgres."""

from typing import Any

import pytest
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from adaptrag.chunking.chunks import TextChunk
from adaptrag.ingest import save_chunk_set
from adaptrag.models import Base, Chunk, ChunkSet, Document

PARAMS: dict[str, Any] = {"size_words": 200, "overlap_words": 40}
TWO_CHUNKS = [
    TextChunk(index=0, text="first chunk", page_start=1, page_end=1),
    TextChunk(index=1, text="second chunk", page_start=1, page_end=2),
]


def add_document(session: Session) -> int:
    document = Document(file_hash="a" * 64, filename="paper.pdf", size_bytes=100)
    session.add(document)
    session.flush()
    return document.id


def count(session: Session, model: type[Base]) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def save(session: Session, document_id: int, params: dict[str, Any]) -> bool:
    return save_chunk_set(session, document_id, "fixed_size", params, "0.1.0", TWO_CHUNKS)


def test_chunks_are_stored_in_order(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        document_id = add_document(session)
        assert save(session, document_id, PARAMS) is True
        session.commit()

        chunks = session.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()
        assert [c.text for c in chunks] == ["first chunk", "second chunk"]
        assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (1, 2)]


def test_identical_chunk_set_is_skipped(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        document_id = add_document(session)
        save(session, document_id, PARAMS)
        assert save(session, document_id, PARAMS) is False
        session.commit()

        assert count(session, ChunkSet) == 1
        assert count(session, Chunk) == 2


def test_different_params_make_a_second_set(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        document_id = add_document(session)
        save(session, document_id, PARAMS)
        assert save(session, document_id, {"size_words": 100, "overlap_words": 20}) is True
        session.commit()

        assert count(session, ChunkSet) == 2
        assert count(session, Chunk) == 4


def test_deleting_document_removes_its_chunks(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        document_id = add_document(session)
        save(session, document_id, PARAMS)
        session.commit()

        session.execute(delete(Document).where(Document.id == document_id))
        session.commit()

        assert count(session, ChunkSet) == 0
        assert count(session, Chunk) == 0


def test_chunk_index_is_unique_within_a_set(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        document_id = add_document(session)
        save(session, document_id, PARAMS)
        session.commit()

        chunk_set_id = session.scalars(select(ChunkSet.id)).one()
        session.add(
            Chunk(chunk_set_id=chunk_set_id, chunk_index=0, text="dup", page_start=1, page_end=1)
        )
        with pytest.raises(IntegrityError):
            session.commit()
