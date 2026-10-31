"""Tests for storing chunk vectors with pgvector (database tests)."""

import pytest
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from adaptrag.chunking.chunks import TextChunk
from adaptrag.embedding.fake import FakeEmbedder
from adaptrag.ingest import save_chunk_set
from adaptrag.models import EMBEDDING_DIMENSION, Chunk, ChunkEmbedding, Document
from adaptrag.vectorstore.pgvector import embeddings_exist, save_embeddings

TWO_CHUNKS = [
    TextChunk(index=0, text="first chunk", page_start=1, page_end=1),
    TextChunk(index=1, text="second chunk", page_start=1, page_end=2),
]
EMBEDDER = FakeEmbedder(dimension=EMBEDDING_DIMENSION)


def add_chunk_set(session: Session) -> tuple[int, int, list[int]]:
    """A document with one two-chunk set. Returns (document_id, chunk_set_id, chunk_ids)."""
    document = Document(file_hash="a" * 64, filename="paper.pdf", size_bytes=100)
    session.add(document)
    session.flush()
    params = {"size_words": 200, "overlap_words": 40}
    chunk_set_id, _ = save_chunk_set(
        session, document.id, "fixed_size", params, "0.1.0", TWO_CHUNKS
    )
    chunk_ids = list(
        session.scalars(
            select(Chunk.id).where(Chunk.chunk_set_id == chunk_set_id).order_by(Chunk.chunk_index)
        )
    )
    return document.id, chunk_set_id, chunk_ids


def embed(session: Session, chunk_ids: list[int], model_name: str = "fake") -> None:
    vectors = EMBEDDER.embed_passages([chunk.text for chunk in TWO_CHUNKS])
    save_embeddings(session, model_name, chunk_ids, vectors)


def count_embeddings(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(ChunkEmbedding)) or 0


def test_saved_embeddings_are_found(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        _, chunk_set_id, chunk_ids = add_chunk_set(session)
        assert embeddings_exist(session, chunk_set_id, "fake") is False
        embed(session, chunk_ids)
        session.commit()

        assert embeddings_exist(session, chunk_set_id, "fake") is True
        assert count_embeddings(session) == 2


def test_vectors_round_trip(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        _, _, chunk_ids = add_chunk_set(session)
        embed(session, chunk_ids)
        session.commit()

        stored = session.scalars(
            select(ChunkEmbedding.embedding).where(ChunkEmbedding.chunk_id == chunk_ids[0])
        ).one()
        expected = EMBEDDER.embed_passages([TWO_CHUNKS[0].text])[0]
        # pgvector stores 32-bit floats, so compare approximately.
        assert list(stored) == pytest.approx(expected, abs=1e-6)


def test_other_model_name_is_not_found(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        _, chunk_set_id, chunk_ids = add_chunk_set(session)
        embed(session, chunk_ids, model_name="fake")
        session.commit()

        assert embeddings_exist(session, chunk_set_id, "another-model") is False


def test_deleting_document_removes_its_embeddings(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        document_id, _, chunk_ids = add_chunk_set(session)
        embed(session, chunk_ids)
        session.commit()
        assert count_embeddings(session) == 2

        session.execute(delete(Document).where(Document.id == document_id))
        session.commit()

        assert count_embeddings(session) == 0


def test_wrong_dimension_is_rejected(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        _, _, chunk_ids = add_chunk_set(session)
        short = FakeEmbedder(dimension=8).embed_passages(["first chunk", "second chunk"])
        with pytest.raises(ValueError):
            save_embeddings(session, "fake", chunk_ids, short)
        session.commit()

        assert count_embeddings(session) == 0


def test_mismatched_lengths_are_rejected(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        _, _, chunk_ids = add_chunk_set(session)
        one_vector = EMBEDDER.embed_passages(["first chunk"])
        with pytest.raises(ValueError):
            save_embeddings(session, "fake", chunk_ids, one_vector)
        session.commit()

        assert count_embeddings(session) == 0
