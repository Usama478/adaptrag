"""Tests for storing chunk vectors with pgvector (database tests)."""

import pytest
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from adaptrag.chunking.chunks import TextChunk
from adaptrag.embedding.fake import FakeEmbedder
from adaptrag.ingest import save_chunk_set
from adaptrag.models import EMBEDDING_DIMENSION, Chunk, ChunkEmbedding, Document
from adaptrag.vectorstore.pgvector import embeddings_exist, save_embeddings, search

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


# --- Search (Step 5) ---
# FakeEmbedder scores carry no meaning (note 0014), so these tests check which rows come
# back, their order and the score formula, never that a "relevant" chunk wins.

FIVE_TEXTS = ["alpha chunk", "beta chunk", "gamma chunk", "delta chunk", "epsilon chunk"]


def add_embedded_set(
    session: Session,
    texts: list[str],
    file_hash: str = "b" * 64,
    model_name: str = "fake",
) -> tuple[int, list[int]]:
    """A document whose one chunk set is embedded. Chunk i is on page i + 1.

    Returns (chunk_set_id, chunk_ids in chunk order).
    """
    document = Document(file_hash=file_hash, filename="paper.pdf", size_bytes=100)
    session.add(document)
    session.flush()
    chunks = [
        TextChunk(index=i, text=text, page_start=i + 1, page_end=i + 1)
        for i, text in enumerate(texts)
    ]
    params = {"size_words": 200, "overlap_words": 40}
    chunk_set_id, _ = save_chunk_set(session, document.id, "fixed_size", params, "0.1.0", chunks)
    chunk_ids = list(
        session.scalars(
            select(Chunk.id).where(Chunk.chunk_set_id == chunk_set_id).order_by(Chunk.chunk_index)
        )
    )
    save_embeddings(session, model_name, chunk_ids, EMBEDDER.embed_passages(texts))
    session.flush()
    return chunk_set_id, chunk_ids


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_own_vector_finds_its_chunk_first(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        chunk_set_id, chunk_ids = add_embedded_set(session, FIVE_TEXTS)
        query = EMBEDDER.embed_passages([FIVE_TEXTS[2]])[0]

        results = search(session, chunk_set_id, "fake", query, k=5)

        top = results[0]
        assert top.chunk_id == chunk_ids[2]
        assert top.chunk_index == 2
        assert top.text == "gamma chunk"
        assert (top.page_start, top.page_end) == (3, 3)
        # Same direction: distance 0, so score 1 (float32 storage: approximately).
        assert top.score == pytest.approx(1.0, abs=1e-5)


def test_results_are_best_first_with_cosine_scores(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        chunk_set_id, chunk_ids = add_embedded_set(session, FIVE_TEXTS)
        query = EMBEDDER.embed_query("which chunk?")

        results = search(session, chunk_set_id, "fake", query, k=5)

        scores = [result.score for result in results]
        assert len(results) == 5
        assert scores == sorted(scores, reverse=True)
        # Vectors are normalised, so cosine similarity = dot product = 1 - distance.
        vectors = dict(zip(chunk_ids, EMBEDDER.embed_passages(FIVE_TEXTS), strict=True))
        for result in results:
            assert result.score == pytest.approx(dot(query, vectors[result.chunk_id]), abs=1e-5)


def test_k_limits_the_number_of_results(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        chunk_set_id, _ = add_embedded_set(session, FIVE_TEXTS)
        query = EMBEDDER.embed_query("which chunk?")

        assert len(search(session, chunk_set_id, "fake", query, k=2)) == 2
        assert len(search(session, chunk_set_id, "fake", query, k=1)) == 1
        # Asking for more than exist returns all of them, each once.
        assert len(search(session, chunk_set_id, "fake", query, k=20)) == 5


def test_other_chunk_sets_and_models_are_never_returned(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        chunk_set_id, chunk_ids = add_embedded_set(session, FIVE_TEXTS)
        # The same chunks also have vectors from another model ...
        other_vectors = FakeEmbedder(dimension=EMBEDDING_DIMENSION).embed_passages(
            [f"other {text}" for text in FIVE_TEXTS]
        )
        save_embeddings(session, "other-model", chunk_ids, other_vectors)
        # ... and another document has an identical chunk set.
        add_embedded_set(session, FIVE_TEXTS, file_hash="c" * 64)
        session.flush()
        query = EMBEDDER.embed_query("which chunk?")

        results = search(session, chunk_set_id, "fake", query, k=20)

        assert sorted(result.chunk_id for result in results) == sorted(chunk_ids)
        assert search(session, chunk_set_id, "no-such-model", query, k=20) == []


def test_wrong_length_query_vector_is_rejected(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        chunk_set_id, _ = add_embedded_set(session, FIVE_TEXTS)
        short = FakeEmbedder(dimension=8).embed_query("which chunk?")
        with pytest.raises(ValueError):
            search(session, chunk_set_id, "fake", short, k=5)


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_is_rejected(db_engine: Engine, k: int) -> None:
    with Session(db_engine) as session:
        chunk_set_id, _ = add_embedded_set(session, FIVE_TEXTS)
        query = EMBEDDER.embed_query("which chunk?")
        with pytest.raises(ValueError):
            search(session, chunk_set_id, "fake", query, k=k)
