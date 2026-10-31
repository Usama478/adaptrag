"""Tests for answer_question (database: chunks and fake vectors in the test database).

FakeEmbedder scores carry no meaning, so min_score is set to the extremes: -1.0 lets
every result through (cosine >= -1), 1.01 blocks every result (cosine <= 1).
"""

import logging

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from adaptrag.answering.answer import NOT_FOUND_MESSAGE, answer_question
from adaptrag.chunking.chunks import TextChunk
from adaptrag.embedding.fake import FakeEmbedder
from adaptrag.ingest import save_chunk_set
from adaptrag.llm.fake import FakeLLM
from adaptrag.models import EMBEDDING_DIMENSION, Chunk, Document
from adaptrag.vectorstore.pgvector import save_embeddings

EMBEDDER = FakeEmbedder(dimension=EMBEDDING_DIMENSION)
TEXTS = ["alpha chunk", "beta chunk", "gamma chunk"]
QUESTION = "Which chunk?"
PASS_ALL = -1.0
BLOCK_ALL = 1.01


def add_embedded_set(session: Session, model_name: str = "fake") -> tuple[int, dict[int, int]]:
    """A document with one embedded chunk set; chunk i is on pages i+1 to i+2.

    Returns (chunk_set_id, {chunk_id: page_start}).
    """
    document = Document(file_hash="d" * 64, filename="paper.pdf", size_bytes=100)
    session.add(document)
    session.flush()
    chunks = [
        TextChunk(index=i, text=text, page_start=i + 1, page_end=i + 2)
        for i, text in enumerate(TEXTS)
    ]
    params = {"size_words": 200, "overlap_words": 40}
    chunk_set_id, _ = save_chunk_set(session, document.id, "fixed_size", params, "0.1.0", chunks)
    rows = session.execute(
        select(Chunk.id, Chunk.page_start)
        .where(Chunk.chunk_set_id == chunk_set_id)
        .order_by(Chunk.chunk_index)
    ).all()
    chunk_ids = [chunk_id for chunk_id, _ in rows]
    save_embeddings(session, model_name, chunk_ids, EMBEDDER.embed_passages(TEXTS))
    session.flush()
    return chunk_set_id, {chunk_id: page for chunk_id, page in rows}


def test_weak_retrieval_is_not_found_and_llm_not_called(db_engine: Engine) -> None:
    llm = FakeLLM(reply="Should not be used [1].")
    with Session(db_engine) as session:
        chunk_set_id, _ = add_embedded_set(session)

        result = answer_question(
            session, chunk_set_id, QUESTION, 3, EMBEDDER, llm, min_score=BLOCK_ALL
        )

    assert result.found is False
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []
    assert llm.calls == 0
    assert len(result.sources) == 3  # still shown, so a caller can see what was searched


def test_no_vectors_is_not_found_and_llm_not_called(db_engine: Engine) -> None:
    llm = FakeLLM()
    with Session(db_engine) as session:
        chunk_set_id, _ = add_embedded_set(session, model_name="another-model")

        result = answer_question(
            session, chunk_set_id, QUESTION, 3, EMBEDDER, llm, min_score=PASS_ALL
        )

    assert result.found is False
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []
    assert llm.calls == 0


@pytest.mark.parametrize("reply", ["NOT_FOUND", "  NOT_FOUND\n"])
def test_llm_not_found_reply_is_not_found(db_engine: Engine, reply: str) -> None:
    llm = FakeLLM(reply=reply)
    with Session(db_engine) as session:
        chunk_set_id, _ = add_embedded_set(session)

        result = answer_question(
            session, chunk_set_id, QUESTION, 3, EMBEDDER, llm, min_score=PASS_ALL
        )

    assert llm.calls == 1
    assert result.found is False
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []


def test_citations_are_deduplicated_in_order_and_invalid_ones_dropped(db_engine: Engine) -> None:
    llm = FakeLLM(reply="Claim one [2]. Claim two [1]. Again [2]. Made up [9].")
    with Session(db_engine) as session:
        chunk_set_id, page_of = add_embedded_set(session)

        result = answer_question(
            session, chunk_set_id, QUESTION, 3, EMBEDDER, llm, min_score=PASS_ALL
        )

    assert result.found is True
    assert result.answer == "Claim one [2]. Claim two [1]. Again [2]. Made up [9]."
    assert [citation.source for citation in result.citations] == [2, 1]
    for citation in result.citations:
        source = result.sources[citation.source - 1]
        assert citation.chunk_id == source.chunk_id
        # Pages come from the stored chunk (chunk i covers pages i+1 to i+2).
        assert citation.page_start == page_of[citation.chunk_id]
        assert citation.page_end == page_of[citation.chunk_id] + 1
    # The prompt held the retrieved texts, numbered in the same order as result.sources.
    assert llm.last_user is not None
    assert f'<source id="1" pages="{result.sources[0].page_start}-' in llm.last_user
    assert result.sources[0].text in llm.last_user


def test_answer_without_citation_is_returned_with_a_warning(
    db_engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    llm = FakeLLM(reply="An answer with no source number.")
    with Session(db_engine) as session:
        chunk_set_id, _ = add_embedded_set(session)

        with caplog.at_level(logging.WARNING, logger="adaptrag.answering.answer"):
            result = answer_question(
                session, chunk_set_id, QUESTION, 3, EMBEDDER, llm, min_score=PASS_ALL
            )

    assert result.found is True
    assert result.answer == "An answer with no source number."
    assert result.citations == []
    assert any(
        record.levelno == logging.WARNING and "no valid citation" in record.getMessage()
        for record in caplog.records
    )
