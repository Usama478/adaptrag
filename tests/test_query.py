"""Tests for POST /documents/{id}/query (dense retrieval, no LLM).

The fake embedder's scores carry no meaning (note 0014): these tests check status codes,
response shape, order and limits, not whether the "right" chunk wins.
"""

from collections.abc import Callable
from typing import Any

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from adaptrag.embedding.fake import FakeEmbedder
from adaptrag.models import EMBEDDING_DIMENSION, Chunk, ChunkSet

ClientFactory = Callable[..., TestClient]

QUESTION = "What does the paper propose?"


def make_pdf_bytes(pages: int) -> bytes:
    """A real PDF with 50 different words per page (word 'pN' on page N).

    The fixed-size chunker (200 words, 40 overlap) makes several chunks with
    different texts from it, so their fake vectors differ.
    """
    doc = pymupdf.open()
    for page in range(1, pages + 1):
        doc.new_page().insert_text((72, 72), (f"p{page} " * 10 + "\n") * 5)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def upload(client: TestClient, content: bytes) -> int:
    response = client.post("/documents", files={"file": ("paper.pdf", content, "application/pdf")})
    document_id: int = response.json()["document_id"]
    return document_id


def ask(client: TestClient, document_id: int, **body: Any) -> httpx.Response:
    return client.post(f"/documents/{document_id}/query", json=body)


def test_ready_document_returns_top_k_best_first(make_client: ClientFactory) -> None:
    client = make_client()
    # 12 pages x 50 words = 600 words -> 4 chunks.
    document_id = upload(client, make_pdf_bytes(pages=12))

    response = ask(client, document_id, question=QUESTION, k=3)

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == document_id
    assert isinstance(body["chunk_set_id"], int)
    assert body["model_name"] == "fake"
    results = body["results"]
    assert len(results) == 3
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)
    for result in results:
        assert 1 <= result["page_start"] <= result["page_end"] <= 12
        assert result["text"]


def test_default_k_is_five_and_never_exceeds_chunk_count(make_client: ClientFactory) -> None:
    client = make_client()
    document_id = upload(client, make_pdf_bytes(pages=12))  # 4 chunks

    results = ask(client, document_id, question=QUESTION).json()["results"]

    assert len(results) == 4
    assert len({result["chunk_id"] for result in results}) == 4


def test_top_result_is_the_most_similar_chunk(
    make_client: ClientFactory, db_engine: Engine
) -> None:
    client = make_client()
    document_id = upload(client, make_pdf_bytes(pages=12))

    results = ask(client, document_id, question=QUESTION, k=1).json()["results"]

    # Work out the expected winner directly from the fake embedder's vectors.
    embedder = FakeEmbedder(dimension=EMBEDDING_DIMENSION)
    query = embedder.embed_query(QUESTION)
    with Session(db_engine) as session:
        chunks = session.execute(select(Chunk.id, Chunk.text)).all()
    vectors = embedder.embed_passages([text for _, text in chunks])
    similarities = [sum(q * v for q, v in zip(query, vector, strict=True)) for vector in vectors]
    best_chunk_id = chunks[similarities.index(max(similarities))][0]

    assert results[0]["chunk_id"] == best_chunk_id
    assert results[0]["score"] == pytest.approx(max(similarities), abs=1e-5)


def test_unknown_document_returns_404(make_client: ClientFactory) -> None:
    response = ask(make_client(), 999999, question=QUESTION)
    assert response.status_code == 404


def test_failed_document_returns_409(make_client: ClientFactory) -> None:
    client = make_client()
    document_id = upload(client, b"%PDF-1.7 fake body")  # the job fails

    response = ask(client, document_id, question=QUESTION)

    assert response.status_code == 409
    assert response.json()["detail"] == "Document is not ready (status: failed)"


def test_ready_document_without_chunk_set_returns_409(
    make_client: ClientFactory, db_engine: Engine
) -> None:
    client = make_client()
    document_id = upload(client, make_pdf_bytes(pages=2))
    with Session(db_engine) as session:
        session.execute(delete(ChunkSet).where(ChunkSet.document_id == document_id))
        session.commit()

    response = ask(client, document_id, question=QUESTION)

    assert response.status_code == 409
    assert response.json()["detail"] == "Document needs re-ingestion for the current settings"


def test_document_embedded_with_another_model_returns_409(make_client: ClientFactory) -> None:
    document_id = upload(make_client(), make_pdf_bytes(pages=2))  # embedded as "fake"
    other = FakeEmbedder(dimension=EMBEDDING_DIMENSION)
    other.model_name = "another-model"

    response = ask(make_client(embedder=other), document_id, question=QUESTION)

    assert response.status_code == 409
    assert response.json()["detail"] == "Document needs re-ingestion for the current settings"


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 1001},
        {"question": QUESTION, "k": 0},
        {"question": QUESTION, "k": 21},
        {},
    ],
    ids=["empty", "blank", "too-long", "k-0", "k-21", "no-question"],
)
def test_invalid_body_returns_422(make_client: ClientFactory, body: dict[str, Any]) -> None:
    client = make_client()
    document_id = upload(client, make_pdf_bytes(pages=2))

    response = ask(client, document_id, **body)

    assert response.status_code == 422
