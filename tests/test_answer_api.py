"""Tests for POST /documents/{id}/answer (FakeEmbedder + FakeLLM, no network)."""

from collections.abc import Callable
from typing import Any

import httpx
import pymupdf
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr

from adaptrag.config import Settings
from adaptrag.llm.fake import FakeLLM
from adaptrag.llm.provider import NOT_CONFIGURED, get_app_llm

ClientFactory = Callable[..., TestClient]

QUESTION = "What does the paper propose?"
PASS_ALL = -1.0  # every cosine score is >= -1, so retrieval never says "not found"


def make_pdf_bytes(pages: int) -> bytes:
    """A real PDF with 50 different words per page (as in test_query.py)."""
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
    return client.post(f"/documents/{document_id}/answer", json=body)


def test_answer_returns_found_with_citations_and_sources(make_client: ClientFactory) -> None:
    llm = FakeLLM(reply="It proposes p-words [1].")
    client = make_client(llm=llm, retrieval_min_score=PASS_ALL)
    document_id = upload(client, make_pdf_bytes(pages=12))  # 4 chunks

    response = ask(client, document_id, question=QUESTION, k=3)

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == document_id
    assert body["found"] is True
    assert body["answer"] == "It proposes p-words [1]."
    sources = body["sources"]
    assert [source["source"] for source in sources] == [1, 2, 3]
    scores = [source["score"] for source in sources]
    assert scores == sorted(scores, reverse=True)
    for source in sources:
        assert set(source) == {
            "source",
            "chunk_id",
            "chunk_index",
            "page_start",
            "page_end",
            "score",
            "text",
        }
    assert body["citations"] == [
        {
            "source": 1,
            "chunk_id": sources[0]["chunk_id"],
            "page_start": sources[0]["page_start"],
            "page_end": sources[0]["page_end"],
        }
    ]
    assert llm.calls == 1


def test_weak_retrieval_returns_not_found_without_calling_llm(make_client: ClientFactory) -> None:
    llm = FakeLLM()
    client = make_client(llm=llm, retrieval_min_score=1.01)
    document_id = upload(client, make_pdf_bytes(pages=2))

    body = ask(client, document_id, question=QUESTION).json()

    assert body["found"] is False
    assert body["answer"] == "Not found in your documents."
    assert body["citations"] == []
    assert llm.calls == 0


def test_unknown_document_returns_404(make_client: ClientFactory) -> None:
    response = ask(make_client(), 999999, question=QUESTION)
    assert response.status_code == 404


def test_failed_document_returns_409(make_client: ClientFactory) -> None:
    client = make_client()
    document_id = upload(client, b"%PDF-1.7 fake body")  # the job fails

    response = ask(client, document_id, question=QUESTION)

    assert response.status_code == 409
    assert response.json()["detail"] == "Document is not ready (status: failed)"


@pytest.mark.parametrize(
    "body",
    [{"question": "   "}, {"question": "x" * 1001}, {"question": QUESTION, "k": 21}, {}],
    ids=["blank", "too-long", "k-21", "no-question"],
)
def test_invalid_body_returns_422(make_client: ClientFactory, body: dict[str, Any]) -> None:
    llm = FakeLLM()
    client = make_client(llm=llm)
    document_id = upload(client, make_pdf_bytes(pages=2))

    response = ask(client, document_id, **body)

    assert response.status_code == 422
    assert llm.calls == 0


def test_llm_error_returns_502(make_client: ClientFactory) -> None:
    client = make_client(llm=FakeLLM(raise_error=True), retrieval_min_score=PASS_ALL)
    document_id = upload(client, make_pdf_bytes(pages=2))

    response = ask(client, document_id, question=QUESTION)

    assert response.status_code == 502
    assert response.json()["detail"] == "The answer service is unavailable. Try again later."


def test_missing_key_returns_503(make_client: ClientFactory) -> None:
    client = make_client()
    document_id = upload(client, make_pdf_bytes(pages=2))

    def no_key() -> FakeLLM:
        raise HTTPException(status_code=503, detail=NOT_CONFIGURED)

    # Replace make_client's FakeLLM override with one that answers 503, like the real
    # provider does without a key (the real provider is tested directly below).
    client.app.dependency_overrides[get_app_llm] = no_key

    response = ask(client, document_id, question=QUESTION)

    assert response.status_code == 503
    assert response.json()["detail"] == "Answering is not configured (no API key)."


@pytest.mark.parametrize("key", [None, SecretStr(""), SecretStr("   ")])
def test_provider_refuses_without_key(key: SecretStr | None) -> None:
    # The real dependency, called directly: no key -> 503 before openai is even imported.
    with pytest.raises(HTTPException) as caught:
        get_app_llm(Settings(openai_api_key=key))
    assert caught.value.status_code == 503
    assert caught.value.detail == NOT_CONFIGURED


def test_query_still_works_when_llm_fails(make_client: ClientFactory) -> None:
    client = make_client(llm=FakeLLM(raise_error=True), retrieval_min_score=PASS_ALL)
    document_id = upload(client, make_pdf_bytes(pages=2))

    query = client.post(f"/documents/{document_id}/query", json={"question": QUESTION})
    answer = ask(client, document_id, question=QUESTION)

    assert query.status_code == 200
    assert query.json()["results"]
    assert answer.status_code == 502
