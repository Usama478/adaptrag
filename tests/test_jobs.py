"""Tests for the background ingest job and GET /jobs/{id}."""

from collections.abc import Callable
from typing import Any

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from adaptrag import ingest
from adaptrag.models import Base, Chunk, ChunkSet, Document
from adaptrag.parsing.pdf import PARSER_VERSION

ClientFactory = Callable[..., TestClient]


def make_pdf_bytes(pages: int = 2) -> bytes:
    """A real PDF with enough text per page to pass the scanned check (50 words per page)."""
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page().insert_text((72, 72), ("word " * 10 + "\n") * 5)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def upload(client: TestClient, content: bytes, name: str = "paper.pdf") -> httpx.Response:
    return client.post("/documents", files={"file": (name, content, "application/pdf")})


def get_document(engine: Engine, document_id: int) -> Document:
    with Session(engine) as session:
        return session.get_one(Document, document_id)


def count_rows(engine: Engine, model: type[Base]) -> int:
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(model)) or 0


def test_good_pdf_job_succeeds(make_client: ClientFactory) -> None:
    client = make_client()
    body = upload(client, make_pdf_bytes()).json()
    job = client.get(f"/jobs/{body['job_id']}").json()
    assert job["status"] == "succeeded"
    assert job["error"] is None
    assert job["document_id"] == body["document_id"]


def test_parsed_document_gets_page_count(make_client: ClientFactory, db_engine: Engine) -> None:
    body = upload(make_client(), make_pdf_bytes(pages=2)).json()
    document = get_document(db_engine, body["document_id"])
    assert document.status == "parsed"
    assert document.page_count == 2


def test_broken_pdf_job_fails_with_message(make_client: ClientFactory, db_engine: Engine) -> None:
    client = make_client()
    body = upload(client, b"%PDF-1.7 fake body").json()
    job = client.get(f"/jobs/{body['job_id']}").json()
    assert job["status"] == "failed"
    assert job["error"]
    assert get_document(db_engine, body["document_id"]).status == "failed"


def test_duplicate_upload_returns_existing_job_id(make_client: ClientFactory) -> None:
    client = make_client()
    pdf = make_pdf_bytes()
    first = upload(client, pdf).json()
    second = upload(client, pdf, name="again.pdf").json()
    assert second["duplicate"] is True
    assert second["job_id"] == first["job_id"]


def test_unknown_job_returns_404(make_client: ClientFactory) -> None:
    response = make_client().get("/jobs/999999")
    assert response.status_code == 404


def test_successful_job_saves_fixed_size_chunks(
    make_client: ClientFactory, db_engine: Engine
) -> None:
    body = upload(make_client(), make_pdf_bytes(pages=2)).json()

    with Session(db_engine) as session:
        chunk_set = session.scalars(
            select(ChunkSet).where(ChunkSet.document_id == body["document_id"])
        ).one()
        chunks = session.scalars(
            select(Chunk).where(Chunk.chunk_set_id == chunk_set.id).order_by(Chunk.chunk_index)
        ).all()

        assert chunk_set.strategy == "fixed_size"
        assert chunk_set.params == {"size_words": 200, "overlap_words": 40}
        assert chunk_set.parser_version == PARSER_VERSION
        assert len(chunks) == 1
        assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)
        assert len(chunks[0].text.split()) == 100


def test_failed_job_saves_no_chunks(make_client: ClientFactory, db_engine: Engine) -> None:
    upload(make_client(), b"%PDF-1.7 fake body")
    assert count_rows(db_engine, ChunkSet) == 0
    assert count_rows(db_engine, Chunk) == 0


def test_crash_while_saving_leaves_no_chunks(
    make_client: ClientFactory, db_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_save = ingest.save_chunk_set

    def save_then_crash(*args: Any, **kwargs: Any) -> bool:
        real_save(*args, **kwargs)
        raise RuntimeError("simulated crash after adding chunks")

    monkeypatch.setattr(ingest, "save_chunk_set", save_then_crash)
    client = make_client()
    body = upload(client, make_pdf_bytes()).json()
    job = client.get(f"/jobs/{body['job_id']}").json()

    assert job["status"] == "failed"
    assert job["error"] == ingest.INTERNAL_ERROR
    assert count_rows(db_engine, ChunkSet) == 0
    assert count_rows(db_engine, Chunk) == 0
