"""Tests for the background ingest job and GET /jobs/{id}."""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from adaptrag import ingest
from adaptrag.embedding.fake import FakeEmbedder
from adaptrag.models import (
    EMBEDDING_DIMENSION,
    Base,
    Chunk,
    ChunkEmbedding,
    ChunkSet,
    Document,
    Job,
)
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


def test_successful_document_is_ready_with_page_count(
    make_client: ClientFactory, db_engine: Engine
) -> None:
    body = upload(make_client(), make_pdf_bytes(pages=2)).json()
    document = get_document(db_engine, body["document_id"])
    assert document.status == "ready"
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

    def save_then_crash(*args: Any, **kwargs: Any) -> tuple[int, bool]:
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


# --- Embedding during ingestion (Step 4) ---


class CrashingEmbedder(FakeEmbedder):
    """A fake whose embedding step fails, to test the rollback."""

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("simulated embedding failure")


def test_successful_job_stores_one_embedding_per_chunk(
    make_client: ClientFactory, db_engine: Engine
) -> None:
    body = upload(make_client(), make_pdf_bytes(pages=2)).json()

    with Session(db_engine) as session:
        chunk_ids = set(session.scalars(select(Chunk.id)))
        embeddings = session.scalars(select(ChunkEmbedding)).all()

    assert len(chunk_ids) >= 1
    assert len(embeddings) == len(chunk_ids)
    assert {e.chunk_id for e in embeddings} == chunk_ids
    assert {e.model_name for e in embeddings} == {"fake"}
    assert all(len(e.embedding) == EMBEDDING_DIMENSION for e in embeddings)
    assert get_document(db_engine, body["document_id"]).status == "ready"


def test_running_the_job_again_adds_no_embeddings(
    make_client: ClientFactory, db_engine: Engine, tmp_path: Path
) -> None:
    body = upload(make_client(), make_pdf_bytes()).json()
    before = count_rows(db_engine, ChunkEmbedding)
    assert before >= 1

    with Session(db_engine) as session:
        job = Job(document_id=body["document_id"])
        session.add(job)
        session.commit()
        second_job_id = job.id

    # make_client sets data_dir = tmp_path; routes/documents.py saves to data_dir / "uploads".
    upload_dir = tmp_path / "uploads"
    ingest.run_ingest_job(
        db_engine, upload_dir, second_job_id, FakeEmbedder(dimension=EMBEDDING_DIMENSION)
    )

    with Session(db_engine) as session:
        assert session.get_one(Job, second_job_id).status == "succeeded"
    assert count_rows(db_engine, ChunkEmbedding) == before
    assert count_rows(db_engine, ChunkSet) == 1


def test_embedding_failure_leaves_nothing_behind(
    make_client: ClientFactory, db_engine: Engine
) -> None:
    client = make_client(embedder=CrashingEmbedder(dimension=EMBEDDING_DIMENSION))
    body = upload(client, make_pdf_bytes()).json()
    job = client.get(f"/jobs/{body['job_id']}").json()

    assert job["status"] == "failed"
    assert job["error"] == ingest.INTERNAL_ERROR
    assert get_document(db_engine, body["document_id"]).status == "failed"
    assert count_rows(db_engine, ChunkSet) == 0
    assert count_rows(db_engine, Chunk) == 0
    assert count_rows(db_engine, ChunkEmbedding) == 0


def over_limit_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "adaptrag.ingest"
        and record.levelno == logging.WARNING
        and "exceed" in record.getMessage()
    ]


def test_chunks_over_token_limit_are_warned_about(
    make_client: ClientFactory, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="adaptrag.ingest")
    # The test PDF gives one chunk of 100 words = 102 fake tokens, far over 10.
    client = make_client(embedder=FakeEmbedder(dimension=EMBEDDING_DIMENSION, max_tokens=10))
    body = upload(client, make_pdf_bytes(pages=2)).json()

    messages = over_limit_warnings(caplog)
    assert len(messages) == 1
    assert f"Document {body['document_id']}" in messages[0]
    assert "1 of 1 chunks exceed 10 tokens" in messages[0]


def test_no_warning_when_chunks_fit(
    make_client: ClientFactory, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="adaptrag.ingest")
    upload(make_client(), make_pdf_bytes(pages=2))  # 102 fake tokens, limit 512
    assert over_limit_warnings(caplog) == []
