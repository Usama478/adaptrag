"""A strategy registered only in this file runs through ingest and /query (note 0019).

The promise: adding a strategy needs no change to ingest, the routes or the database code.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from adaptrag import ingest
from adaptrag.chunking.chunks import TextChunk
from adaptrag.models import Chunk, ChunkSet
from adaptrag.parsing.blocks import ParsedDocument
from adaptrag.parsing.pdf import parse_pdf
from adaptrag.routing.profile import DocumentProfile
from adaptrag.routing.router import RouteDecision
from adaptrag.storage import upload_path
from adaptrag.strategies.base import Strategy
from adaptrag.strategies.registry import STRATEGIES

ClientFactory = Callable[..., TestClient]


class DummyStrategy(Strategy):
    """One chunk per parsed block. It exists only in this test file."""

    name = "dummy"

    @property
    def params(self) -> dict[str, int | float | str | bool]:
        return {"variant": "dummy"}

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        return [
            TextChunk(index=index, text=block.text, page_start=block.page, page_end=block.page)
            for index, block in enumerate(doc.blocks)
        ]


@pytest.fixture(autouse=True)
def router_chooses_the_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(STRATEGIES, DummyStrategy.name, DummyStrategy)

    def route_to_dummy(profile: DocumentProfile) -> RouteDecision:
        return RouteDecision(DummyStrategy.name, "test_rule", "test reason")

    # ingest imported route by name, so the patch must be applied to ingest's reference.
    monkeypatch.setattr(ingest, "route", route_to_dummy)


def make_pdf_bytes(pages: int = 2) -> bytes:
    """A real PDF with 50 words per page, and different words on each page."""
    doc = pymupdf.open()
    for page in range(1, pages + 1):
        doc.new_page().insert_text((72, 72), (f"p{page} " * 10 + "\n") * 5)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def upload(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/documents", files={"file": ("paper.pdf", make_pdf_bytes(), "application/pdf")}
    )
    body: dict[str, Any] = response.json()
    return body


def stored_chunk_set(engine: Engine, document_id: int) -> tuple[ChunkSet, list[Chunk]]:
    """The document's only chunk set (`.one()` fails if there are others) and its chunks."""
    with Session(engine) as session:
        chunk_set = session.scalars(
            select(ChunkSet).where(ChunkSet.document_id == document_id)
        ).one()
        chunks = session.scalars(
            select(Chunk).where(Chunk.chunk_set_id == chunk_set.id).order_by(Chunk.chunk_index)
        ).all()
        return chunk_set, list(chunks)


def test_ingest_saves_the_chunk_set_of_a_test_only_strategy(
    make_client: ClientFactory, db_engine: Engine, tmp_path: Path
) -> None:
    client = make_client()
    body = upload(client)

    job = client.get(f"/jobs/{body['job_id']}").json()
    assert job["status"] == "succeeded", job["error"]

    chunk_set, chunks = stored_chunk_set(db_engine, body["document_id"])
    # make_client saves uploads under tmp_path / "uploads".
    blocks = parse_pdf(upload_path(body["file_hash"], tmp_path / "uploads")).blocks
    assert len(blocks) >= 2  # two pages: the fixed-size chunker would make one chunk of these
    assert chunk_set.strategy == "dummy"
    assert chunk_set.params == {"variant": "dummy"}
    assert [(chunk.text, chunk.page_start, chunk.page_end) for chunk in chunks] == [
        (block.text, block.page, block.page) for block in blocks
    ]


def test_query_searches_the_chunk_set_of_a_test_only_strategy(
    make_client: ClientFactory, db_engine: Engine
) -> None:
    client = make_client()
    document_id = upload(client)["document_id"]
    chunk_set, chunks = stored_chunk_set(db_engine, document_id)

    response = client.post(
        f"/documents/{document_id}/query", json={"question": "What does the paper propose?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_set_id"] == chunk_set.id
    assert body["results"]
    assert {result["chunk_id"] for result in body["results"]} <= {chunk.id for chunk in chunks}
