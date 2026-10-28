import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from adaptrag.models import Document

PDF_BYTES = b"%PDF-1.7 fake body"

ClientFactory = Callable[..., TestClient]


def upload(client: TestClient, content: bytes = PDF_BYTES, name: str = "a.pdf") -> httpx.Response:
    files = {"file": (name, content, "application/pdf")}
    return client.post("/documents", files=files)


def count_documents(engine: Engine) -> int:
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(Document)) or 0


def test_new_upload_creates_document(make_client: ClientFactory, db_engine: Engine) -> None:
    response = upload(make_client())
    assert response.status_code == 202
    assert response.json()["status"] == "uploaded"
    assert response.json()["duplicate"] is False
    assert count_documents(db_engine) == 1


def test_upload_saves_file_named_by_hash(make_client: ClientFactory, tmp_path: Path) -> None:
    file_hash = upload(make_client()).json()["file_hash"]
    assert file_hash == hashlib.sha256(PDF_BYTES).hexdigest()
    assert (tmp_path / "uploads" / f"{file_hash}.pdf").read_bytes() == PDF_BYTES


def test_rejected_file_is_not_saved(
    make_client: ClientFactory, db_engine: Engine, tmp_path: Path
) -> None:
    response = upload(make_client(), content=b"I am not a PDF", name="fake.pdf")
    assert response.status_code == 415
    assert not (tmp_path / "uploads").exists()
    assert count_documents(db_engine) == 0


def test_rejects_wrong_extension(make_client: ClientFactory) -> None:
    response = upload(make_client(), content=b"hello", name="notes.txt")
    assert response.status_code == 415


def test_rejects_too_large(make_client: ClientFactory) -> None:
    big = b"%PDF-" + b"x" * (2 * 1024 * 1024)
    response = upload(make_client(max_upload_mb=1), content=big, name="big.pdf")
    assert response.status_code == 413


def test_duplicate_upload_returns_same_document(make_client: ClientFactory) -> None:
    client = make_client()
    first = upload(client, name="a.pdf")
    second = upload(client, name="renamed.pdf")
    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["document_id"] == first.json()["document_id"]


def test_duplicate_creates_only_one_row(make_client: ClientFactory, db_engine: Engine) -> None:
    client = make_client()
    upload(client)
    upload(client)
    assert count_documents(db_engine) == 1
