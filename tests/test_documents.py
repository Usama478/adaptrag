import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from adaptrag.config import Settings, get_settings
from adaptrag.main import create_app


def make_client(data_dir: Path, max_upload_mb: int = 20) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        max_upload_mb=max_upload_mb, data_dir=data_dir
    )
    return TestClient(app)


def test_accepts_pdf(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    files = {"file": ("a.pdf", b"%PDF-1.7 fake body", "application/pdf")}
    response = client.post("/documents", files=files)
    assert response.status_code == 202
    assert response.json()["status"] == "accepted"


def test_upload_saves_file_named_by_hash(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    content = b"%PDF-1.7 fake body"
    files = {"file": ("a.pdf", content, "application/pdf")}
    file_hash = client.post("/documents", files=files).json()["file_hash"]
    assert file_hash == hashlib.sha256(content).hexdigest()
    assert (tmp_path / "uploads" / f"{file_hash}.pdf").read_bytes() == content


def test_rejected_file_is_not_saved(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    files = {"file": ("fake.pdf", b"I am not a PDF", "application/pdf")}
    assert client.post("/documents", files=files).status_code == 415
    assert not (tmp_path / "uploads").exists()


def test_rejects_wrong_extension(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    assert client.post("/documents", files=files).status_code == 415


def test_rejects_too_large(tmp_path: Path) -> None:
    client = make_client(tmp_path, max_upload_mb=1)
    big = b"%PDF-" + b"x" * (2 * 1024 * 1024)
    files = {"file": ("big.pdf", big, "application/pdf")}
    assert client.post("/documents", files=files).status_code == 413
