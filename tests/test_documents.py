from fastapi.testclient import TestClient

from adaptrag.config import Settings, get_settings
from adaptrag.main import create_app


def make_client(max_upload_mb: int = 20) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(max_upload_mb=max_upload_mb)
    return TestClient(app)


def test_accepts_pdf() -> None:
    client = make_client()
    files = {"file": ("a.pdf", b"%PDF-1.7 fake body", "application/pdf")}
    response = client.post("/documents", files=files)
    assert response.status_code == 202
    assert response.json()["status"] == "accepted"


def test_rejects_wrong_extension() -> None:
    client = make_client()
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    assert client.post("/documents", files=files).status_code == 415


def test_rejects_fake_pdf() -> None:
    client = make_client()
    files = {"file": ("fake.pdf", b"I am not a PDF", "application/pdf")}
    assert client.post("/documents", files=files).status_code == 415


def test_rejects_too_large() -> None:
    client = make_client(max_upload_mb=1)
    big = b"%PDF-" + b"x" * (2 * 1024 * 1024)
    files = {"file": ("big.pdf", big, "application/pdf")}
    assert client.post("/documents", files=files).status_code == 413
