import hashlib
from pathlib import Path


def compute_sha256(data: bytes) -> str:
    """Return the SHA-256 fingerprint of the file contents as 64 hex characters."""
    return hashlib.sha256(data).hexdigest()


def upload_path(file_hash: str, upload_dir: Path) -> Path:
    """Where the upload with this hash is stored: <upload_dir>/<hash>.pdf."""
    return upload_dir / f"{file_hash}.pdf"


def save_upload(data: bytes, file_hash: str, upload_dir: Path) -> Path:
    """Save the file as <hash>.pdf inside upload_dir. Skip writing if it already exists."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_path(file_hash, upload_dir)
    if not path.exists():
        path.write_bytes(data)
    return path
