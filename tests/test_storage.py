from pathlib import Path

from adaptrag.storage import compute_sha256, save_upload

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_hash_matches_known_value() -> None:
    assert compute_sha256(b"abc") == ABC_SHA256


def test_same_content_gives_same_hash() -> None:
    assert compute_sha256(b"hello") == compute_sha256(b"hello")


def test_one_byte_change_gives_different_hash() -> None:
    assert compute_sha256(b"hello") != compute_sha256(b"hello!")


def test_save_upload_writes_file(tmp_path: Path) -> None:
    path = save_upload(b"%PDF-data", "abc123", tmp_path / "uploads")
    assert path == tmp_path / "uploads" / "abc123.pdf"
    assert path.read_bytes() == b"%PDF-data"


def test_saving_twice_keeps_one_file(tmp_path: Path) -> None:
    upload_dir = tmp_path / "uploads"
    save_upload(b"%PDF-data", "abc123", upload_dir)
    save_upload(b"%PDF-data", "abc123", upload_dir)
    assert len(list(upload_dir.iterdir())) == 1
