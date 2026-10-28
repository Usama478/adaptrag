from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from adaptrag.config import Settings, get_settings
from adaptrag.db import get_session
from adaptrag.models import Document
from adaptrag.storage import compute_sha256, save_upload

router = APIRouter()

PDF_MAGIC = b"%PDF-"


def find_by_hash(session: Session, file_hash: str) -> Document | None:
    """Return the document with this file hash, or None if it's new."""
    return session.scalar(select(Document).where(Document.file_hash == file_hash))


def document_response(document: Document, duplicate: bool) -> dict[str, str | int | bool]:
    """The JSON shape returned for both new and duplicate uploads."""
    return {
        "document_id": document.id,
        "filename": document.filename,
        "file_hash": document.file_hash,
        "status": document.status,
        "duplicate": duplicate,
    }


@router.post("/documents", status_code=202)
def upload_document(
    file: UploadFile,
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, str | int | bool]:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only .pdf files are accepted.")

    data = file.file.read()

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum is {settings.max_upload_mb} MB.",
        )

    if not data.startswith(PDF_MAGIC):
        raise HTTPException(status_code=415, detail="This file is not a valid PDF.")

    file_hash = compute_sha256(data)

    existing = find_by_hash(session, file_hash)
    if existing is not None:
        response.status_code = 200
        return document_response(existing, duplicate=True)

    save_upload(data, file_hash, settings.data_dir / "uploads")
    document = Document(file_hash=file_hash, filename=filename, size_bytes=len(data))
    session.add(document)
    try:
        session.commit()
    except IntegrityError:
        # Another request stored the same file between our lookup and our commit.
        session.rollback()
        existing = find_by_hash(session, file_hash)
        if existing is None:
            raise
        response.status_code = 200
        return document_response(existing, duplicate=True)

    return document_response(document, duplicate=False)
