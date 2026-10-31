from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from adaptrag.config import Settings, get_settings
from adaptrag.db import get_engine, get_session
from adaptrag.embedding.base import Embedder
from adaptrag.embedding.provider import get_app_embedder
from adaptrag.ingest import run_ingest_job
from adaptrag.models import Document, Job
from adaptrag.storage import compute_sha256, save_upload

router = APIRouter()

PDF_MAGIC = b"%PDF-"

DocumentResponse = dict[str, str | int | bool | None]


def find_by_hash(session: Session, file_hash: str) -> Document | None:
    """Return the document with this file hash, or None if it's new."""
    return session.scalar(select(Document).where(Document.file_hash == file_hash))


def latest_job_id(session: Session, document_id: int) -> int | None:
    """The newest job for a document, or None (documents uploaded before jobs existed)."""
    return session.scalar(
        select(Job.id).where(Job.document_id == document_id).order_by(Job.id.desc())
    )


def document_response(document: Document, job_id: int | None, duplicate: bool) -> DocumentResponse:
    """The JSON shape returned for both new and duplicate uploads."""
    return {
        "document_id": document.id,
        "job_id": job_id,
        "filename": document.filename,
        "file_hash": document.file_hash,
        "status": document.status,
        "duplicate": duplicate,
    }


@router.post("/documents", status_code=202)
def upload_document(
    file: UploadFile,
    response: Response,
    background_tasks: BackgroundTasks,
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[Session, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_app_embedder)],
) -> DocumentResponse:
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
        return document_response(existing, latest_job_id(session, existing.id), duplicate=True)

    upload_dir = settings.data_dir / "uploads"
    save_upload(data, file_hash, upload_dir)
    document = Document(file_hash=file_hash, filename=filename, size_bytes=len(data))
    session.add(document)
    try:
        session.flush()  # send the INSERT now, so document.id exists for the job
    except IntegrityError:
        # Another request stored the same file between our lookup and our insert.
        session.rollback()
        existing = find_by_hash(session, file_hash)
        if existing is None:
            raise
        response.status_code = 200
        return document_response(existing, latest_job_id(session, existing.id), duplicate=True)

    job = Job(document_id=document.id)
    session.add(job)
    session.commit()  # document and job are saved together, or not at all

    engine = get_engine(settings.database_url)
    background_tasks.add_task(run_ingest_job, engine, upload_dir, job.id, embedder)
    return document_response(document, job.id, duplicate=False)
