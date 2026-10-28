from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from adaptrag.config import Settings, get_settings
from adaptrag.storage import compute_sha256, save_upload

router = APIRouter()

PDF_MAGIC = b"%PDF-"


@router.post("/documents", status_code=202)
async def upload_document(
    file: UploadFile,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str | int]:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only .pdf files are accepted.")

    data = await file.read()

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum is {settings.max_upload_mb} MB.",
        )

    if not data.startswith(PDF_MAGIC):
        raise HTTPException(status_code=415, detail="This file is not a valid PDF.")

    file_hash = compute_sha256(data)
    save_upload(data, file_hash, settings.data_dir / "uploads")

    return {
        "filename": filename,
        "file_hash": file_hash,
        "size_bytes": len(data),
        "status": "accepted",
    }
