from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from adaptrag.db import get_session
from adaptrag.models import Job

router = APIRouter()


@router.get("/jobs/{job_id}")
def get_job(
    job_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, str | int | None]:
    """Report a job's status, and its error message if it failed."""
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job.id,
        "document_id": job.document_id,
        "status": job.status,
        "error": job.error,
    }
