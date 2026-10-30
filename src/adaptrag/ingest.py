"""Ingestion job: parse an uploaded PDF and record the outcome.

`run_ingest_job` is a plain function on purpose (the seam in CLAUDE.md):
FastAPI BackgroundTasks calls it today; a Redis-backed queue can call the
same function later. Stage 2 adds chunking and embedding here.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from adaptrag.models import Document, Job
from adaptrag.parsing.pdf import ParseError, parse_pdf
from adaptrag.storage import upload_path

logger = logging.getLogger(__name__)

INTERNAL_ERROR = "Processing failed because of an internal error."


def finish(job: Job, document: Document, succeeded: bool, error: str | None) -> None:
    """Record the end of a job on both the job and its document."""
    job.status = "succeeded" if succeeded else "failed"
    job.error = error
    job.finished_at = datetime.now(UTC)
    document.status = "parsed" if succeeded else "failed"


def run_ingest_job(engine: Engine, upload_dir: Path, job_id: int) -> None:
    """Parse the job's document and mark the job succeeded or failed."""
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job is None:
            logger.error("Ingest job %s not found", job_id)
            return
        document = session.get_one(Document, job.document_id)

        job.status = "running"
        session.commit()

        try:
            parsed = parse_pdf(upload_path(document.file_hash, upload_dir))
        except ParseError as exc:
            finish(job, document, succeeded=False, error=str(exc))
        except Exception:
            logger.exception("Ingest job %s crashed", job_id)
            finish(job, document, succeeded=False, error=INTERNAL_ERROR)
        else:
            document.page_count = parsed.page_count
            finish(job, document, succeeded=True, error=None)
        session.commit()
