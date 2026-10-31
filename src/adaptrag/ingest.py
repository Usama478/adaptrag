"""Ingestion job: parse an uploaded PDF, chunk it, and store the chunks.

`run_ingest_job` is a plain function on purpose (the seam in CLAUDE.md):
FastAPI BackgroundTasks calls it today; a Redis-backed queue can call the
same function later.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from adaptrag.chunking.chunks import TextChunk
from adaptrag.chunking.fixed import (
    DEFAULT_OVERLAP_WORDS,
    DEFAULT_SIZE_WORDS,
    STRATEGY_NAME,
    chunk_fixed_size,
)
from adaptrag.models import Chunk, ChunkSet, Document, Job
from adaptrag.parsing.pdf import ParseError, parse_pdf
from adaptrag.storage import upload_path

logger = logging.getLogger(__name__)

INTERNAL_ERROR = "Processing failed because of an internal error."

# Until the router exists (Stage 3), every document gets the fixed-size control strategy.
# The same dict is passed to the chunker and stored, so the stored settings are always
# exactly the settings that were used.
FIXED_SIZE_PARAMS: dict[str, Any] = {
    "size_words": DEFAULT_SIZE_WORDS,
    "overlap_words": DEFAULT_OVERLAP_WORDS,
}


def finish(job: Job, document: Document, succeeded: bool, error: str | None) -> None:
    """Record the end of a job on both the job and its document."""
    job.status = "succeeded" if succeeded else "failed"
    job.error = error
    job.finished_at = datetime.now(UTC)
    document.status = "parsed" if succeeded else "failed"


def save_chunk_set(
    session: Session,
    document_id: int,
    strategy: str,
    params: dict[str, Any],
    parser_version: str,
    chunks: list[TextChunk],
) -> bool:
    """Add a chunk set and its chunks to the session (the caller commits).

    Returns False, and adds nothing, if an identical set already exists.
    """
    existing = session.scalar(
        select(ChunkSet.id).where(
            ChunkSet.document_id == document_id,
            ChunkSet.strategy == strategy,
            ChunkSet.params == params,
            ChunkSet.parser_version == parser_version,
        )
    )
    if existing is not None:
        return False

    chunk_set = ChunkSet(
        document_id=document_id,
        strategy=strategy,
        params=params,
        parser_version=parser_version,
    )
    session.add(chunk_set)
    session.flush()
    session.add_all(
        Chunk(
            chunk_set_id=chunk_set.id,
            chunk_index=chunk.index,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )
        for chunk in chunks
    )
    return True


def run_ingest_job(engine: Engine, upload_dir: Path, job_id: int) -> None:
    """Parse and chunk the job's document, then mark the job succeeded or failed."""
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
            chunks = chunk_fixed_size(parsed, **FIXED_SIZE_PARAMS)
            saved = save_chunk_set(
                session,
                document.id,
                STRATEGY_NAME,
                FIXED_SIZE_PARAMS,
                parsed.parser_version,
                chunks,
            )
            if not saved:
                logger.info("Document %s already has this chunk set; skipped", document.id)
        except ParseError as exc:
            session.rollback()
            finish(job, document, succeeded=False, error=str(exc))
        except Exception:
            session.rollback()
            logger.exception("Ingest job %s crashed", job_id)
            finish(job, document, succeeded=False, error=INTERNAL_ERROR)
        else:
            document.page_count = parsed.page_count
            finish(job, document, succeeded=True, error=None)
        session.commit()
