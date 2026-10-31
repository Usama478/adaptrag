"""Ingestion job: parse an uploaded PDF, chunk it, store the chunks, embed them.

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
from adaptrag.embedding.base import Embedder
from adaptrag.models import Chunk, ChunkSet, Document, Job
from adaptrag.parsing.pdf import ParseError, parse_pdf
from adaptrag.storage import upload_path
from adaptrag.vectorstore.pgvector import embeddings_exist, save_embeddings

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
    # "ready" = parsed, chunked and embedded: the document can be searched.
    document.status = "ready" if succeeded else "failed"


def save_chunk_set(
    session: Session,
    document_id: int,
    strategy: str,
    params: dict[str, Any],
    parser_version: str,
    chunks: list[TextChunk],
) -> tuple[int, bool]:
    """Add a chunk set and its chunks to the session (the caller commits).

    Returns (chunk_set_id, created). If an identical set already exists, nothing is
    added and the existing set's id is returned with created=False.
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
        return existing, False

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
    return chunk_set.id, True


def embed_chunk_set(
    session: Session, document_id: int, chunk_set_id: int, embedder: Embedder
) -> None:
    """Embed every chunk of the set with this model, unless that was done before.

    Adds rows to the session; the caller commits.
    """
    if embeddings_exist(session, chunk_set_id, embedder.model_name):
        logger.info(
            "Document %s chunk set %s already embedded with %s; skipped",
            document_id,
            chunk_set_id,
            embedder.model_name,
        )
        return

    # This query first flushes the chunks added by save_chunk_set (autoflush),
    # so their ids exist.
    rows = session.execute(
        select(Chunk.id, Chunk.text)
        .where(Chunk.chunk_set_id == chunk_set_id)
        .order_by(Chunk.chunk_index)
    ).all()
    chunk_ids = [chunk_id for chunk_id, _ in rows]
    texts = [text for _, text in rows]

    # Option A (note 0012): keep the chunk, but say how many the model will cut off.
    over = sum(1 for text in texts if embedder.count_tokens(text) > embedder.max_tokens)
    if over:
        logger.warning(
            "Document %s: %s of %s chunks exceed %s tokens; the model ignores the text "
            "past the limit",
            document_id,
            over,
            len(texts),
            embedder.max_tokens,
        )

    vectors = embedder.embed_passages(texts)
    save_embeddings(session, embedder.model_name, chunk_ids, vectors)


def run_ingest_job(engine: Engine, upload_dir: Path, job_id: int, embedder: Embedder) -> None:
    """Parse, chunk and embed the job's document, then mark the job succeeded or failed.

    Chunks and vectors are saved in one transaction: a failure anywhere leaves neither.
    """
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
            chunk_set_id, created = save_chunk_set(
                session,
                document.id,
                STRATEGY_NAME,
                FIXED_SIZE_PARAMS,
                parsed.parser_version,
                chunks,
            )
            if not created:
                logger.info("Document %s already has this chunk set; skipped", document.id)
            embed_chunk_set(session, document.id, chunk_set_id, embedder)
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
