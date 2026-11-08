"""Run the real ingestion pipeline over every PDF in a dev folder (developer tool).

Each PDF is parsed, profiled, routed, chunked and embedded with the real model, through
the same run_ingest_job() the API uses. A PDF already in the database is reused and gets
a new job; anything already stored (same chunk set, same model) is skipped, not duplicated.
Reads the database URL and embedding settings from settings (ADAPTRAG_* or .env).
Run it from the repo root: uploads live in data_dir / "uploads", as in routes/documents.py.

Only dev documents may be routed and inspected here: any path inside data/test_set is
refused (note 0017).

Usage: python scripts/ingest_folder.py data/samples
"""

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from adaptrag.config import get_settings
from adaptrag.db import get_engine
from adaptrag.embedding.base import Embedder
from adaptrag.embedding.sentence_transformer import get_embedder
from adaptrag.ingest import run_ingest_job
from adaptrag.models import Chunk, Document, Job
from adaptrag.retrieval.dense import current_chunk_set_id
from adaptrag.routes.documents import find_by_hash
from adaptrag.storage import compute_sha256, save_upload
from adaptrag.strategies.registry import build_strategy

TEST_SET = (Path(__file__).resolve().parents[1] / "data" / "test_set").resolve()

# Label in the final counts for documents whose job failed.
FAILED = "(failed)"


def queue_job(engine: Engine, filename: str, size_bytes: int, file_hash: str) -> int:
    """Find or create the document for this hash, add a queued job, and return the job id."""
    with Session(engine) as session:
        document = find_by_hash(session, file_hash)
        if document is None:
            document = Document(file_hash=file_hash, filename=filename, size_bytes=size_bytes)
            session.add(document)
            session.flush()  # send the INSERT now, so document.id exists for the job
        job = Job(document_id=document.id)
        session.add(job)
        session.commit()  # document and job are saved together, or not at all
        return job.id


def count_current_chunks(
    session: Session, document_id: int, strategy_name: str, embedder: Embedder
) -> int:
    """How many chunks the document has in the chunk set its strategy uses today."""
    strategy = build_strategy(strategy_name, embedder)
    chunk_set_id = current_chunk_set_id(session, document_id, strategy.name, strategy.params)
    count = session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.chunk_set_id == chunk_set_id)
    )
    return count or 0


def print_result(
    engine: Engine, job_id: int, filename: str, seconds: float, embedder: Embedder
) -> str:
    """Print the outcome of one job. Returns the strategy, or FAILED, for the final counts."""
    with Session(engine) as session:
        job = session.get_one(Job, job_id)
        document = session.get_one(Document, job.document_id)

        if job.status != "succeeded" or document.strategy is None:
            print(f"document {document.id}: {filename}, job {job.status}, {seconds:.1f}s")
            print(f"  error: {job.error}")
            return FAILED

        chunks = count_current_chunks(session, document.id, document.strategy, embedder)
        print(
            f"document {document.id}: {filename}, job {job.status}, "
            f"strategy {document.strategy}, rule {document.route_rule}, "
            f"{chunks} chunks, {seconds:.1f}s"
        )
        print(f"  reason: {document.route_reason}")
        return document.strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest every PDF in a dev folder.")
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    folder = args.folder.resolve()
    if folder.is_relative_to(TEST_SET):
        print("Error: this script is for dev documents only; data/test_set is not allowed.")
        sys.exit(1)

    paths = sorted(folder.glob("*.pdf"))
    if not paths:
        print(f"Error: no PDF files in {folder}")
        sys.exit(1)

    # Show the job's INFO and WARNING messages (skips, chunks over the token limit).
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    settings = get_settings()
    engine = get_engine(settings.database_url)
    upload_dir = settings.data_dir / "uploads"

    run_start = time.perf_counter()

    # Load the model before creating any job, so a failed load leaves no stuck "queued" job.
    embedder = get_embedder(
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_query_instruction,
    )

    # Keyed by file hash, so two identical files in the folder count as one document.
    outcomes: dict[str, str] = {}
    for path in paths:
        data = path.read_bytes()
        file_hash = compute_sha256(data)
        save_upload(data, file_hash, upload_dir)  # writes only if the file is missing
        job_id = queue_job(engine, path.name, len(data), file_hash)

        job_start = time.perf_counter()
        run_ingest_job(engine, upload_dir, job_id, embedder)
        seconds = time.perf_counter() - job_start

        outcomes[file_hash] = print_result(engine, job_id, path.name, seconds, embedder)

    print()
    print("documents per strategy:")
    for strategy, count in sorted(Counter(outcomes.values()).items()):
        print(f"  {strategy}: {count}")
    print(f"total time: {time.perf_counter() - run_start:.1f}s (includes loading the model)")


if __name__ == "__main__":
    main()
