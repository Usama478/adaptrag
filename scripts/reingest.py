"""Process an existing document again: new job, parse, chunk, embed with the real model.

Useful for documents ingested before a step existed (for example, before embeddings).
Anything already stored (same chunk set, same model) is skipped, not duplicated.
Reads the database URL and embedding settings from settings (ADAPTRAG_* or .env).
Run it from the repo root: uploads are read from data_dir / "uploads", as in
routes/documents.py.

Usage: python scripts/reingest.py DOCUMENT_ID
"""

import argparse
import logging

from sqlalchemy.orm import Session

from adaptrag.config import get_settings
from adaptrag.db import get_engine
from adaptrag.embedding.sentence_transformer import get_embedder
from adaptrag.ingest import run_ingest_job
from adaptrag.models import Document, Job


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run ingestion for one document.")
    parser.add_argument("document_id", type=int, help="id of an existing document")
    args = parser.parse_args()

    # Show the job's INFO and WARNING messages (skips, chunks over the token limit).
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    settings = get_settings()
    engine = get_engine(settings.database_url)

    with Session(engine) as session:
        if session.get(Document, args.document_id) is None:
            raise SystemExit(f"document {args.document_id} not found")

    # Load the model before creating the job, so a failed load leaves no stuck "queued" job.
    embedder = get_embedder(
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_query_instruction,
    )

    with Session(engine) as session:
        job = Job(document_id=args.document_id)
        session.add(job)
        session.commit()
        job_id = job.id

    upload_dir = settings.data_dir / "uploads"
    run_ingest_job(engine, upload_dir, job_id, embedder)

    with Session(engine) as session:
        job = session.get_one(Job, job_id)
        print(f"job {job.id}: status {job.status}, error {job.error}")


if __name__ == "__main__":
    main()
