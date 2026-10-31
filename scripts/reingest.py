"""Process an existing document again: new job, parse, chunk, embed with the real model.

Useful for documents ingested before a step existed (for example, before embeddings).
Anything already stored (same chunk set, same model) is skipped, not duplicated.
Reads the database URL and embedding settings from settings (ADAPTRAG_* or .env).
Run it from the repo root: uploads are read from data_dir / "uploads", as in
routes/documents.py.

--file PATH puts a missing upload back (for example, a document uploaded through Docker,
whose file is in the Docker volume, not in the local data/uploads). The file must have
the document's SHA-256; otherwise the script stops and changes nothing.

Usage: python scripts/reingest.py DOCUMENT_ID [--file PATH]
"""

import argparse
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from adaptrag.config import get_settings
from adaptrag.db import get_engine
from adaptrag.embedding.sentence_transformer import get_embedder
from adaptrag.ingest import run_ingest_job
from adaptrag.models import Document, Job
from adaptrag.storage import compute_sha256, save_upload, upload_path


def restore_upload(source: Path, expected_hash: str, upload_dir: Path) -> None:
    """Save `source` as the document's upload file if it is missing.

    Stops (SystemExit) before writing anything if `source` is not the document's file.
    """
    if not source.is_file():
        raise SystemExit(f"file not found: {source}")
    data = source.read_bytes()
    actual_hash = compute_sha256(data)
    if actual_hash != expected_hash:
        raise SystemExit(
            f"{source} is not this document's file (SHA-256 {actual_hash}, "
            f"document has {expected_hash}); nothing changed"
        )

    target = upload_path(expected_hash, upload_dir)
    if target.exists():
        print(f"upload file already present: {target}")
    else:
        save_upload(data, expected_hash, upload_dir)
        print(f"upload file saved: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run ingestion for one document.")
    parser.add_argument("document_id", type=int, help="id of an existing document")
    parser.add_argument(
        "--file", type=Path, help="the document's PDF, saved as its upload file if missing"
    )
    args = parser.parse_args()

    # Show the job's INFO and WARNING messages (skips, chunks over the token limit).
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    settings = get_settings()
    engine = get_engine(settings.database_url)
    upload_dir = settings.data_dir / "uploads"

    with Session(engine) as session:
        document = session.get(Document, args.document_id)
        if document is None:
            raise SystemExit(f"document {args.document_id} not found")
        file_hash = document.file_hash

    # Checked before the model loads and before the job exists, so a wrong file
    # changes nothing.
    if args.file is not None:
        restore_upload(args.file, file_hash, upload_dir)

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

    run_ingest_job(engine, upload_dir, job_id, embedder)

    with Session(engine) as session:
        job = session.get_one(Job, job_id)
        print(f"job {job.id}: status {job.status}, error {job.error}")


if __name__ == "__main__":
    main()
