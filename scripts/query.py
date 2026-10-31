"""Ask one document a question and print the most similar chunks (dense retrieval, no LLM).

Same checks and messages as POST /documents/{id}/query, printed as plain text.
Uses the real embedder from settings (ADAPTRAG_* or .env); the model loads only after
the document passes the checks.

Usage: python scripts/query.py DOCUMENT_ID "QUESTION" [--k 5]
"""

import argparse

from sqlalchemy.orm import Session

from adaptrag.config import get_settings
from adaptrag.db import get_engine
from adaptrag.embedding.sentence_transformer import get_embedder
from adaptrag.models import Document
from adaptrag.retrieval.dense import (
    DEFAULT_K,
    MAX_K,
    MAX_QUESTION_CHARS,
    NEEDS_REINGESTION,
    current_chunk_set_id,
    not_ready_message,
    retrieve_dense,
)
from adaptrag.vectorstore.pgvector import embeddings_exist

PREVIEW_CHARS = 150


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the top chunks for a question.")
    parser.add_argument("document_id", type=int, help="id of a ready document")
    parser.add_argument("question", help="the question, in quotes")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"results, 1-{MAX_K}")
    args = parser.parse_args()

    # The same limits as the endpoint's request body.
    question = args.question.strip()
    if not 1 <= len(question) <= MAX_QUESTION_CHARS:
        raise SystemExit(f"The question must have 1-{MAX_QUESTION_CHARS} characters.")
    if not 1 <= args.k <= MAX_K:
        raise SystemExit(f"--k must be between 1 and {MAX_K}.")

    settings = get_settings()
    engine = get_engine(settings.database_url)
    # The real embedder's model_name is the model setting, so the checks below need no
    # model; it is loaded only when the document can be searched.
    model_name = settings.embedding_model

    with Session(engine) as session:
        document = session.get(Document, args.document_id)
        if document is None:
            raise SystemExit("Document not found.")
        if document.status != "ready":
            raise SystemExit(not_ready_message(document.status))
        chunk_set_id = current_chunk_set_id(session, document.id)
        if chunk_set_id is None or not embeddings_exist(session, chunk_set_id, model_name):
            raise SystemExit(NEEDS_REINGESTION)

        embedder = get_embedder(
            settings.embedding_model,
            settings.embedding_device,
            settings.embedding_query_instruction,
        )
        results = retrieve_dense(session, chunk_set_id, embedder, question, args.k)

    print(f"document {document.id} ({document.filename}), chunk set {chunk_set_id}, {model_name}")
    for rank, result in enumerate(results, start=1):
        if result.page_start == result.page_end:
            pages = f"p. {result.page_start}"
        else:
            pages = f"pp. {result.page_start}-{result.page_end}"
        print(f"{rank}. {result.score:.4f}  {pages}  {result.text[:PREVIEW_CHARS]}")


if __name__ == "__main__":
    main()
