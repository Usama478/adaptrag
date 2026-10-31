"""Ask one document a question and print the LLM answer with page citations.

Same checks and messages as POST /documents/{id}/answer, printed as plain text.
Uses the real embedder and the real LLM from settings (ADAPTRAG_* or .env). Token usage
is logged at INFO by llm/openai_client.py. The API key is never printed.

Usage: python scripts/answer.py DOCUMENT_ID "QUESTION" [--k 5]
"""

import argparse
import logging

from sqlalchemy.orm import Session

from adaptrag.answering.answer import answer_question
from adaptrag.config import get_settings
from adaptrag.db import get_engine
from adaptrag.embedding.sentence_transformer import get_embedder
from adaptrag.llm.base import LLMError
from adaptrag.llm.openai_client import get_openai_llm
from adaptrag.llm.provider import NOT_CONFIGURED
from adaptrag.models import Document
from adaptrag.retrieval.dense import (
    DEFAULT_K,
    MAX_K,
    MAX_QUESTION_CHARS,
    NEEDS_REINGESTION,
    current_chunk_set_id,
    not_ready_message,
)
from adaptrag.vectorstore.pgvector import embeddings_exist


def pages(start: int, end: int) -> str:
    return f"p. {start}" if start == end else f"pp. {start}-{end}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print an LLM answer with page citations.")
    parser.add_argument("document_id", type=int, help="id of a ready document")
    parser.add_argument("question", help="the question, in quotes")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"sources, 1-{MAX_K}")
    args = parser.parse_args()

    question = args.question.strip()
    if not 1 <= len(question) <= MAX_QUESTION_CHARS:
        raise SystemExit(f"The question must have 1-{MAX_QUESTION_CHARS} characters.")
    if not 1 <= args.k <= MAX_K:
        raise SystemExit(f"--k must be between 1 and {MAX_K}.")

    settings = get_settings()
    # Shows the INFO token-usage line (and httpx's request line, which has no key).
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    key = settings.openai_api_key
    if key is None or not key.get_secret_value().strip():
        raise SystemExit(NOT_CONFIGURED)

    engine = get_engine(settings.database_url)
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
        llm = get_openai_llm(
            key.get_secret_value(),
            settings.llm_model,
            settings.llm_timeout_seconds,
            settings.llm_max_output_tokens,
            settings.llm_temperature,
        )
        try:
            result = answer_question(
                session,
                chunk_set_id,
                question,
                args.k,
                embedder,
                llm,
                settings.retrieval_min_score,
            )
        except LLMError as exc:
            raise SystemExit(f"LLM failed: {exc} (cause: {exc.__cause__!r})") from exc

    print(f"document {document.id} ({document.filename}), {model_name} + {settings.llm_model}")
    print(f"min_score {settings.retrieval_min_score}")
    print(f"found: {result.found}")
    print(f"answer: {result.answer}")
    print("citations:")
    for citation in result.citations:
        page_text = pages(citation.page_start, citation.page_end)
        print(f"  [{citation.source}] chunk {citation.chunk_id}, {page_text}")
    if not result.citations:
        print("  (none)")
    print("top scores:")
    for number, source in enumerate(result.sources, start=1):
        page_text = pages(source.page_start, source.page_end)
        print(f"  [{number}] {source.score:.4f}  {page_text}  chunk {source.chunk_id}")


if __name__ == "__main__":
    main()
