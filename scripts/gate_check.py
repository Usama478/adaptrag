"""Stage 2 gate: answer every verified question of one paper and score the answers.

Runs the real pipeline exactly as POST /documents/{id}/answer does: same checks, current
chunk set, real embedder and real LLM from settings, retrieval_min_score from settings.
Scoring rules live in src/adaptrag/evaluation/gate.py (unit-tested). Token usage is
logged at INFO by llm/openai_client.py. The API key is never printed.

Each scored question is one real LLM call (costs money). Seconds per answer cover
answer_question only (query embedding, search, LLM call); the model is loaded before
the first question, so its load time is not included.

Usage: python scripts/gate_check.py QUESTIONS_FILE --document-id ID [--k 5]
"""

import argparse
import logging
import time
from pathlib import Path

from sqlalchemy.orm import Session

from adaptrag.answering.answer import answer_question
from adaptrag.config import get_settings
from adaptrag.db import get_engine
from adaptrag.embedding.sentence_transformer import get_embedder
from adaptrag.evaluation.gate import (
    GATE_MIN_CITED,
    GATE_MIN_QUESTIONS,
    NOT_VERIFIED,
    QuestionFileError,
    cited_ok,
    gate_passed,
    load_questions,
    retrieved_ok,
)
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


def page_range(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score verified questions (Stage 2 gate).")
    parser.add_argument("questions_file", type=Path, help="e.g. eval/questions/paper_01.json")
    parser.add_argument("--document-id", type=int, required=True, help="id of a ready document")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"sources, 1-{MAX_K}")
    args = parser.parse_args()

    if not 1 <= args.k <= MAX_K:
        raise SystemExit(f"--k must be between 1 and {MAX_K}.")
    try:
        questions, skipped = load_questions(args.questions_file)
    except (OSError, QuestionFileError) as exc:
        raise SystemExit(f"Cannot load questions: {exc}") from exc
    for question in questions:
        if not 1 <= len(question.question.strip()) <= MAX_QUESTION_CHARS:
            raise SystemExit(f"{question.id}: the question must have 1-{MAX_QUESTION_CHARS} chars.")

    settings = get_settings()
    # Shows the INFO token-usage line (and httpx's request line, which has no key).
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    key = settings.openai_api_key
    if key is None or not key.get_secret_value().strip():
        raise SystemExit(NOT_CONFIGURED)

    engine = get_engine(settings.database_url)
    model_name = settings.embedding_model

    for entry in skipped:
        label = "SKIPPED (not verified)" if entry.reason == NOT_VERIFIED else "REFUSED"
        detail = "" if entry.reason == NOT_VERIFIED else f" ({entry.reason})"
        print(f"{entry.id}  {label}{detail}")
    if not questions:
        raise SystemExit("No verified questions to score.")

    cited_count = 0
    retrieved_count = 0
    seconds: list[float] = []

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
        print(f"document {document.id} ({document.filename}), {model_name} + {settings.llm_model}")
        print(f"min_score {settings.retrieval_min_score}, k {args.k}")

        for question in questions:
            evidence_pages = [item.page for item in question.evidence]
            evidence_texts = [item.text for item in question.evidence]

            started = time.perf_counter()
            try:
                result = answer_question(
                    session,
                    chunk_set_id,
                    question.question.strip(),
                    args.k,
                    embedder,
                    llm,
                    settings.retrieval_min_score,
                )
            except LLMError as exc:
                raise SystemExit(
                    f"{question.id}: LLM failed: {exc} (cause: {exc.__cause__!r})"
                ) from exc
            elapsed = time.perf_counter() - started
            seconds.append(elapsed)

            is_cited = cited_ok(result.found, result.citations, evidence_pages)
            is_retrieved = retrieved_ok([source.text for source in result.sources], evidence_texts)
            if is_cited:
                cited_count += 1
            if is_retrieved:
                retrieved_count += 1

            cited_pages = ", ".join(page_range(c.page_start, c.page_end) for c in result.citations)
            answer_start = " ".join(result.answer.split())[:100]
            print(
                f"{question.id}  cited_ok={is_cited}  retrieved_ok={is_retrieved}  "
                f"cited [{cited_pages}]  evidence {evidence_pages}  "
                f"{elapsed:.2f}s  {answer_start}"
            )

    total = len(questions)
    print()
    print(f"cited the right page: {cited_count}/{total}")
    print(f"evidence in top-k: {retrieved_count}/{total}")
    verdict = "PASS" if gate_passed(cited_count, total) else "FAIL"
    note = "" if total >= GATE_MIN_QUESTIONS else f" (only {total} questions scored)"
    print(f"gate (>= {GATE_MIN_CITED}/{GATE_MIN_QUESTIONS}): {verdict}{note}")
    print(f"seconds per answer: mean {sum(seconds) / total:.2f}, max {max(seconds):.2f}")


if __name__ == "__main__":
    main()
