"""Answer a question from one document's chunks, with page citations (note 0015).

Not found has two layers:
1. Retrieval: no chunk, or the best score is below `min_score` -> not found, and the
   LLM is not called (no cost, no chance to guess).
2. The LLM: the sources are on topic but do not hold the answer -> it replies NOT_FOUND.
   Measured scores show why layer 1 alone is not enough: in-domain questions the paper
   cannot answer score 0.61-0.70, close to answerable ones.
"""

import logging
import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from adaptrag.answering.prompt import NOT_FOUND_TOKEN, build_prompt
from adaptrag.embedding.base import Embedder
from adaptrag.llm.base import LLM
from adaptrag.retrieval.dense import retrieve_dense
from adaptrag.retrieval.results import ScoredChunk

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "Not found in your documents."

# "[3]" -> 3. Only this exact form counts; the prompt asks for it.
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True, slots=True)
class Citation:
    """One cited source: its number in the prompt, and the chunk and pages it stands for."""

    source: int
    chunk_id: int
    page_start: int
    page_end: int


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """found: False means the answer is NOT_FOUND_MESSAGE.

    sources: the retrieved chunks in prompt order (source n = sources[n - 1]), also when
    nothing was found, so a caller can show what was searched.
    """

    found: bool
    answer: str
    citations: list[Citation]
    sources: list[ScoredChunk]


def parse_citations(reply: str, sources: list[ScoredChunk]) -> list[Citation]:
    """Citations in order of first appearance; numbers outside 1..len(sources) are dropped."""
    citations: list[Citation] = []
    seen: set[int] = set()
    for match in _CITATION_PATTERN.finditer(reply):
        number = int(match.group(1))
        if not 1 <= number <= len(sources) or number in seen:
            continue
        seen.add(number)
        source = sources[number - 1]
        citations.append(
            Citation(
                source=number,
                chunk_id=source.chunk_id,
                page_start=source.page_start,
                page_end=source.page_end,
            )
        )
    return citations


def answer_question(
    session: Session,
    chunk_set_id: int,
    question: str,
    k: int,
    embedder: Embedder,
    llm: LLM,
    min_score: float,
) -> AnswerResult:
    """Retrieve the top k chunks, then answer from them or say not found.

    Raises LLMError if the LLM fails; the caller decides what the user sees.
    """
    sources = retrieve_dense(session, chunk_set_id, embedder, question, k)

    # Layer 1: weak or empty retrieval. The LLM is not called.
    if not sources or sources[0].score < min_score:
        top = f"{sources[0].score:.4f}" if sources else "none"
        logger.info("Not found by retrieval: top score %s < min_score %.2f", top, min_score)
        return AnswerResult(found=False, answer=NOT_FOUND_MESSAGE, citations=[], sources=sources)

    system, user = build_prompt(question, sources)
    reply = llm.complete(system, user).strip()

    # Layer 2: the model says the sources do not contain the answer.
    if reply == NOT_FOUND_TOKEN:
        logger.info("Not found by the LLM (top score %.4f)", sources[0].score)
        return AnswerResult(found=False, answer=NOT_FOUND_MESSAGE, citations=[], sources=sources)

    citations = parse_citations(reply, sources)
    if not citations:
        # Still returned (the text may be right), but it cannot be checked against pages.
        logger.warning("Answer has no valid citation (chunk set %d)", chunk_set_id)
    return AnswerResult(found=True, answer=reply, citations=citations, sources=sources)
