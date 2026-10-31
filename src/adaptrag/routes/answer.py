"""POST /documents/{document_id}/answer: an LLM answer with page citations (note 0015).

Same body, validation, 404 and 409 checks as /query. /query stays retrieval-only.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from adaptrag.answering.answer import answer_question
from adaptrag.config import Settings, get_settings
from adaptrag.db import get_session
from adaptrag.embedding.base import Embedder
from adaptrag.embedding.provider import get_app_embedder
from adaptrag.llm.base import LLM, LLMError
from adaptrag.llm.provider import get_app_llm
from adaptrag.models import Document
from adaptrag.retrieval.dense import NEEDS_REINGESTION, current_chunk_set_id, not_ready_message
from adaptrag.routes.query import QueryRequest
from adaptrag.vectorstore.pgvector import embeddings_exist

logger = logging.getLogger(__name__)

router = APIRouter()

ANSWER_UNAVAILABLE = "The answer service is unavailable. Try again later."


class AnswerCitation(BaseModel):
    source: int
    chunk_id: int
    page_start: int
    page_end: int


class AnswerSource(BaseModel):
    source: int
    chunk_id: int
    chunk_index: int
    page_start: int
    page_end: int
    score: float
    text: str


class AnswerResponse(BaseModel):
    document_id: int
    found: bool
    answer: str
    citations: list[AnswerCitation]
    sources: list[AnswerSource]


@router.post("/documents/{document_id}/answer")
def answer_document(
    document_id: int,
    body: QueryRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_app_embedder)],
    llm: Annotated[LLM, Depends(get_app_llm)],
) -> AnswerResponse:
    """Answer from the document's top-k chunks, or say "Not found in your documents."."""
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if document.status != "ready":
        raise HTTPException(status_code=409, detail=not_ready_message(document.status))

    chunk_set_id = current_chunk_set_id(session, document_id)
    if chunk_set_id is None or not embeddings_exist(session, chunk_set_id, embedder.model_name):
        raise HTTPException(status_code=409, detail=NEEDS_REINGESTION)

    try:
        result = answer_question(
            session,
            chunk_set_id,
            body.question,
            body.k,
            embedder,
            llm,
            settings.retrieval_min_score,
        )
    except LLMError as exc:
        # The real cause (with the SDK error chained) goes to the log, not to the user.
        logger.error("Answer failed for document %d: %s", document_id, exc, exc_info=exc)
        raise HTTPException(status_code=502, detail=ANSWER_UNAVAILABLE) from exc

    return AnswerResponse(
        document_id=document_id,
        found=result.found,
        answer=result.answer,
        citations=[
            AnswerCitation(
                source=citation.source,
                chunk_id=citation.chunk_id,
                page_start=citation.page_start,
                page_end=citation.page_end,
            )
            for citation in result.citations
        ],
        sources=[
            AnswerSource(
                source=number,
                chunk_id=source.chunk_id,
                chunk_index=source.chunk_index,
                page_start=source.page_start,
                page_end=source.page_end,
                score=source.score,
                text=source.text,
            )
            for number, source in enumerate(result.sources, start=1)
        ],
    )
