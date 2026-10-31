"""POST /documents/{document_id}/query: the top-k chunks for a question (no LLM yet).

Note 0014. No "weak retrieval" threshold here: Step 6 chooses one from measured scores.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy.orm import Session

from adaptrag.db import get_session
from adaptrag.embedding.base import Embedder
from adaptrag.embedding.provider import get_app_embedder
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

router = APIRouter()


class QueryRequest(BaseModel):
    # Whitespace is stripped before the length check, so "   " is rejected with 422.
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
    ]
    k: int = Field(default=DEFAULT_K, ge=1, le=MAX_K)


class QueryResult(BaseModel):
    chunk_id: int
    chunk_index: int
    page_start: int
    page_end: int
    score: float
    text: str


class QueryResponse(BaseModel):
    # Pydantic reserves names starting with "model_" by default; `model_name` is ours.
    model_config = ConfigDict(protected_namespaces=())

    document_id: int
    chunk_set_id: int
    model_name: str
    results: list[QueryResult]


@router.post("/documents/{document_id}/query")
def query_document(
    document_id: int,
    body: QueryRequest,
    session: Annotated[Session, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_app_embedder)],
) -> QueryResponse:
    """Return the k chunks of a ready document most similar to the question, best first."""
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if document.status != "ready":
        raise HTTPException(status_code=409, detail=not_ready_message(document.status))

    chunk_set_id = current_chunk_set_id(session, document_id)
    if chunk_set_id is None or not embeddings_exist(session, chunk_set_id, embedder.model_name):
        raise HTTPException(status_code=409, detail=NEEDS_REINGESTION)

    results = retrieve_dense(session, chunk_set_id, embedder, body.question, body.k)
    return QueryResponse(
        document_id=document_id,
        chunk_set_id=chunk_set_id,
        model_name=embedder.model_name,
        results=[
            QueryResult(
                chunk_id=result.chunk_id,
                chunk_index=result.chunk_index,
                page_start=result.page_start,
                page_end=result.page_end,
                score=result.score,
                text=result.text,
            )
            for result in results
        ],
    )
