"""Dense retrieval: embed the question, then find the nearest chunk vectors (note 0014).

Only the chunk set that ingestion would create now is searched. Until the router exists
(Stage 3), that is the fixed-size set with today's params and parser version. An older
set (other params or parser version) is never searched: the document needs re-ingestion.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from adaptrag.chunking.fixed import STRATEGY_NAME
from adaptrag.embedding.base import Embedder
from adaptrag.ingest import FIXED_SIZE_PARAMS
from adaptrag.models import ChunkSet
from adaptrag.parsing.pdf import PARSER_VERSION
from adaptrag.retrieval.results import ScoredChunk
from adaptrag.vectorstore.pgvector import search

# Limits shared by POST /documents/{id}/query and scripts/query.py.
DEFAULT_K = 5
MAX_K = 20
MAX_QUESTION_CHARS = 1000

# Error messages shared by the endpoint and the script.
NEEDS_REINGESTION = "Document needs re-ingestion for the current settings"


def not_ready_message(status: str) -> str:
    return f"Document is not ready (status: {status})"


def current_chunk_set_id(session: Session, document_id: int) -> int | None:
    """The id of the chunk set ingestion would create now for this document, or None."""
    return session.scalar(
        select(ChunkSet.id).where(
            ChunkSet.document_id == document_id,
            ChunkSet.strategy == STRATEGY_NAME,
            ChunkSet.params == FIXED_SIZE_PARAMS,
            ChunkSet.parser_version == PARSER_VERSION,
        )
    )


def retrieve_dense(
    session: Session, chunk_set_id: int, embedder: Embedder, question: str, k: int
) -> list[ScoredChunk]:
    """The k chunks of the set most similar to the question, best first.

    The question is embedded as a query (with the model's query instruction); it is
    compared only with vectors made by the same model.
    """
    query_vector = embedder.embed_query(question)
    return search(session, chunk_set_id, embedder.model_name, query_vector, k)
