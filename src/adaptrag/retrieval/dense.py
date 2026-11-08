"""Dense retrieval: embed the question, then find the nearest chunk vectors (note 0014).

Only the chunk set that ingestion would create now is searched: the caller's strategy
with its params, and today's parser version. An older set (other params or parser
version) is never searched: the document needs re-ingestion.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from adaptrag.embedding.base import Embedder
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


def current_chunk_set_id(
    session: Session, document_id: int, strategy_name: str, params: dict[str, Any]
) -> int | None:
    """The id of the chunk set ingestion would create now with these settings, or None.

    Takes a name and params, not a Strategy: strategies import this module, and the
    scripts make this check before the embedding model (which a Strategy needs) loads.
    """
    return session.scalar(
        select(ChunkSet.id).where(
            ChunkSet.document_id == document_id,
            ChunkSet.strategy == strategy_name,
            ChunkSet.params == params,
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
