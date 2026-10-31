"""Save chunk vectors in Postgres, check whether a chunk set is embedded, and search it.

The embedding cache key is structural (note 0013):
document (file hash) -> chunk set (parser version, strategy, params) -> model name.
"Already embedded" means rows exist for this chunk set and this model.

All vector SQL lives in this module (note 0014), so an index or another store later
changes only this file.
"""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from adaptrag.models import EMBEDDING_DIMENSION, Chunk, ChunkEmbedding
from adaptrag.retrieval.results import ScoredChunk


def embeddings_exist(session: Session, chunk_set_id: int, model_name: str) -> bool:
    """True if any chunk of this chunk set has a vector from this model."""
    query = (
        select(ChunkEmbedding.chunk_id)
        .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
        .where(Chunk.chunk_set_id == chunk_set_id, ChunkEmbedding.model_name == model_name)
        .limit(1)
    )
    return session.scalar(query) is not None


def save_embeddings(
    session: Session,
    model_name: str,
    chunk_ids: Sequence[int],
    vectors: Sequence[Sequence[float]],
) -> None:
    """Add one embedding row per chunk to the session (the caller commits).

    Checks everything before adding anything, so a bad input adds no rows at all.
    """
    if len(chunk_ids) != len(vectors):
        raise ValueError(f"got {len(chunk_ids)} chunk ids but {len(vectors)} vectors")
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"vector has {len(vector)} numbers; the column holds {EMBEDDING_DIMENSION}"
            )

    session.add_all(
        ChunkEmbedding(chunk_id=chunk_id, model_name=model_name, embedding=list(vector))
        for chunk_id, vector in zip(chunk_ids, vectors, strict=True)
    )


def search(
    session: Session,
    chunk_set_id: int,
    model_name: str,
    query_vector: Sequence[float],
    k: int,
) -> list[ScoredChunk]:
    """The k chunks of one chunk set closest to the query vector, best first.

    Exact search (note 0013): Postgres compares the query with every vector of this
    chunk set and model; no approximate index. Only vectors made by `model_name` are
    compared, because vectors from different models are not comparable.

    Order: smallest cosine distance first; equal distances by chunk_index, so the same
    query always returns the same order. score = 1 - distance (cosine similarity).
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if len(query_vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"query vector has {len(query_vector)} numbers; the column holds {EMBEDDING_DIMENSION}"
        )

    # pgvector's `<=>` operator: 1 - cosine similarity, from 0 (same direction) to 2.
    distance = ChunkEmbedding.embedding.cosine_distance(list(query_vector)).label("distance")
    query = (
        select(
            Chunk.id,
            Chunk.chunk_index,
            Chunk.text,
            Chunk.page_start,
            Chunk.page_end,
            distance,
        )
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .where(Chunk.chunk_set_id == chunk_set_id, ChunkEmbedding.model_name == model_name)
        .order_by(distance, Chunk.chunk_index)
        .limit(k)
    )
    return [
        ScoredChunk(
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            text=text,
            page_start=page_start,
            page_end=page_end,
            score=1.0 - float(chunk_distance),
        )
        for chunk_id, chunk_index, text, page_start, page_end, chunk_distance in session.execute(
            query
        )
    ]
