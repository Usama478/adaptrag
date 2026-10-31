"""Save chunk vectors in Postgres and check whether a chunk set is already embedded.

The embedding cache key is structural (note 0013):
document (file hash) -> chunk set (parser version, strategy, params) -> model name.
"Already embedded" means rows exist for this chunk set and this model.
"""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from adaptrag.models import EMBEDDING_DIMENSION, Chunk, ChunkEmbedding


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
