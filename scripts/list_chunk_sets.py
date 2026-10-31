"""Print every stored chunk set, how many chunks it holds, and how many are embedded.

Embeddings are counted for the configured model only (settings.embedding_model).
Reads the database URL from settings (ADAPTRAG_DATABASE_URL or .env).

Usage: python scripts/list_chunk_sets.py
"""

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from adaptrag.config import get_settings
from adaptrag.db import get_engine
from adaptrag.models import Chunk, ChunkEmbedding, ChunkSet


def main() -> None:
    settings = get_settings()
    engine = get_engine(settings.database_url)
    model_name = settings.embedding_model

    chunk_count = func.count(Chunk.id)
    # At most one embedding per chunk per model (primary key), so this join does not
    # repeat chunks and both counts stay correct.
    embedding_count = func.count(ChunkEmbedding.chunk_id)
    query = (
        select(ChunkSet, chunk_count, embedding_count)
        .outerjoin(Chunk, Chunk.chunk_set_id == ChunkSet.id)
        .outerjoin(
            ChunkEmbedding,
            and_(ChunkEmbedding.chunk_id == Chunk.id, ChunkEmbedding.model_name == model_name),
        )
        .group_by(ChunkSet.id)
        .order_by(ChunkSet.id)
    )
    print(f"embeddings counted for model {model_name}")
    with Session(engine) as session:
        for chunk_set, chunks, embeddings in session.execute(query):
            print(
                f"document {chunk_set.document_id}: {chunk_set.strategy} {chunk_set.params} "
                f"parser {chunk_set.parser_version}, {chunks} chunks, {embeddings} embeddings"
            )


if __name__ == "__main__":
    main()
