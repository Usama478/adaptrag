"""Print every stored chunk set and how many chunks it holds.

Reads the database URL from settings (ADAPTRAG_DATABASE_URL or .env).

Usage: python scripts/list_chunk_sets.py
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from adaptrag.config import get_settings
from adaptrag.db import get_engine
from adaptrag.models import Chunk, ChunkSet


def main() -> None:
    engine = get_engine(get_settings().database_url)
    chunk_count = func.count(Chunk.id)
    query = (
        select(ChunkSet, chunk_count)
        .outerjoin(Chunk, Chunk.chunk_set_id == ChunkSet.id)
        .group_by(ChunkSet.id)
        .order_by(ChunkSet.id)
    )
    with Session(engine) as session:
        for chunk_set, count in session.execute(query):
            print(
                f"document {chunk_set.document_id}: {chunk_set.strategy} {chunk_set.params} "
                f"parser {chunk_set.parser_version}, {count} chunks"
            )


if __name__ == "__main__":
    main()
