"""add chunk embeddings

Revision ID: 5c8e1a7d4f2b
Revises: 3b2dd0114f3e
Create Date: 2026-10-31 12:00:00.000000

Hand-written (the coding agent cannot run alembic). It must match
`ChunkEmbedding` in models.py exactly; `python -m alembic check` verifies that.
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5c8e1a7d4f2b"
down_revision: str | Sequence[str] | None = "3b2dd0114f3e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # The vector type comes from the pgvector extension; it must exist before the table.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=200), nullable=False),
        # 384 is written out, not imported from models.py: a migration must keep
        # describing the schema as it was, even if the constant changes later.
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=384), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.id"],
            name=op.f("fk_chunk_embeddings_chunk_id_chunks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("chunk_id", "model_name", name=op.f("pk_chunk_embeddings")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("chunk_embeddings")
    op.execute("DROP EXTENSION IF EXISTS vector")
