"""add document routing columns

Revision ID: 8f3a6c1e9d47
Revises: 5c8e1a7d4f2b
Create Date: 2026-11-09 12:00:00.000000

Hand-written (the coding agent cannot run alembic). It must match the routing columns
of `Document` in models.py exactly; `python -m alembic check` verifies that.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "8f3a6c1e9d47"
down_revision: str | Sequence[str] | None = "5c8e1a7d4f2b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable: documents ingested before routing have no decision.
    op.add_column(
        "documents",
        sa.Column("profile", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column("documents", sa.Column("strategy", sa.String(length=50), nullable=True))
    op.add_column("documents", sa.Column("route_rule", sa.String(length=50), nullable=True))
    op.add_column("documents", sa.Column("route_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("documents", "route_reason")
    op.drop_column("documents", "route_rule")
    op.drop_column("documents", "strategy")
    op.drop_column("documents", "profile")
