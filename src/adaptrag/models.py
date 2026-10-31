from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, MetaData, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Parent class for all tables. SQLAlchemy and Alembic use it to find every table."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int]
    status: Mapped[str] = mapped_column(String(20), default="uploaded")
    page_count: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Job(Base):
    """One processing attempt for a document (its "ticket")."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(20), default="queued")
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ChunkSet(Base):
    """One way of cutting a document: a strategy, its settings and the parser version."""

    __tablename__ = "chunk_sets"
    __table_args__ = (UniqueConstraint("document_id", "strategy", "params", "parser_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    strategy: Mapped[str] = mapped_column(String(50))
    params: Mapped[dict[str, Any]] = mapped_column(JSONB)
    parser_version: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Chunk(Base):
    """One stored piece of text from a chunk set, with the pages it came from."""

    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("chunk_set_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    chunk_set_id: Mapped[int] = mapped_column(ForeignKey("chunk_sets.id", ondelete="CASCADE"))
    chunk_index: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int]
    page_end: Mapped[int]
