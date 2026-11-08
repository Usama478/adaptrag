"""The Strategy base class: how one family of documents is chunked and searched.

An abstract base class rather than a Protocol, because strategies are our own family and
share code: retrieval is dense for every strategy until Stage 5, so `retrieve` lives here.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from sqlalchemy.orm import Session

from adaptrag.chunking.chunks import TextChunk
from adaptrag.embedding.base import Embedder
from adaptrag.parsing.blocks import ParsedDocument
from adaptrag.retrieval.dense import retrieve_dense
from adaptrag.retrieval.results import ScoredChunk


class Strategy(ABC):
    """One way to chunk a document and search its chunks.

    name: short, stable id, stored in chunk_sets.strategy. Renaming one orphans its
    stored chunk sets, so treat it as permanent.
    """

    name: ClassVar[str]

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        """The embedder given at construction; a strategy never builds its own."""
        return self._embedder

    @property
    @abstractmethod
    def params(self) -> dict[str, int | float | str | bool]:
        """Everything that changes the chunks; stored in chunk_sets.params.

        Part of the cache key, so it must be JSON-serialisable and identical across runs.
        """

    @abstractmethod
    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        """Cut the document into chunks. Pure: no database, no embedding, no randomness."""

    def retrieve(
        self, session: Session, chunk_set_id: int, query: str, k: int
    ) -> list[ScoredChunk]:
        """The k chunks of the set most similar to the query, best first (dense search)."""
        return retrieve_dense(session, chunk_set_id, self.embedder, query, k)
