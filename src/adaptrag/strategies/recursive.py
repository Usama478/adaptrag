"""The recursive strategy: cut at paragraph and sentence boundaries, keep tables apart.

Uses the same size budget as fixed-size on purpose: the benchmark should measure where
the cuts go, not how big the chunks are.
"""

from adaptrag.chunking.chunks import TextChunk
from adaptrag.chunking.fixed import DEFAULT_OVERLAP_WORDS, DEFAULT_SIZE_WORDS
from adaptrag.chunking.segments import chunk_blocks
from adaptrag.parsing.blocks import ParsedDocument
from adaptrag.strategies.base import Strategy

# One dict feeds both the stored params and the splitter, so they can never disagree.
# Bump "version" whenever the separators or the table handling (chunking/segments.py)
# change: they are code, not params, but they change the chunks, so they must still change
# the cache key.
PARAMS: dict[str, int] = {
    "size_words": DEFAULT_SIZE_WORDS,
    "overlap_words": DEFAULT_OVERLAP_WORDS,
    "version": 1,
}


class RecursiveStrategy(Strategy):
    """Recursive splitting of the prose between tables; each table is chunked on its own."""

    name = "recursive"

    @property
    def params(self) -> dict[str, int | float | str | bool]:
        # A copy, so a caller changing the result cannot change the cache key.
        return dict(PARAMS)

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        pieces = chunk_blocks(doc.blocks, PARAMS["size_words"], PARAMS["overlap_words"])
        return [
            TextChunk(index=index, text=text, page_start=page_start, page_end=page_end)
            for index, (text, page_start, page_end) in enumerate(pieces)
        ]
