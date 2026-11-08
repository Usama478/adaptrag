"""The fixed-size strategy: the control every other strategy is compared with."""

from adaptrag.chunking.chunks import TextChunk
from adaptrag.chunking.fixed import DEFAULT_OVERLAP_WORDS, DEFAULT_SIZE_WORDS, chunk_fixed_size
from adaptrag.parsing.blocks import ParsedDocument
from adaptrag.strategies.base import Strategy

# One dict feeds both the stored params and the chunker, so they can never disagree.
PARAMS: dict[str, int] = {
    "size_words": DEFAULT_SIZE_WORDS,
    "overlap_words": DEFAULT_OVERLAP_WORDS,
}


class FixedSizeStrategy(Strategy):
    """Windows of N words overlapping by M words, ignoring the document's structure."""

    name = "fixed_size"

    @property
    def params(self) -> dict[str, int | float | str | bool]:
        # A copy, so a caller changing the result cannot change the cache key.
        return dict(PARAMS)

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        return chunk_fixed_size(doc, **PARAMS)
