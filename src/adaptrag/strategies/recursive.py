"""The recursive strategy: cut at paragraph and sentence boundaries, keep tables apart.

Uses the same size budget as fixed-size on purpose: the benchmark should measure where
the cuts go, not how big the chunks are.
"""

from collections.abc import Sequence

from adaptrag.chunking.chunks import TextChunk
from adaptrag.chunking.fixed import DEFAULT_OVERLAP_WORDS, DEFAULT_SIZE_WORDS
from adaptrag.chunking.splitter import split_with_offsets, word_count
from adaptrag.chunking.text_map import join_blocks, pages_for
from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument
from adaptrag.strategies.base import Strategy

# One dict feeds both the stored params and the splitter, so they can never disagree.
# Bump "version" whenever the separators or the table handling change: they are code, not
# params, but they change the chunks, so they must still change the cache key.
PARAMS: dict[str, int] = {
    "size_words": DEFAULT_SIZE_WORDS,
    "overlap_words": DEFAULT_OVERLAP_WORDS,
    "version": 1,
}

# Try a line break first, so an oversized table is cut between rows where possible.
TABLE_SEPARATORS = ("\n", " ")

# A chunk before it gets its index: text, page_start, page_end.
Piece = tuple[str, int, int]


def _prose_pieces(blocks: Sequence[ParsedBlock]) -> list[Piece]:
    """Split consecutive headings and paragraphs as one text, finding each piece's pages."""
    mapped = join_blocks(blocks)
    pieces: list[Piece] = []
    for start, piece in split_with_offsets(
        mapped.text, PARAMS["size_words"], PARAMS["overlap_words"]
    ):
        page_start, page_end = pages_for(mapped, start, start + len(piece))
        pieces.append((piece, page_start, page_end))
    return pieces


def _table_pieces(table: ParsedBlock) -> list[Piece]:
    """One piece for a table within budget; otherwise pieces of whole rows, with no overlap."""
    size_words = PARAMS["size_words"]
    if word_count(table.text) <= size_words:
        return [(table.text, table.page, table.page)]
    return [
        (piece, table.page, table.page)
        for _, piece in split_with_offsets(table.text, size_words, 0, separators=TABLE_SEPARATORS)
    ]


class RecursiveStrategy(Strategy):
    """Recursive splitting of the prose between tables; each table is chunked on its own."""

    name = "recursive"

    @property
    def params(self) -> dict[str, int | float | str | bool]:
        # A copy, so a caller changing the result cannot change the cache key.
        return dict(PARAMS)

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        pieces: list[Piece] = []
        segment: list[ParsedBlock] = []
        for block in doc.blocks:
            if block.block_type is BlockType.TABLE:
                # A table ends the current run of prose, so the two are never joined.
                pieces.extend(_prose_pieces(segment))
                segment = []
                pieces.extend(_table_pieces(block))
            else:
                segment.append(block)
        pieces.extend(_prose_pieces(segment))

        return [
            TextChunk(index=index, text=text, page_start=page_start, page_end=page_end)
            for index, (text, page_start, page_end) in enumerate(pieces)
        ]
