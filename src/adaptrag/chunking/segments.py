"""Turn parsed blocks into text pieces with pages; shared by the block-based strategies.

Pure code: no database and no embedding. A strategy adds anything of its own (such as a
heading prefix) and assigns the chunk indexes.

Changing the rules here changes the chunks of every strategy that uses this module, so
bump "version" in the params of each of them.
"""

from collections.abc import Sequence

from adaptrag.chunking.splitter import split_with_offsets, word_count
from adaptrag.chunking.text_map import join_blocks, pages_for
from adaptrag.parsing.blocks import BlockType, ParsedBlock

# Try a line break first, so an oversized table is cut between rows where possible.
TABLE_SEPARATORS = ("\n", " ")

# A chunk before it gets its index: text, page_start, page_end.
Piece = tuple[str, int, int]


def chunk_prose(blocks: Sequence[ParsedBlock], size_words: int, overlap_words: int) -> list[Piece]:
    """Split consecutive non-table blocks as one text, finding each piece's pages."""
    mapped = join_blocks(blocks)
    pieces: list[Piece] = []
    for start, piece in split_with_offsets(mapped.text, size_words, overlap_words):
        page_start, page_end = pages_for(mapped, start, start + len(piece))
        pieces.append((piece, page_start, page_end))
    return pieces


def chunk_table(table: ParsedBlock, size_words: int) -> list[Piece]:
    """One piece for a table within budget; otherwise pieces of whole rows, with no overlap."""
    if word_count(table.text) <= size_words:
        return [(table.text, table.page, table.page)]
    return [
        (piece, table.page, table.page)
        for _, piece in split_with_offsets(table.text, size_words, 0, separators=TABLE_SEPARATORS)
    ]


def chunk_blocks(blocks: Sequence[ParsedBlock], size_words: int, overlap_words: int) -> list[Piece]:
    """Chunk blocks in document order: runs of prose together, each table on its own."""
    pieces: list[Piece] = []
    prose: list[ParsedBlock] = []
    for block in blocks:
        if block.block_type is BlockType.TABLE:
            # A table ends the current run of prose, so the two are never joined.
            pieces.extend(chunk_prose(prose, size_words, overlap_words))
            prose = []
            pieces.extend(chunk_table(block, size_words))
        else:
            prose.append(block)
    pieces.extend(chunk_prose(prose, size_words, overlap_words))
    return pieces
