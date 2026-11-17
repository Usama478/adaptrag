"""Turn parsed blocks into text pieces with pages; shared by the block-based strategies.

Pure code: no database and no embedding. A strategy adds anything of its own (such as a
heading prefix) and assigns the chunk indexes.

Changing the rules here changes the chunks of every strategy that uses this module, so
bump "version" in the params of each of them. The table token budget (Stage 4) is
different: it is part of a strategy's params whenever it is on, so the cache key changes
by itself. It is on by default (DEFAULT_TABLE_TOKEN_BUDGET, note 0025); None turns it off.
"""

import re
from collections.abc import Callable, Sequence

from adaptrag.chunking.splitter import split_with_offsets, word_count
from adaptrag.chunking.text_map import join_blocks, pages_for
from adaptrag.parsing.blocks import BlockType, ParsedBlock

# The default token budget of a table chunk for the strategies that chunk tables (note 0025:
# text past 512 tokens is never embedded by bge-small). Defined here once, imported elsewhere.
DEFAULT_TABLE_TOKEN_BUDGET = 512

# Try a line break first, so an oversized table is cut between rows where possible.
TABLE_SEPARATORS = ("\n", " ")

# The row under a Markdown table header, e.g. "| --- | :-: |".
TABLE_DIVIDER = re.compile(r"\|(?:\s*:?-+:?\s*\|)+")

# A chunk before it gets its index: text, page_start, page_end.
Piece = tuple[str, int, int]

# Counts the embedding model's tokens in a text (Embedder.count_tokens in the app).
TokenCounter = Callable[[str], int]


def chunk_prose(blocks: Sequence[ParsedBlock], size_words: int, overlap_words: int) -> list[Piece]:
    """Split consecutive non-table blocks as one text, finding each piece's pages."""
    mapped = join_blocks(blocks)
    pieces: list[Piece] = []
    for start, piece in split_with_offsets(mapped.text, size_words, overlap_words):
        page_start, page_end = pages_for(mapped, start, start + len(piece))
        pieces.append((piece, page_start, page_end))
    return pieces


def _header_row_count(rows: Sequence[str]) -> int:
    """The header is the first row, plus the divider row under it when there is one."""
    if len(rows) >= 2 and TABLE_DIVIDER.fullmatch(rows[1].strip()):
        return 2
    return 1


def _split_row(row: str, header: Sequence[str], fits: Callable[[str], bool]) -> list[str]:
    """Cut one row that is too big for a chunk into runs of words, each under the header.

    A single word that does not fit is still kept (it cannot be cut), so that no text
    is lost.
    """

    def with_header(words: Sequence[str]) -> str:
        return "\n".join([*header, " ".join(words)])

    pieces: list[str] = []
    part: list[str] = []
    for word in row.split():
        if part and not fits(with_header([*part, word])):
            pieces.append(with_header(part))
            part = []
        part.append(word)
    if part:
        pieces.append(with_header(part))
    return pieces


def split_table_by_budget(
    text: str, size_words: int, token_budget: int, count_tokens: TokenCounter
) -> list[str]:
    """Cut a table into pieces of whole rows that fit both the word and the token budget.

    The header (first row and its divider) starts every piece, so each piece can be read
    on its own. A header that would use more than half of either budget is not repeated.
    A row too big for one piece is cut between words. Rows get no overlap.
    """

    def fits(candidate: str) -> bool:
        return word_count(candidate) <= size_words and count_tokens(candidate) <= token_budget

    rows = text.split("\n")
    header = rows[: _header_row_count(rows)]
    header_text = "\n".join(header)
    if (
        len(header) == len(rows)
        or word_count(header_text) * 2 > size_words
        or count_tokens(header_text) * 2 > token_budget
    ):
        header = []
    body = rows[len(header) :]

    pieces: list[str] = []
    current = list(header)
    for row in body:
        if fits("\n".join([*current, row])):
            current.append(row)
            continue
        if len(current) > len(header):
            # The piece is full: close it and try the row again under a fresh header.
            pieces.append("\n".join(current))
            current = list(header)
            if fits("\n".join([*current, row])):
                current.append(row)
                continue
        # Here `current` holds only the header, and the row is too big even so.
        pieces.extend(_split_row(row, header, fits))
    if len(current) > len(header):
        pieces.append("\n".join(current))
    return pieces


def chunk_table(
    table: ParsedBlock,
    size_words: int,
    token_budget: int | None = None,
    count_tokens: TokenCounter | None = None,
) -> list[Piece]:
    """One piece for a table within budget; otherwise pieces of whole rows, with no overlap.

    Without a token budget (the default) only words are counted and the header is not
    repeated. With one, every piece also stays within `token_budget` model tokens and
    starts with the header row (see `split_table_by_budget`).
    """
    if token_budget is None:
        if word_count(table.text) <= size_words:
            return [(table.text, table.page, table.page)]
        return [
            (piece, table.page, table.page)
            for _, piece in split_with_offsets(
                table.text, size_words, 0, separators=TABLE_SEPARATORS
            )
        ]

    if token_budget < 1:
        raise ValueError(f"token_budget must be 1 or more, got {token_budget}")
    if count_tokens is None:
        raise ValueError("a token budget needs a token counter")
    if word_count(table.text) <= size_words and count_tokens(table.text) <= token_budget:
        return [(table.text, table.page, table.page)]
    return [
        (piece, table.page, table.page)
        for piece in split_table_by_budget(table.text, size_words, token_budget, count_tokens)
    ]


def chunk_blocks(
    blocks: Sequence[ParsedBlock],
    size_words: int,
    overlap_words: int,
    table_token_budget: int | None = None,
    count_tokens: TokenCounter | None = None,
) -> list[Piece]:
    """Chunk blocks in document order: runs of prose together, each table on its own."""
    pieces: list[Piece] = []
    prose: list[ParsedBlock] = []
    for block in blocks:
        if block.block_type is BlockType.TABLE:
            # A table ends the current run of prose, so the two are never joined.
            pieces.extend(chunk_prose(prose, size_words, overlap_words))
            prose = []
            pieces.extend(chunk_table(block, size_words, table_token_budget, count_tokens))
        else:
            prose.append(block)
    pieces.extend(chunk_prose(prose, size_words, overlap_words))
    return pieces
