"""The section-aware strategy: chunks stay inside a section and start with its heading path.

The prefix gives each chunk the context its own words may lack (a paragraph that never
says "training" can still sit under "3.2 Training"). It counts towards the size budget,
so section-aware chunks are no bigger than those of the other strategies.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from functools import partial

from adaptrag.chunking.chunks import TextChunk
from adaptrag.chunking.fixed import DEFAULT_OVERLAP_WORDS, DEFAULT_SIZE_WORDS
from adaptrag.chunking.segments import chunk_blocks
from adaptrag.chunking.splitter import word_count
from adaptrag.embedding.base import Embedder
from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument
from adaptrag.strategies.base import Strategy

# Bump "version" whenever the prefix or section rules below, or the shared rules in
# chunking/segments.py, change: they are code, not params, but they change the chunks.
PARAMS: dict[str, int] = {
    "size_words": DEFAULT_SIZE_WORDS,
    "overlap_words": DEFAULT_OVERLAP_WORDS,
    "version": 1,
}

TABLE_TOKEN_BUDGET_PARAM = "table_token_budget"

# A very long heading is usually a parsing mistake; it must not use up the chunk's budget.
MAX_PREFIX_WORDS = 40

PATH_SEPARATOR = " > "

# Between the prefix and the chunk's own text. A prefix never contains it: headings are
# single lines.
PREFIX_SEPARATOR = "\n\n"


@dataclass(slots=True)
class _Section:
    """The heading path of one section and its body blocks (never a heading)."""

    path: tuple[str, ...]
    body: list[ParsedBlock] = field(default_factory=list)


def _sections(blocks: Sequence[ParsedBlock]) -> list[_Section]:
    """Group blocks into sections; blocks before the first heading get an empty path."""
    sections = [_Section(path=())]
    for block in blocks:
        if block.block_type is BlockType.HEADING:
            sections.append(_Section(path=block.heading_path))
        else:
            sections[-1].body.append(block)
    return sections


def _prefix(path: tuple[str, ...]) -> str:
    """The joined heading path, or the innermost heading's first words if that is too long."""
    joined = PATH_SEPARATOR.join(path)
    if word_count(joined) <= MAX_PREFIX_WORDS:
        return joined
    return " ".join(path[-1].split()[:MAX_PREFIX_WORDS])


def heading_prefixes(doc: ParsedDocument) -> frozenset[str]:
    """Every prefix this strategy can put at the top of one of the document's chunks."""
    prefixes = (_prefix(section.path) for section in _sections(doc.blocks))
    return frozenset(prefix for prefix in prefixes if prefix)


def strip_heading_prefix(text: str, prefixes: Collection[str]) -> str:
    """The chunk text without its heading prefix; unchanged if it starts with none of them.

    The evaluation uses it so that prefix words neither help nor hurt a match (Stage 4).
    """
    head, separator, body = text.partition(PREFIX_SEPARATOR)
    if separator and head in prefixes:
        return body
    return text


class SectionAwareStrategy(Strategy):
    """Recursive splitting inside each section, with the heading path at the top of a chunk.

    table_token_budget: None (the default, and what the registry builds) keeps tables on
    the word budget only. A number also keeps every table chunk, prefix included, within
    that many model tokens and repeats the header row (Stage 4 experiment, note 0025).
    """

    name = "section_aware"

    def __init__(self, embedder: Embedder, table_token_budget: int | None = None) -> None:
        super().__init__(embedder)
        if table_token_budget is not None and table_token_budget < 1:
            raise ValueError(f"table_token_budget must be 1 or more, got {table_token_budget}")
        self._table_token_budget = table_token_budget

    @property
    def params(self) -> dict[str, int | float | str | bool]:
        # A copy, so a caller changing the result cannot change the cache key.
        params: dict[str, int | float | str | bool] = dict(PARAMS)
        # Left out when there is no budget, so chunk sets stored before Stage 4 still match.
        if self._table_token_budget is not None:
            params[TABLE_TOKEN_BUDGET_PARAM] = self._table_token_budget
        return params

    def _count_tokens_with_prefix(self, prefix: str, text: str) -> int:
        """Tokens of the chunk as it will be embedded: the prefix counts too."""
        full_text = f"{prefix}{PREFIX_SEPARATOR}{text}" if prefix else text
        return self.embedder.count_tokens(full_text)

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for section in _sections(doc.blocks):
            prefix = _prefix(section.path)
            budget = PARAMS["size_words"] - word_count(prefix)
            for piece, page_start, page_end in chunk_blocks(
                section.body,
                budget,
                PARAMS["overlap_words"],
                self._table_token_budget,
                partial(self._count_tokens_with_prefix, prefix),
            ):
                chunks.append(
                    TextChunk(
                        index=len(chunks),
                        text=f"{prefix}{PREFIX_SEPARATOR}{piece}" if prefix else piece,
                        # Pages come from the piece: the prefix has no page of its own.
                        page_start=page_start,
                        page_end=page_end,
                    )
                )
        return chunks
