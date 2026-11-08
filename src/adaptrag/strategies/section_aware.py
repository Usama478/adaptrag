"""The section-aware strategy: chunks stay inside a section and start with its heading path.

The prefix gives each chunk the context its own words may lack (a paragraph that never
says "training" can still sit under "3.2 Training"). It counts towards the size budget,
so section-aware chunks are no bigger than those of the other strategies.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from adaptrag.chunking.chunks import TextChunk
from adaptrag.chunking.fixed import DEFAULT_OVERLAP_WORDS, DEFAULT_SIZE_WORDS
from adaptrag.chunking.segments import chunk_blocks
from adaptrag.chunking.splitter import word_count
from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument
from adaptrag.strategies.base import Strategy

# Bump "version" whenever the prefix or section rules below, or the shared rules in
# chunking/segments.py, change: they are code, not params, but they change the chunks.
PARAMS: dict[str, int] = {
    "size_words": DEFAULT_SIZE_WORDS,
    "overlap_words": DEFAULT_OVERLAP_WORDS,
    "version": 1,
}

# A very long heading is usually a parsing mistake; it must not use up the chunk's budget.
MAX_PREFIX_WORDS = 40

PATH_SEPARATOR = " > "


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


class SectionAwareStrategy(Strategy):
    """Recursive splitting inside each section, with the heading path at the top of a chunk."""

    name = "section_aware"

    @property
    def params(self) -> dict[str, int | float | str | bool]:
        # A copy, so a caller changing the result cannot change the cache key.
        return dict(PARAMS)

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for section in _sections(doc.blocks):
            prefix = _prefix(section.path)
            budget = PARAMS["size_words"] - word_count(prefix)
            for piece, page_start, page_end in chunk_blocks(
                section.body, budget, PARAMS["overlap_words"]
            ):
                chunks.append(
                    TextChunk(
                        index=len(chunks),
                        text=f"{prefix}\n\n{piece}" if prefix else piece,
                        # Pages come from the piece: the prefix has no page of its own.
                        page_start=page_start,
                        page_end=page_end,
                    )
                )
        return chunks
