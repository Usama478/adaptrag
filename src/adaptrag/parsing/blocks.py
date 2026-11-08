"""Data shapes produced by the PDF parser.

The parser turns a PDF into a ParsedDocument. Everything after it (profiler,
router, chunkers) reads only these shapes, never the PDF library's own objects,
so the PDF library can be swapped without touching the rest of the app.
"""

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise


class BlockType(StrEnum):
    """What kind of content a block holds."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    """One piece of a document, in reading order.

    page: 1-based physical page in the PDF (what a PDF viewer jumps to).
    heading_path: headings this block sits under, outermost first, e.g.
        ("3 Method", "3.2 Training"). A HEADING block's path ends with itself.
        Blocks before the first heading have an empty path.
    level: heading depth from 1 to 6 for a HEADING block, None for every other block.
    """

    text: str
    page: int
    block_type: BlockType
    heading_path: tuple[str, ...] = ()
    level: int | None = None

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError(f"page must be 1 or more, got {self.page}")
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if self.block_type is BlockType.HEADING:
            if self.level is None or not 1 <= self.level <= 6:
                raise ValueError(f"a heading needs a level from 1 to 6, got {self.level}")
        elif self.level is not None:
            raise ValueError(f"only headings have a level, got {self.level}")


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """A whole parsed PDF: its blocks in reading order, plus document facts.

    fallback_pages: pages built from plain text because the Markdown lost too much.
    """

    page_count: int
    parser_version: str
    blocks: tuple[ParsedBlock, ...]
    fallback_pages: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.page_count < 1:
            raise ValueError(f"page_count must be 1 or more, got {self.page_count}")
        if any(first >= second for first, second in pairwise(self.fallback_pages)):
            raise ValueError(f"fallback_pages must be strictly increasing: {self.fallback_pages}")
        for page in self.fallback_pages:
            if not 1 <= page <= self.page_count:
                raise ValueError(f"fallback page {page} is outside 1-{self.page_count}")
        for block in self.blocks:
            if block.page > self.page_count:
                raise ValueError(
                    f"block on page {block.page} but document has {self.page_count} pages"
                )
