"""Map character ranges of a joined document string back to PDF pages.

Library splitters work on one long string and report where each piece starts.
Citations need pages, so this module remembers which characters came from which page.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from adaptrag.parsing.blocks import ParsedBlock


@dataclass(frozen=True, slots=True)
class PageSpan:
    """The characters of one block: text[start:end] (end exclusive), found on page."""

    start: int
    end: int
    page: int


@dataclass(frozen=True, slots=True)
class MappedText:
    """A joined document string and one PageSpan per block, in order."""

    text: str
    spans: tuple[PageSpan, ...]


def join_blocks(blocks: Sequence[ParsedBlock], separator: str = "\n\n") -> MappedText:
    """Join block texts with the separator; each span covers only its block's characters."""
    spans: list[PageSpan] = []
    offset = 0
    for block in blocks:
        end = offset + len(block.text)
        spans.append(PageSpan(start=offset, end=end, page=block.page))
        offset = end + len(separator)
    text = separator.join(block.text for block in blocks)
    return MappedText(text=text, spans=tuple(spans))


def pages_for(mapped: MappedText, start: int, end: int) -> tuple[int, int]:
    """Return (smallest page, largest page) for the character range [start, end).

    A range holding only separator characters takes the page of the nearest span
    before it, or the first span if there is none before.
    """
    if start < 0:
        raise ValueError(f"start must be 0 or more, got {start}")
    if end > len(mapped.text):
        raise ValueError(f"end ({end}) must not be past the text length ({len(mapped.text)})")
    if start >= end:
        raise ValueError(f"start ({start}) must be before end ({end})")
    if not mapped.spans:
        raise ValueError("mapped text has no spans")

    overlapping = [span.page for span in mapped.spans if span.start < end and start < span.end]
    if overlapping:
        return min(overlapping), max(overlapping)

    spans_before = [span for span in mapped.spans if span.end <= start]
    page = spans_before[-1].page if spans_before else mapped.spans[0].page
    return page, page
