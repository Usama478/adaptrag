"""In-memory shape of a chunk: a piece of document text plus where it came from."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One piece of a document, ready to be embedded.

    index: position of the chunk in the document, starting at 0.
    page_start, page_end: 1-based physical pages the text came from.
    A chunk that crosses a page boundary has page_end greater than page_start.
    """

    index: int
    text: str
    page_start: int
    page_end: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be 0 or more, got {self.index}")
        if not self.text.strip():
            raise ValueError("chunk text must not be blank")
        if self.page_start < 1:
            raise ValueError(f"page_start must be 1 or more, got {self.page_start}")
        if self.page_end < self.page_start:
            raise ValueError(
                f"page_end ({self.page_end}) must not be before page_start ({self.page_start})"
            )
