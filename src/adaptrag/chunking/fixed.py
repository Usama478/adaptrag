"""Fixed-size chunking: the control strategy.

Splits a document's text into windows of N words, each overlapping the previous one by
M words. It ignores headings, paragraphs and page breaks on purpose: it shows what
retrieval looks like when the chunker knows nothing about the document's structure.
"""

from adaptrag.chunking.chunks import TextChunk
from adaptrag.parsing.blocks import ParsedDocument

STRATEGY_NAME = "fixed_size"
DEFAULT_SIZE_WORDS = 200
DEFAULT_OVERLAP_WORDS = 40


def chunk_fixed_size(
    doc: ParsedDocument,
    size_words: int = DEFAULT_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[TextChunk]:
    """Cut the document into overlapping windows of words, keeping page ranges."""
    if size_words < 1:
        raise ValueError(f"size_words must be 1 or more, got {size_words}")
    if not 0 <= overlap_words < size_words:
        raise ValueError(
            f"overlap_words must be 0 or more and smaller than size_words, got {overlap_words}"
        )

    words: list[str] = []
    pages: list[int] = []
    for block in doc.blocks:
        for word in block.text.split():
            words.append(word)
            pages.append(block.page)

    chunks: list[TextChunk] = []
    step = size_words - overlap_words
    start = 0
    while start < len(words):
        end = min(start + size_words, len(words))
        chunks.append(
            TextChunk(
                index=len(chunks),
                text=" ".join(words[start:end]),
                page_start=pages[start],
                page_end=pages[end - 1],
            )
        )
        if end == len(words):
            break
        start += step
    return chunks
