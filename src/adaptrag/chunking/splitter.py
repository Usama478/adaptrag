"""Recursive text splitting, returning each piece with its start offset.

The only module that imports langchain_text_splitters.

Offsets are found here because the library's add_start_index assumes that
length_function counts characters; our sizes are in words.
"""

from collections.abc import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def word_count(text: str) -> int:
    """Count words the same way as the fixed-size chunker, so strategies share one budget."""
    return len(text.split())


def split_with_offsets(
    text: str,
    size_words: int,
    overlap_words: int,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[tuple[int, str]]:
    """Split text into overlapping pieces and return (start offset, piece) pairs in order.

    keep_separator="end" leaves a sentence's full stop on that sentence. Pieces are
    stripped, so they hold no leading or trailing whitespace.
    """
    if size_words < 1:
        raise ValueError(f"size_words must be 1 or more, got {size_words}")
    if not 0 <= overlap_words < size_words:
        raise ValueError(
            f"overlap_words must be 0 or more and smaller than size_words, got {overlap_words}"
        )
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        separators=list(separators),
        keep_separator="end",
        chunk_size=size_words,
        chunk_overlap=overlap_words,
        length_function=word_count,
        strip_whitespace=True,
    )

    pairs: list[tuple[int, str]] = []
    search_from = 0
    for piece in splitter.split_text(text):
        start = text.find(piece, search_from)
        if start == -1:
            raise RuntimeError("a split piece was not found in the original text")
        # Page numbers are computed from these offsets, so never return a wrong one.
        if text[start : start + len(piece)] != piece:
            raise RuntimeError(f"offset {start} does not match its split piece")
        pairs.append((start, piece))
        search_from = start + 1
    return pairs
