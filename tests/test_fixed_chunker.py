"""Tests for the fixed-size chunker (pure logic, no database)."""

from itertools import pairwise

import pytest

from adaptrag.chunking.fixed import chunk_fixed_size
from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument


def words(start: int, stop: int) -> str:
    """Numbered words like "w0 w1 w2", so positions are easy to check."""
    return " ".join(f"w{i}" for i in range(start, stop))


def make_doc(*page_texts: str) -> ParsedDocument:
    """A document with one paragraph block per page."""
    blocks = tuple(
        ParsedBlock(
            text=text,
            page=number,
            block_type=BlockType.PARAGRAPH,
            heading_path=(),
        )
        for number, text in enumerate(page_texts, start=1)
    )
    return ParsedDocument(page_count=len(page_texts), parser_version="test", blocks=blocks)


def test_short_document_gives_one_chunk() -> None:
    chunks = chunk_fixed_size(make_doc(words(0, 5)), size_words=10, overlap_words=2)

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == words(0, 5)
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 1)


def test_windows_move_by_size_minus_overlap() -> None:
    chunks = chunk_fixed_size(make_doc(words(0, 10)), size_words=4, overlap_words=2)

    assert [c.text for c in chunks] == [words(0, 4), words(2, 6), words(4, 8), words(6, 10)]
    assert [c.index for c in chunks] == [0, 1, 2, 3]


def test_last_chunk_can_be_shorter() -> None:
    chunks = chunk_fixed_size(make_doc(words(0, 9)), size_words=4, overlap_words=1)

    assert [c.text for c in chunks] == [words(0, 4), words(3, 7), words(6, 9)]


def test_every_word_is_kept_and_neighbours_overlap() -> None:
    chunks = chunk_fixed_size(make_doc(words(0, 23)), size_words=5, overlap_words=2)

    seen = {word for c in chunks for word in c.text.split()}
    assert seen == set(words(0, 23).split())
    for before, after in pairwise(chunks):
        assert before.text.split()[-2:] == after.text.split()[:2]


def test_chunk_can_cross_a_page_boundary() -> None:
    doc = make_doc(words(0, 3), words(3, 6))
    chunks = chunk_fixed_size(doc, size_words=4, overlap_words=0)

    assert chunks[0].text == words(0, 4)
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 2), (2, 2)]


def test_default_settings_are_200_words_with_40_overlap() -> None:
    chunks = chunk_fixed_size(make_doc(words(0, 250)))

    assert len(chunks) == 2
    assert chunks[0].text == words(0, 200)
    assert chunks[1].text == words(160, 250)


@pytest.mark.parametrize(
    ("size_words", "overlap_words"),
    [(0, 0), (4, -1), (4, 4), (4, 5)],
    ids=["size-zero", "overlap-negative", "overlap-equals-size", "overlap-bigger-than-size"],
)
def test_rejects_bad_settings(size_words: int, overlap_words: int) -> None:
    with pytest.raises(ValueError):
        chunk_fixed_size(
            make_doc(words(0, 10)),
            size_words=size_words,
            overlap_words=overlap_words,
        )
