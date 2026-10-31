"""Tests for the TextChunk shape (pure logic, no database)."""

import pytest

from adaptrag.chunking.chunks import TextChunk


def test_valid_chunk_is_created() -> None:
    chunk = TextChunk(index=0, text="some text", page_start=3, page_end=4)

    assert chunk.page_start == 3
    assert chunk.page_end == 4


def test_index_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="index"):
        TextChunk(index=-1, text="some text", page_start=1, page_end=1)


def test_text_must_not_be_blank() -> None:
    with pytest.raises(ValueError, match="blank"):
        TextChunk(index=0, text="   ", page_start=1, page_end=1)


def test_page_start_must_be_one_or_more() -> None:
    with pytest.raises(ValueError, match="page_start"):
        TextChunk(index=0, text="some text", page_start=0, page_end=1)


def test_page_end_must_not_be_before_page_start() -> None:
    with pytest.raises(ValueError, match="page_end"):
        TextChunk(index=0, text="some text", page_start=4, page_end=3)
