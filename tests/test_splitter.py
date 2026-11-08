"""Tests for the recursive splitter wrapper (pure logic, no database)."""

from itertools import pairwise

import pytest

from adaptrag.chunking.splitter import split_with_offsets, word_count


def words(count: int, prefix: str = "w") -> str:
    """Return distinct space-separated words, e.g. "w0 w1 w2"."""
    return " ".join(f"{prefix}{index}" for index in range(count))


def sentences(count: int, words_each: int, tag: str) -> str:
    """Return sentences of distinct words, joined by ". " and ending with a full stop."""
    return ". ".join(words(words_each, f"{tag}s{index}w") for index in range(count)) + "."


PARAGRAPHS_AND_SENTENCES = "\n\n".join(sentences(6, 12, f"p{index}") for index in range(4))
RUN_OF_WORDS = words(500)
NO_SEPARATORS = "abcdefghij" * 100


def test_word_count_splits_on_any_whitespace() -> None:
    assert word_count("  two\n\nwords\t") == 2


@pytest.mark.parametrize("text", [PARAGRAPHS_AND_SENTENCES, RUN_OF_WORDS, NO_SEPARATORS])
def test_every_piece_matches_the_text_at_its_offset(text: str) -> None:
    pairs = split_with_offsets(text, size_words=30, overlap_words=5)

    assert len(pairs) > 1
    for start, piece in pairs:
        assert text[start : start + len(piece)] == piece


def test_two_paragraphs_become_exactly_two_pieces() -> None:
    first = words(120, "a")
    second = words(120, "b")
    text = f"{first}\n\n{second}"

    pairs = split_with_offsets(text, size_words=200, overlap_words=40)

    assert pairs == [(0, first), (len(first) + 2, second)]


def test_long_run_gives_bounded_overlapping_pieces() -> None:
    pairs = split_with_offsets(RUN_OF_WORDS, size_words=200, overlap_words=40)

    assert len(pairs) > 1
    assert all(word_count(piece) <= 200 for _, piece in pairs)
    for (start, piece), (next_start, _) in pairwise(pairs):
        assert start < next_start < start + len(piece)


@pytest.mark.parametrize("text", [PARAGRAPHS_AND_SENTENCES, RUN_OF_WORDS])
def test_no_word_is_lost(text: str) -> None:
    pairs = split_with_offsets(text, size_words=30, overlap_words=5)

    assert {word for _, piece in pairs for word in piece.split()} == set(text.split())


def test_sentences_keep_their_full_stop() -> None:
    text = sentences(3, 6, "t")

    pairs = split_with_offsets(text, size_words=10, overlap_words=0)

    assert len(pairs) == 3
    assert all(piece.endswith(".") for _, piece in pairs)


def test_repetitive_text_gives_strictly_increasing_offsets() -> None:
    text = "The model was trained on the same data again. " * 100

    pairs = split_with_offsets(text, size_words=50, overlap_words=10)

    assert len(pairs) > 1
    for start, piece in pairs:
        assert text[start : start + len(piece)] == piece
    assert all(first[0] < second[0] for first, second in pairwise(pairs))


def test_given_separators_are_used() -> None:
    pairs = split_with_offsets(
        "one two|three four", size_words=3, overlap_words=0, separators=("|",)
    )

    assert pairs == [(0, "one two|"), (8, "three four")]


@pytest.mark.parametrize("text", ["", "   \n\n\t "])
def test_blank_text_gives_no_pieces(text: str) -> None:
    assert split_with_offsets(text, size_words=10, overlap_words=2) == []


@pytest.mark.parametrize(
    ("size_words", "overlap_words", "message"),
    [
        (0, 0, "size_words"),
        (-1, 0, "size_words"),
        (10, -1, "overlap_words"),
        (10, 10, "overlap_words"),
        (10, 11, "overlap_words"),
    ],
)
def test_bad_size_or_overlap_raises(size_words: int, overlap_words: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        split_with_offsets("some text", size_words=size_words, overlap_words=overlap_words)
