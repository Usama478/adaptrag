"""Tests for the page map (pure logic, no database)."""

import pytest

from adaptrag.chunking.text_map import MappedText, PageSpan, join_blocks, pages_for
from adaptrag.parsing.blocks import BlockType, ParsedBlock


def block(text: str, page: int) -> ParsedBlock:
    return ParsedBlock(text=text, page=page, block_type=BlockType.PARAGRAPH)


def two_blocks() -> MappedText:
    """Text "first\\n\\nsecond": block 1 is [0, 5), separator [5, 7), block 2 is [7, 13)."""
    return join_blocks([block("first", 1), block("second", 2)])


def test_join_gives_exact_text_and_span_offsets() -> None:
    mapped = two_blocks()

    assert mapped.text == "first" + "\n\n" + "second"
    assert mapped.spans == (
        PageSpan(start=0, end=5, page=1),
        PageSpan(start=7, end=13, page=2),
    )


def test_custom_separator_offsets_use_its_length() -> None:
    mapped = join_blocks([block("first", 1), block("second", 2)], separator="\n")

    assert mapped.text == "first\nsecond"
    assert mapped.spans == (
        PageSpan(start=0, end=5, page=1),
        PageSpan(start=6, end=12, page=2),
    )


@pytest.mark.parametrize(("start", "end", "page"), [(1, 4, 1), (8, 12, 2)])
def test_range_inside_one_block_gives_its_page(start: int, end: int, page: int) -> None:
    assert pages_for(two_blocks(), start, end) == (page, page)


def test_range_over_parts_of_both_blocks_gives_both_pages() -> None:
    assert pages_for(two_blocks(), 3, 10) == (1, 2)


def test_end_is_exclusive() -> None:
    # The range stops right before the first character of "second" (offset 7).
    assert pages_for(two_blocks(), 0, 7) == (1, 1)


@pytest.mark.parametrize(("start", "end"), [(5, 7), (5, 6), (6, 7)])
def test_range_touching_only_the_separator_gives_earlier_page(start: int, end: int) -> None:
    assert pages_for(two_blocks(), start, end) == (1, 1)


def test_separator_range_uses_the_nearest_span_before_it() -> None:
    mapped = join_blocks([block("aaa", 2), block("bbb", 5), block("ccc", 9)])

    # "aaa\n\nbbb\n\nccc": offsets 8 and 9 are the separator after "bbb".
    assert pages_for(mapped, 8, 10) == (5, 5)


def test_range_before_every_span_uses_the_first_span() -> None:
    mapped = MappedText(text="  abc", spans=(PageSpan(start=2, end=5, page=4),))

    assert pages_for(mapped, 0, 2) == (4, 4)


def test_range_over_three_blocks_gives_smallest_and_largest_page() -> None:
    mapped = join_blocks([block("one", 1), block("two", 1), block("six", 3)])

    assert pages_for(mapped, 0, len(mapped.text)) == (1, 3)


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (-1, 3, "start must be 0 or more"),
        (0, 14, "must not be past the text length"),
        (4, 4, "must be before end"),
        (5, 3, "must be before end"),
    ],
)
def test_invalid_range_raises(start: int, end: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pages_for(two_blocks(), start, end)


def test_no_spans_raises() -> None:
    # The range is valid for this text, so only the missing spans can cause the error.
    mapped = MappedText(text="abc", spans=())

    with pytest.raises(ValueError, match="no spans"):
        pages_for(mapped, 0, 1)
