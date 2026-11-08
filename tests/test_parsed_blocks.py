"""Unit tests for the parser data shapes. No database needed."""

from dataclasses import FrozenInstanceError

import pytest

from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument


def make_block(page: int = 1, text: str = "Some text.") -> ParsedBlock:
    return ParsedBlock(text=text, page=page, block_type=BlockType.PARAGRAPH)


def test_valid_block_keeps_its_fields() -> None:
    block = ParsedBlock(
        text="We train for 10 epochs.",
        page=3,
        block_type=BlockType.PARAGRAPH,
        heading_path=("3 Method", "3.2 Training"),
    )
    assert block.text == "We train for 10 epochs."
    assert block.page == 3
    assert block.block_type is BlockType.PARAGRAPH
    assert block.heading_path == ("3 Method", "3.2 Training")


def test_heading_path_defaults_to_empty() -> None:
    assert make_block().heading_path == ()


def test_level_defaults_to_none() -> None:
    assert make_block().level is None


@pytest.mark.parametrize("level", [1, 6])
def test_heading_keeps_a_level_from_one_to_six(level: int) -> None:
    block = ParsedBlock(text="Intro", page=1, block_type=BlockType.HEADING, level=level)
    assert block.level == level


@pytest.mark.parametrize("bad_level", [None, 0, 7])
def test_heading_needs_a_level_from_one_to_six(bad_level: int | None) -> None:
    with pytest.raises(ValueError, match="a heading needs a level"):
        ParsedBlock(text="Intro", page=1, block_type=BlockType.HEADING, level=bad_level)


@pytest.mark.parametrize("block_type", [BlockType.PARAGRAPH, BlockType.TABLE])
def test_only_headings_have_a_level(block_type: BlockType) -> None:
    with pytest.raises(ValueError, match="only headings have a level"):
        ParsedBlock(text="Body", page=1, block_type=block_type, level=1)


def test_block_is_immutable() -> None:
    block = make_block()
    with pytest.raises(FrozenInstanceError):
        block.page = 2


@pytest.mark.parametrize("bad_page", [0, -1])
def test_page_must_be_one_or_more(bad_page: int) -> None:
    with pytest.raises(ValueError, match="page must be 1 or more"):
        make_block(page=bad_page)


@pytest.mark.parametrize("blank_text", ["", "   \n\t"])
def test_text_must_not_be_blank(blank_text: str) -> None:
    with pytest.raises(ValueError, match="text must not be empty"):
        make_block(text=blank_text)


def test_block_type_round_trips_through_its_string() -> None:
    assert BlockType("table") is BlockType.TABLE
    assert BlockType.TABLE == "table"


def test_unknown_block_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        BlockType("figure")


def test_valid_document_keeps_blocks_in_order() -> None:
    blocks = (make_block(page=1, text="First."), make_block(page=2, text="Second."))
    doc = ParsedDocument(page_count=2, parser_version="0.1.0", blocks=blocks)
    assert [b.text for b in doc.blocks] == ["First.", "Second."]


def test_document_rejects_block_beyond_page_count() -> None:
    with pytest.raises(ValueError, match="page 5"):
        ParsedDocument(page_count=4, parser_version="0.1.0", blocks=(make_block(page=5),))


def test_fallback_pages_default_to_empty() -> None:
    doc = ParsedDocument(page_count=1, parser_version="0.2.0", blocks=())
    assert doc.fallback_pages == ()


def test_valid_fallback_pages_are_kept() -> None:
    doc = ParsedDocument(page_count=3, parser_version="0.2.0", blocks=(), fallback_pages=(1, 3))
    assert doc.fallback_pages == (1, 3)


@pytest.mark.parametrize("bad_pages", [(2, 2), (3, 2)], ids=["duplicate", "decreasing"])
def test_fallback_pages_must_be_strictly_increasing(bad_pages: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ParsedDocument(page_count=3, parser_version="0.2.0", blocks=(), fallback_pages=bad_pages)


@pytest.mark.parametrize("bad_page", [0, 4])
def test_fallback_pages_must_be_inside_the_document(bad_page: int) -> None:
    with pytest.raises(ValueError, match="outside 1-3"):
        ParsedDocument(page_count=3, parser_version="0.2.0", blocks=(), fallback_pages=(bad_page,))


def test_document_needs_at_least_one_page() -> None:
    with pytest.raises(ValueError, match="page_count"):
        ParsedDocument(page_count=0, parser_version="0.1.0", blocks=())
