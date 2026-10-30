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


def test_document_needs_at_least_one_page() -> None:
    with pytest.raises(ValueError, match="page_count"):
        ParsedDocument(page_count=0, parser_version="0.1.0", blocks=())
