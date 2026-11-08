"""Document profile: exact counts, the section-number rule, rounding, empty documents."""

import json

import pytest

from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument
from adaptrag.routing.profile import DocumentProfile, profile_document


def heading(text: str, page: int = 1) -> ParsedBlock:
    return ParsedBlock(text=text, page=page, block_type=BlockType.HEADING, level=1)


def paragraph(text: str, page: int = 1) -> ParsedBlock:
    return ParsedBlock(text=text, page=page, block_type=BlockType.PARAGRAPH)


def table(text: str, page: int = 1) -> ParsedBlock:
    return ParsedBlock(text=text, page=page, block_type=BlockType.TABLE)


def document(
    *blocks: ParsedBlock, page_count: int = 1, fallback_pages: tuple[int, ...] = ()
) -> ParsedDocument:
    return ParsedDocument(
        page_count=page_count,
        parser_version="test",
        blocks=blocks,
        fallback_pages=fallback_pages,
    )


def small_document() -> ParsedDocument:
    """20 words: heading 2, paragraph 4, heading 1, table 4, paragraph 9."""
    return document(
        heading("1 Introduction", page=1),
        paragraph("alpha beta gamma delta", page=1),
        heading("Method", page=2),
        table("| Model | Score |\n| --- | --- |\n| A | 0.9 |", page=3),
        paragraph("one two three four five six seven eight nine", page=4),
        page_count=4,
        fallback_pages=(2,),
    )


def test_counts_are_exact() -> None:
    profile = profile_document(small_document())

    assert profile.page_count == 4
    assert profile.words == 20  # the "---" separator row holds no words
    assert profile.headings == 2
    assert profile.tables == 1
    assert profile.fallback_pages == 1


def test_shares_and_rates_of_a_small_document() -> None:
    profile = profile_document(small_document())

    assert profile.numbered_heading_share == 0.5
    assert profile.headings_per_1k_words == 100.0
    assert profile.table_word_share == 0.2
    assert profile.fallback_page_share == 0.25
    assert profile.words_per_page == 5.0


@pytest.mark.parametrize(
    "text",
    ["3 Method", "3.1 Training", "3.1.2 Details", "3. Method", "III. RESULTS", "A. Setup"],
)
def test_heading_with_a_section_number_counts(text: str) -> None:
    assert profile_document(document(heading(text))).numbered_heading_share == 1.0


@pytest.mark.parametrize(
    "text",
    ["Abstract", "What We Do", "3D models are hard", "A Survey of Models", "A.Setup"],
)
def test_heading_without_a_section_number_does_not_count(text: str) -> None:
    assert profile_document(document(heading(text))).numbered_heading_share == 0.0


def test_numbered_heading_share_is_rounded() -> None:
    doc = document(heading("1 Intro"), heading("2 Method"), heading("Abstract"))

    assert profile_document(doc).numbered_heading_share == 0.667


def test_rates_and_shares_are_rounded_to_three_decimals() -> None:
    # 3 words: 1 in the heading and 2 in the table.
    doc = document(heading("Method"), table("| a | b |"), page_count=7, fallback_pages=(2, 3))

    profile = profile_document(doc)

    assert profile.headings_per_1k_words == 333.333
    assert profile.table_word_share == 0.667
    assert profile.fallback_page_share == 0.286
    assert profile.words_per_page == 0.429


def test_document_with_no_blocks_gives_zeros() -> None:
    # ParsedDocument requires at least one page, so page_count stays 1.
    assert profile_document(document()) == DocumentProfile(
        page_count=1,
        words=0,
        headings=0,
        numbered_heading_share=0.0,
        headings_per_1k_words=0.0,
        tables=0,
        table_word_share=0.0,
        fallback_pages=0,
        fallback_page_share=0.0,
        words_per_page=0.0,
    )


def test_heading_with_no_words_does_not_divide_by_zero() -> None:
    profile = profile_document(document(heading("***")))

    assert profile.words == 0
    assert profile.headings == 1
    assert profile.headings_per_1k_words == 0.0
    assert profile.table_word_share == 0.0


def test_document_without_headings_has_a_zero_numbered_share() -> None:
    profile = profile_document(document(paragraph("just some words")))

    assert profile.headings == 0
    assert profile.numbered_heading_share == 0.0
    assert profile.words == 3


def test_to_dict_survives_a_json_round_trip() -> None:
    data = profile_document(small_document()).to_dict()

    assert json.loads(json.dumps(data)) == {
        "page_count": 4,
        "words": 20,
        "headings": 2,
        "numbered_heading_share": 0.5,
        "headings_per_1k_words": 100.0,
        "tables": 1,
        "table_word_share": 0.2,
        "fallback_pages": 1,
        "fallback_page_share": 0.25,
        "words_per_page": 5.0,
    }
