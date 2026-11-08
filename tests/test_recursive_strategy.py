"""RecursiveStrategy: cuts at paragraph boundaries and keeps tables apart (pure tests)."""

import json

from adaptrag.chunking.chunks import TextChunk
from adaptrag.embedding.fake import FakeEmbedder
from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument
from adaptrag.strategies.fixed_size import FixedSizeStrategy
from adaptrag.strategies.recursive import RecursiveStrategy
from adaptrag.strategies.registry import build_strategy, strategy_names

SMALL_TABLE = "| Model | Score |\n| --- | --- |\n| A | 0.9 |"


def words(start: int, stop: int) -> str:
    """Numbered words like "w0 w1 w2", so every paragraph's text is different."""
    return " ".join(f"w{i}" for i in range(start, stop))


def paragraph(text: str, page: int = 1) -> ParsedBlock:
    return ParsedBlock(text=text, page=page, block_type=BlockType.PARAGRAPH)


def table(text: str, page: int = 1) -> ParsedBlock:
    return ParsedBlock(text=text, page=page, block_type=BlockType.TABLE)


def make_doc(*blocks: ParsedBlock) -> ParsedDocument:
    page_count = max((block.page for block in blocks), default=1)
    return ParsedDocument(page_count=page_count, parser_version="test", blocks=blocks)


def chunk(doc: ParsedDocument) -> list[TextChunk]:
    return RecursiveStrategy(FakeEmbedder()).chunk(doc)


def big_table_rows() -> list[str]:
    """A header, a divider and 80 rows of 7 words each: 560 words of rows alone."""
    rows = [f"| r{i} a | r{i} b |" for i in range(80)]
    return ["| first | second |", "| --- | --- |", *rows]


def mixed_doc() -> ParsedDocument:
    """A heading, prose over budget, a small table, an oversized table and closing prose."""
    return make_doc(
        ParsedBlock(
            text="Introduction",
            page=1,
            block_type=BlockType.HEADING,
            heading_path=("Introduction",),
            level=1,
        ),
        paragraph(words(0, 500), page=1),
        table(SMALL_TABLE, page=2),
        table("\n".join(big_table_rows()), page=3),
        paragraph(words(500, 560), page=3),
    )


def test_registered_under_recursive() -> None:
    strategy = build_strategy("recursive", FakeEmbedder())

    assert type(strategy) is RecursiveStrategy
    assert "recursive" in strategy_names()


def test_params_equal_the_agreed_settings_and_survive_json() -> None:
    params = RecursiveStrategy(FakeEmbedder()).params

    # Literal on purpose: comparing with recursive.py could never catch a change there.
    assert params == {"size_words": 200, "overlap_words": 40, "version": 1}
    assert json.loads(json.dumps(params)) == params


def test_two_long_paragraphs_give_one_chunk_each() -> None:
    first, second = words(0, 120), words(120, 240)

    chunks = chunk(make_doc(paragraph(first), paragraph(second)))

    assert [c.text for c in chunks] == [first, second]


def test_short_paragraphs_on_two_pages_share_a_chunk_with_both_pages() -> None:
    doc = make_doc(paragraph(words(0, 60), page=1), paragraph(words(60, 120), page=2))

    chunks = chunk(doc)

    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_small_table_is_its_own_chunk_with_its_page() -> None:
    doc = make_doc(
        paragraph(words(0, 20), page=1),
        table(SMALL_TABLE, page=2),
        paragraph(words(20, 40), page=2),
    )

    chunks = chunk(doc)

    assert len(chunks) == 3
    assert chunks[1].text == SMALL_TABLE
    assert (chunks[1].page_start, chunks[1].page_end) == (2, 2)
    assert "|" not in chunks[0].text
    assert "|" not in chunks[2].text


def test_oversized_table_is_split_into_whole_rows() -> None:
    rows = big_table_rows()
    table_text = "\n".join(rows)
    assert len(table_text.split()) > 200  # otherwise this would prove nothing

    chunks = chunk(make_doc(table(table_text, page=4)))

    assert len(chunks) > 1
    for c in chunks:
        assert (c.page_start, c.page_end) == (4, 4)
        for line in c.text.split("\n"):
            assert line.startswith("|")
            assert line.endswith("|")
    # Every row appears, in order, and exactly once: tables get no overlap.
    assert [line for c in chunks for line in c.text.split("\n")] == rows


def test_no_text_is_lost() -> None:
    doc = mixed_doc()

    chunks = chunk(doc)

    assert len(chunks) > 1
    block_words = {word for block in doc.blocks for word in block.text.split()}
    chunk_words = {word for c in chunks for word in c.text.split()}
    assert block_words <= chunk_words


def test_indexes_count_from_zero_in_document_order() -> None:
    chunks = chunk(mixed_doc())

    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Document order, not just numbered: the pages never go backwards.
    assert [c.page_start for c in chunks] == sorted(c.page_start for c in chunks)


def test_document_with_no_blocks_gives_no_chunks() -> None:
    assert chunk(make_doc()) == []


def test_chunks_differ_from_fixed_size_on_the_same_document() -> None:
    doc = make_doc(*(paragraph(words(i, i + 150)) for i in range(0, 600, 150)))

    recursive = [c.text for c in chunk(doc)]
    fixed = [c.text for c in FixedSizeStrategy(FakeEmbedder()).chunk(doc)]

    assert recursive != fixed
