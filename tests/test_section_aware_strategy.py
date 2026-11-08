"""SectionAwareStrategy: chunks stay in one section and carry its heading path (pure tests)."""

import json

from adaptrag.chunking.chunks import TextChunk
from adaptrag.embedding.fake import FakeEmbedder
from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument
from adaptrag.strategies.recursive import RecursiveStrategy
from adaptrag.strategies.registry import build_strategy, strategy_names
from adaptrag.strategies.section_aware import SectionAwareStrategy

SMALL_TABLE = "| Model | Score |\n| --- | --- |\n| A | 0.9 |"


def words(start: int, stop: int) -> str:
    """Numbered words like "w0 w1 w2", so every block's text is different."""
    return " ".join(f"w{i}" for i in range(start, stop))


def heading(*path: str, page: int = 1) -> ParsedBlock:
    """A heading whose path ends with itself, as the parser makes it."""
    return ParsedBlock(
        text=path[-1], page=page, block_type=BlockType.HEADING, heading_path=path, level=len(path)
    )


def paragraph(text: str, page: int = 1) -> ParsedBlock:
    return ParsedBlock(text=text, page=page, block_type=BlockType.PARAGRAPH)


def table(text: str, page: int = 1) -> ParsedBlock:
    return ParsedBlock(text=text, page=page, block_type=BlockType.TABLE)


def make_doc(*blocks: ParsedBlock) -> ParsedDocument:
    page_count = max((block.page for block in blocks), default=1)
    return ParsedDocument(page_count=page_count, parser_version="test", blocks=blocks)


def chunk(doc: ParsedDocument) -> list[TextChunk]:
    return SectionAwareStrategy(FakeEmbedder()).chunk(doc)


def big_table_rows() -> list[str]:
    """A header, a divider and 80 rows of 7 words each: 560 words of rows alone."""
    rows = [f"| r{i} a | r{i} b |" for i in range(80)]
    return ["| first | second |", "| --- | --- |", *rows]


def test_registered_under_section_aware() -> None:
    strategy = build_strategy("section_aware", FakeEmbedder())

    assert type(strategy) is SectionAwareStrategy
    assert "section_aware" in strategy_names()


def test_params_equal_the_agreed_settings_and_survive_json() -> None:
    params = SectionAwareStrategy(FakeEmbedder()).params

    # Literal on purpose: comparing with section_aware.py could never catch a change there.
    assert params == {"size_words": 200, "overlap_words": 40, "version": 1}
    assert json.loads(json.dumps(params)) == params


def test_two_short_sections_stay_apart_where_recursive_joins_them() -> None:
    doc = make_doc(
        heading("Intro"),
        paragraph(words(0, 60)),
        heading("Method"),
        paragraph(words(60, 120)),
    )

    recursive = RecursiveStrategy(FakeEmbedder()).chunk(doc)
    section_aware = chunk(doc)

    assert len(recursive) == 1
    assert [c.text for c in section_aware] == [
        f"Intro\n\n{words(0, 60)}",
        f"Method\n\n{words(60, 120)}",
    ]


def test_nested_heading_gives_the_full_path_and_an_empty_parent_gives_no_chunk() -> None:
    doc = make_doc(
        heading("3 Method"),
        heading("3 Method", "3.2 Training"),
        paragraph(words(0, 30)),
    )

    chunks = chunk(doc)

    assert [c.text for c in chunks] == [f"3 Method > 3.2 Training\n\n{words(0, 30)}"]


def test_text_before_the_first_heading_has_no_prefix() -> None:
    doc = make_doc(paragraph(words(0, 30)), heading("Intro"), paragraph(words(30, 60)))

    chunks = chunk(doc)

    assert [c.text for c in chunks] == [words(0, 30), f"Intro\n\n{words(30, 60)}"]


def test_every_chunk_with_its_prefix_has_at_most_200_words() -> None:
    long_heading = words(5000, 5030)  # 30 words: a budget that ignored it would overflow
    doc = make_doc(
        heading(long_heading),
        paragraph(words(0, 1000)),
        table("\n".join(big_table_rows())),
    )

    chunks = chunk(doc)

    assert len(chunks) > 5
    assert all(len(c.text.split()) <= 200 for c in chunks)


def test_small_table_is_its_own_chunk_after_the_prefix() -> None:
    doc = make_doc(
        heading("4 Results"),
        paragraph(words(0, 20)),
        table(SMALL_TABLE),
        paragraph(words(20, 40)),
    )

    chunks = chunk(doc)

    assert [c.text for c in chunks] == [
        f"4 Results\n\n{words(0, 20)}",
        f"4 Results\n\n{SMALL_TABLE}",
        f"4 Results\n\n{words(20, 40)}",
    ]


def test_oversized_table_gives_the_prefix_then_whole_rows_only() -> None:
    rows = big_table_rows()
    prefix = "4 Results\n\n"

    chunks = chunk(make_doc(heading("4 Results"), table("\n".join(rows))))

    assert len(chunks) > 1
    assert all(c.text.startswith(prefix) for c in chunks)
    bodies = [c.text.removeprefix(prefix).split("\n") for c in chunks]
    # Every row appears, in order, and exactly once: tables get no overlap.
    assert [line for body in bodies for line in body] == rows


def test_pages_come_from_the_section_blocks() -> None:
    doc = make_doc(
        heading("Intro", page=2),
        paragraph(words(0, 30), page=2),
        paragraph(words(30, 60), page=3),
    )

    chunks = chunk(doc)

    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (2, 3)


def test_the_page_of_the_heading_does_not_count() -> None:
    doc = make_doc(heading("Intro", page=1), paragraph(words(0, 30), page=2))

    chunks = chunk(doc)

    assert (chunks[0].page_start, chunks[0].page_end) == (2, 2)


def test_no_text_is_lost() -> None:
    doc = make_doc(
        paragraph(words(0, 40), page=1),
        heading("Alpha", page=1),
        paragraph(words(40, 600), page=1),
        table(SMALL_TABLE, page=2),
        table("\n".join(big_table_rows()), page=2),
        heading("Beta", page=3),
        heading("Beta", "Gamma", page=3),
        paragraph(words(600, 700), page=3),
    )

    chunks = chunk(doc)

    body_words = {
        word
        for block in doc.blocks
        if block.block_type is not BlockType.HEADING
        for word in block.text.split()
    }
    chunk_words = {word for c in chunks for word in c.text.split()}
    assert body_words <= chunk_words
    assert {"Alpha", "Beta", "Gamma"} <= chunk_words


def test_a_prefix_over_40_words_is_cut_to_the_innermost_heading() -> None:
    outer, inner = words(100, 130), words(200, 250)
    doc = make_doc(heading(outer), heading(outer, inner), paragraph(words(0, 30)))

    chunks = chunk(doc)

    assert [c.text for c in chunks] == [f"{words(200, 240)}\n\n{words(0, 30)}"]


def test_a_short_innermost_heading_is_kept_whole_when_the_path_is_too_long() -> None:
    outer, inner = words(100, 130), words(200, 220)  # 30 + ">" + 20 = 51 words
    doc = make_doc(heading(outer), heading(outer, inner), paragraph(words(0, 30)))

    chunks = chunk(doc)

    assert [c.text for c in chunks] == [f"{inner}\n\n{words(0, 30)}"]


def test_a_prefix_of_exactly_40_words_is_kept() -> None:
    outer, inner = words(100, 119), words(200, 220)  # 19 + ">" + 20 = 40 words
    doc = make_doc(heading(outer), heading(outer, inner), paragraph(words(0, 30)))

    chunks = chunk(doc)

    assert [c.text for c in chunks] == [f"{outer} > {inner}\n\n{words(0, 30)}"]


def test_indexes_count_from_zero_across_sections() -> None:
    doc = make_doc(
        heading("Alpha"),
        paragraph(words(0, 500)),
        heading("Beta"),
        paragraph(words(500, 900)),
    )

    chunks = chunk(doc)

    assert len(chunks) > 2
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Document order, not just numbered: every Alpha chunk comes before every Beta chunk.
    sections = [c.text.split("\n\n")[0] for c in chunks]
    assert sections == sorted(sections)


def test_document_with_no_blocks_gives_no_chunks() -> None:
    assert chunk(make_doc()) == []


def test_chunks_differ_from_recursive_on_the_same_document() -> None:
    doc = make_doc(
        heading("Alpha"),
        paragraph(words(0, 150)),
        heading("Beta"),
        paragraph(words(150, 300)),
    )

    section_aware = [c.text for c in chunk(doc)]
    recursive = [c.text for c in RecursiveStrategy(FakeEmbedder()).chunk(doc)]

    assert section_aware != recursive
