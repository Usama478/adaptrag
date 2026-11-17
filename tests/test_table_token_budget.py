"""The table token budget: shared table chunking and the two strategies (pure unit tests).

Token counters here are fakes, so no model is loaded.
"""

import json

import pytest

from adaptrag.chunking.segments import DEFAULT_TABLE_TOKEN_BUDGET, chunk_table
from adaptrag.embedding.fake import FakeEmbedder
from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument
from adaptrag.strategies.recursive import RecursiveStrategy
from adaptrag.strategies.registry import build_strategy
from adaptrag.strategies.section_aware import SectionAwareStrategy

HEADER = "| name | score |"
DIVIDER = "| --- | --- |"


def one_token_per_word(text: str) -> int:
    return len(text.split())


def two_tokens_per_word(text: str) -> int:
    """Like a table of numbers: far more tokens than words."""
    return 2 * len(text.split())


def body_rows(count: int) -> list[str]:
    return [f"| r{i} | {i} |" for i in range(count)]  # 5 words each


def table_block(rows: list[str], page: int = 3) -> ParsedBlock:
    return ParsedBlock(text="\n".join(rows), page=page, block_type=BlockType.TABLE)


def standard_table(row_count: int = 30) -> ParsedBlock:
    return table_block([HEADER, DIVIDER, *body_rows(row_count)])


def make_doc(*blocks: ParsedBlock) -> ParsedDocument:
    page_count = max(block.page for block in blocks)
    return ParsedDocument(page_count=page_count, parser_version="test", blocks=blocks)


def strategy_table() -> ParsedBlock:
    """A header, a divider and 80 rows of 7 words each: far over any budget used here."""
    rows = [f"| r{i} a | r{i} b |" for i in range(80)]
    return table_block(["| first | second |", "| --- | --- |", *rows], page=1)


def test_without_a_budget_the_header_is_not_repeated() -> None:
    table = standard_table()  # 160 words

    pieces = chunk_table(table, 50)

    assert len(pieces) > 1
    lines = [line for text, _, _ in pieces for line in text.split("\n")]
    assert lines == table.text.split("\n")  # every row once, the header only at the top


def test_with_a_budget_every_piece_starts_with_the_header_rows() -> None:
    pieces = chunk_table(standard_table(), 200, token_budget=60, count_tokens=two_tokens_per_word)

    assert len(pieces) > 1
    for text, page_start, page_end in pieces:
        assert text.split("\n")[:2] == [HEADER, DIVIDER]
        assert (page_start, page_end) == (3, 3)


def test_with_a_budget_no_piece_is_over_it() -> None:
    pieces = chunk_table(standard_table(), 200, token_budget=60, count_tokens=two_tokens_per_word)

    # Header 20 tokens + 4 rows of 10 = 60: full pieces sit exactly on the budget.
    assert max(two_tokens_per_word(text) for text, _, _ in pieces) == 60


def test_with_a_budget_every_body_row_appears_once_in_order() -> None:
    pieces = chunk_table(standard_table(), 200, token_budget=60, count_tokens=two_tokens_per_word)

    rows = [line for text, _, _ in pieces for line in text.split("\n")[2:]]
    assert rows == body_rows(30)


def test_a_table_within_both_budgets_stays_one_piece() -> None:
    table = standard_table(row_count=2)

    pieces = chunk_table(table, 200, token_budget=1000, count_tokens=two_tokens_per_word)

    assert pieces == [(table.text, 3, 3)]


def test_a_table_within_the_word_budget_but_over_the_token_budget_is_split() -> None:
    table = standard_table(row_count=10)  # 60 words: fine for 200 words, but 120 tokens

    assert len(chunk_table(table, 200)) == 1
    pieces = chunk_table(table, 200, token_budget=60, count_tokens=two_tokens_per_word)
    assert len(pieces) == 3  # 4 + 4 + 2 rows


def test_the_word_budget_still_applies_with_a_token_budget() -> None:
    pieces = chunk_table(standard_table(), 30, token_budget=1000, count_tokens=one_token_per_word)

    assert len(pieces) > 1
    assert all(len(text.split()) <= 30 for text, _, _ in pieces)


def test_a_row_too_big_for_one_piece_is_cut_between_words() -> None:
    cells = [f"c{i}" for i in range(100)]
    big_row = "| " + " ".join(cells) + " |"
    table = table_block([HEADER, DIVIDER, big_row])

    pieces = chunk_table(table, 200, token_budget=40, count_tokens=one_token_per_word)

    assert len(pieces) > 1
    assert all(one_token_per_word(text) <= 40 for text, _, _ in pieces)
    # Under each repeated header: the row's words, all of them, in order.
    words = [word for text, _, _ in pieces for word in text.split("\n")[2].split()]
    assert words == big_row.split()


def test_a_header_over_half_the_budget_is_not_repeated() -> None:
    long_header = "| " + " ".join(f"h{i}" for i in range(28)) + " |"  # 30 words
    rows = [long_header, *body_rows(20)]

    pieces = chunk_table(table_block(rows), 200, token_budget=40, count_tokens=one_token_per_word)

    assert len(pieces) > 1
    assert not pieces[1][0].startswith(long_header)
    assert [line for text, _, _ in pieces for line in text.split("\n")] == rows
    assert all(one_token_per_word(text) <= 40 for text, _, _ in pieces)


def test_a_token_budget_needs_a_counter_and_a_positive_value() -> None:
    with pytest.raises(ValueError):
        chunk_table(standard_table(), 200, token_budget=60)
    with pytest.raises(ValueError):
        chunk_table(standard_table(), 200, token_budget=0, count_tokens=one_token_per_word)


def test_recursive_params_gain_the_budget_only_when_it_is_set() -> None:
    plain = RecursiveStrategy(FakeEmbedder(), table_token_budget=None).params
    budgeted = RecursiveStrategy(FakeEmbedder(), table_token_budget=512).params

    # Literal on purpose. The plain params must not change: they are the stored cache key.
    assert plain == {"size_words": 200, "overlap_words": 40, "version": 1}
    assert budgeted == {
        "size_words": 200,
        "overlap_words": 40,
        "version": 1,
        "table_token_budget": 512,
    }
    assert json.loads(json.dumps(budgeted)) == budgeted


def test_section_aware_params_gain_the_budget_only_when_it_is_set() -> None:
    plain = SectionAwareStrategy(FakeEmbedder(), table_token_budget=None).params
    budgeted = SectionAwareStrategy(FakeEmbedder(), table_token_budget=512).params

    assert plain == {"size_words": 200, "overlap_words": 40, "version": 1}
    assert budgeted == {
        "size_words": 200,
        "overlap_words": 40,
        "version": 1,
        "table_token_budget": 512,
    }


def test_registry_builds_recursive_with_the_default_budget_of_512() -> None:
    params = build_strategy("recursive", FakeEmbedder()).params

    assert params["table_token_budget"] == 512  # literal on purpose
    assert params["table_token_budget"] == DEFAULT_TABLE_TOKEN_BUDGET


def test_registry_builds_section_aware_with_the_default_budget_of_512() -> None:
    params = build_strategy("section_aware", FakeEmbedder()).params

    assert params["table_token_budget"] == 512
    assert params["table_token_budget"] == DEFAULT_TABLE_TOKEN_BUDGET


def test_fixed_size_params_have_no_table_budget() -> None:
    assert "table_token_budget" not in build_strategy("fixed_size", FakeEmbedder()).params


def test_none_leaves_the_budget_out_of_the_params_of_both_strategies() -> None:
    assert "table_token_budget" not in RecursiveStrategy(FakeEmbedder(), None).params
    assert "table_token_budget" not in SectionAwareStrategy(FakeEmbedder(), None).params


def test_the_default_budget_repeats_the_header_without_naming_a_budget() -> None:
    # A default-built strategy splits a long table like the explicit 512 one, header on top.
    embedder = FakeEmbedder()
    doc = make_doc(strategy_table())  # 560 words of rows: over the 200-word budget

    default = RecursiveStrategy(embedder).chunk(doc)
    explicit = RecursiveStrategy(embedder, table_token_budget=512).chunk(doc)

    assert [c.text for c in default] == [c.text for c in explicit]
    assert len(default) > 1
    start = "| first | second |\n| --- | --- |\n"
    assert all(chunk.text.startswith(start) for chunk in default)


def test_recursive_with_a_budget_keeps_table_chunks_within_it_and_prose_unchanged() -> None:
    embedder = FakeEmbedder()  # counts words + 2 as tokens
    prose = ParsedBlock(
        text=" ".join(f"w{i}" for i in range(300)), page=1, block_type=BlockType.PARAGRAPH
    )
    doc = make_doc(prose, strategy_table())

    plain = RecursiveStrategy(embedder, table_token_budget=None).chunk(doc)
    budgeted = RecursiveStrategy(embedder, table_token_budget=50).chunk(doc)

    table_chunks = [chunk for chunk in budgeted if chunk.text.startswith("|")]
    assert len(table_chunks) > 1
    assert all(embedder.count_tokens(chunk.text) <= 50 for chunk in table_chunks)
    start = "| first | second |\n| --- | --- |\n"
    assert all(chunk.text.startswith(start) for chunk in table_chunks)
    assert max(embedder.count_tokens(c.text) for c in plain if c.text.startswith("|")) > 50
    plain_prose = [chunk.text for chunk in plain if not chunk.text.startswith("|")]
    assert [chunk.text for chunk in budgeted if not chunk.text.startswith("|")] == plain_prose


def test_section_aware_budget_counts_the_heading_prefix() -> None:
    embedder = FakeEmbedder()
    heading = ParsedBlock(
        text="4 Results",
        page=1,
        block_type=BlockType.HEADING,
        heading_path=("4 Results",),
        level=1,
    )
    doc = make_doc(heading, strategy_table())

    chunks = SectionAwareStrategy(embedder, table_token_budget=50).chunk(doc)

    assert len(chunks) > 1
    start = "4 Results\n\n| first | second |\n| --- | --- |\n"
    assert all(chunk.text.startswith(start) for chunk in chunks)
    # The whole chunk, prefix included, is what the model embeds. 2 prefix words + 10
    # header words + 5 rows of 7 + 2 = 49 tokens; a sixth row would make 56.
    assert max(embedder.count_tokens(chunk.text) for chunk in chunks) == 49


def test_a_strategy_rejects_a_budget_below_one() -> None:
    with pytest.raises(ValueError):
        RecursiveStrategy(FakeEmbedder(), table_token_budget=0)
    with pytest.raises(ValueError):
        SectionAwareStrategy(FakeEmbedder(), table_token_budget=-5)
