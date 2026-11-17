"""Passage selection for question generation (pure unit tests)."""

import random
from collections import Counter

from adaptrag.evaluation.passages import (
    FACTUAL,
    MULTI_SECTION,
    TABLE,
    Passage,
    far_apart_pairs,
    plan_drafts,
    prose_windows,
    spread_pick,
    table_draft_count,
    table_passages,
    table_slots,
)
from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument


def words(count: int, label: str) -> str:
    """`count` different words, e.g. "a0 a1 a2"."""
    return " ".join(f"{label}{i}" for i in range(count))


def paragraph(text: str, page: int = 1, path: tuple[str, ...] = ()) -> ParsedBlock:
    return ParsedBlock(text=text, page=page, block_type=BlockType.PARAGRAPH, heading_path=path)


def heading(text: str, page: int = 1) -> ParsedBlock:
    return ParsedBlock(
        text=text, page=page, block_type=BlockType.HEADING, heading_path=(text,), level=1
    )


def table(
    row_count: int, page: int = 1, path: tuple[str, ...] = (), digits: bool = True
) -> ParsedBlock:
    rows = ["| name | score |", "| --- | --- |"]
    rows.extend(f"| r{i} | {i} |" if digits else "| x | y |" for i in range(row_count))
    return ParsedBlock(
        text="\n".join(rows), page=page, block_type=BlockType.TABLE, heading_path=path
    )


def make_doc(*blocks: ParsedBlock) -> ParsedDocument:
    page_count = max(block.page for block in blocks)
    return ParsedDocument(page_count=page_count, parser_version="test", blocks=blocks)


def one_window_per_page(page_count: int) -> ParsedDocument:
    """Each page holds one paragraph of 160 words, which is one window."""
    pages = range(1, page_count + 1)
    return make_doc(*(paragraph(words(160, f"p{page}w"), page) for page in pages))


def test_consecutive_paragraphs_are_joined_until_the_target_size() -> None:
    first, second, third = words(80, "a"), words(80, "b"), words(80, "c")

    windows = prose_windows(make_doc(paragraph(first), paragraph(second), paragraph(third)))

    # 80 + 80 words reach the target of 150; the third paragraph starts a new window.
    assert [window.text for window in windows] == [f"{first}\n\n{second}", third]
    assert [window.first_block for window in windows] == [0, 2]


def test_a_window_never_crosses_a_page() -> None:
    first, second = words(100, "a"), words(100, "b")

    windows = prose_windows(make_doc(paragraph(first, page=1), paragraph(second, page=2)))

    assert [(window.text, window.page) for window in windows] == [(first, 1), (second, 2)]


def test_a_window_never_crosses_a_heading_or_contains_it() -> None:
    first, second = words(100, "a"), words(100, "b")
    doc = make_doc(paragraph(first), heading("Method"), paragraph(second, path=("Method",)))

    windows = prose_windows(doc)

    assert [window.text for window in windows] == [first, second]


def test_a_leftover_under_60_words_is_dropped() -> None:
    assert prose_windows(make_doc(paragraph(words(59, "a")))) == []
    assert len(prose_windows(make_doc(paragraph(words(60, "a"))))) == 1


def test_a_long_paragraph_is_cut_to_its_first_300_words() -> None:
    text = words(500, "a")

    windows = prose_windows(make_doc(paragraph(text)))

    assert len(windows) == 1
    assert len(windows[0].text.split()) == 300
    assert text.startswith(windows[0].text)  # an exact piece of the document


def test_the_reference_section_gives_no_passages() -> None:
    path = ("References",)
    body = paragraph(words(200, "a"), path=path)
    doc = make_doc(heading("References"), body, table(5, path=path))

    assert prose_windows(doc) == []
    assert table_passages(doc) == []


def test_a_table_passage_needs_rows_and_a_digit_and_takes_its_caption() -> None:
    good = table(4, page=2)
    caption = "Table 2: Scores of every model."
    doc = make_doc(
        table(0, page=1),  # only the header and the divider: too few rows
        table(4, page=1, digits=False),  # no digit anywhere
        good,
        paragraph(caption, page=2),
    )

    passages = table_passages(doc)

    assert passages == [
        Passage(text=f"{caption}\n\n{good.text}", page=2, kind="table", first_block=2)
    ]


def test_a_big_table_passage_is_cut_to_whole_rows() -> None:
    big = table(100)  # 510 words

    (passage,) = table_passages(make_doc(big))

    rows = passage.text.split("\n")
    assert rows == big.text.split("\n")[: len(rows)]  # whole rows from the top
    assert 250 < len(passage.text.split()) <= 300


def test_spread_pick_takes_one_item_from_each_stretch() -> None:
    picked = spread_pick(list(range(100)), 4, random.Random(0))

    assert len(picked) == 4
    for stretch, item in enumerate(picked):
        assert 25 * stretch <= item < 25 * (stretch + 1)


def test_spread_pick_returns_everything_when_there_are_too_few() -> None:
    assert spread_pick(["a", "b"], 5, random.Random(0)) == ["a", "b"]
    assert spread_pick(["a", "b"], 0, random.Random(0)) == []


def test_pairs_are_far_apart_and_use_no_passage_twice() -> None:
    doc = one_window_per_page(9)  # a third of 9 pages: at least 3 pages apart

    pairs = far_apart_pairs(prose_windows(doc), doc.page_count, 2, random.Random(1))

    assert len(pairs) == 2
    assert all(second.page - first.page >= 3 for first, second in pairs)
    assert len({passage.first_block for pair in pairs for passage in pair}) == 4


def test_no_pairs_when_the_document_is_too_short() -> None:
    doc = one_window_per_page(2)  # pages 1 and 2 are only 1 apart; 2 are needed

    assert far_apart_pairs(prose_windows(doc), doc.page_count, 2, random.Random(1)) == []


def test_the_plan_depends_only_on_the_document_and_the_seed() -> None:
    doc = one_window_per_page(90)

    assert plan_drafts(doc, "seed-a") == plan_drafts(doc, "seed-a")
    assert plan_drafts(doc, "seed-a") != plan_drafts(doc, "seed-b")


def test_the_plan_has_factual_table_and_multi_section_drafts() -> None:
    blocks = []
    for page in range(1, 21):
        blocks.extend([paragraph(words(160, f"p{page}w"), page), table(4, page)])

    slots = plan_drafts(make_doc(*blocks), "seed")

    assert Counter(slot.type for slot in slots) == {FACTUAL: 13, TABLE: 4, MULTI_SECTION: 3}
    assert all(len(slot.passages) == (2 if slot.type == MULTI_SECTION else 1) for slot in slots)
    assert all(slot.passages[0].kind == "table" for slot in slots if slot.type == TABLE)


def test_missing_table_drafts_become_factual_drafts() -> None:
    doc = one_window_per_page(30)
    with_one_table = make_doc(*doc.blocks, table(4, page=30))

    slots = plan_drafts(with_one_table, "seed")

    # One usable table gives min(4, 2 x 1) = 2 table drafts; factual drafts fill up to 20.
    assert Counter(slot.type for slot in slots) == {FACTUAL: 15, TABLE: 2, MULTI_SECTION: 3}


def test_a_document_gets_20_drafts() -> None:
    no_tables = plan_drafts(one_window_per_page(30), "seed")

    assert len(no_tables) == 20
    assert Counter(slot.type for slot in no_tables) == {FACTUAL: 17, MULTI_SECTION: 3}


def test_table_drafts_are_two_per_usable_table_up_to_four() -> None:
    assert [table_draft_count(tables) for tables in (0, 1, 2, 3, 4, 9)] == [0, 2, 4, 4, 4, 4]


def test_one_usable_table_is_drafted_twice_as_variant_0_and_1() -> None:
    (passage,) = table_passages(make_doc(table(4)))

    slots = table_slots([passage], random.Random(0))

    assert [(slot.passages, slot.variant) for slot in slots] == [
        ((passage,), 0),
        ((passage,), 1),
    ]


def test_three_tables_give_four_drafts_and_four_tables_give_four_different_ones() -> None:
    passages = [
        Passage(text=f"table {index}", page=index + 1, kind="table", first_block=index)
        for index in range(4)
    ]

    three = table_slots(passages[:3], random.Random(0))
    four = table_slots(passages, random.Random(0))

    assert [slot.variant for slot in three] == [0, 0, 0, 1]
    assert {slot.passages[0] for slot in three} == set(passages[:3])
    assert [slot.variant for slot in four] == [0, 0, 0, 0]
    assert len({slot.passages[0] for slot in four}) == 4
