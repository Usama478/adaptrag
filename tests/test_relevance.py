"""The relevance rule and the section-aware prefix helpers (pure unit tests)."""

from adaptrag.embedding.fake import FakeEmbedder
from adaptrag.evaluation.relevance import (
    MISSING,
    PARTIAL,
    SPLIT,
    WHOLE,
    evidence_status,
    first_relevant_rank,
    is_relevant,
    match_kind,
)
from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument
from adaptrag.strategies.section_aware import (
    SectionAwareStrategy,
    heading_prefixes,
    strip_heading_prefix,
)

EIGHT_WORDS = "alpha beta gamma delta epsilon zeta eta theta"


def heading(*path: str) -> ParsedBlock:
    return ParsedBlock(
        text=path[-1], page=1, block_type=BlockType.HEADING, heading_path=path, level=len(path)
    )


def paragraph(text: str) -> ParsedBlock:
    return ParsedBlock(text=text, page=1, block_type=BlockType.PARAGRAPH)


def make_doc(*blocks: ParsedBlock) -> ParsedDocument:
    return ParsedDocument(page_count=1, parser_version="test", blocks=blocks)


def test_whole_evidence_matches_despite_case_and_line_breaks() -> None:
    chunk = "The  Encoder has\nsix layers in total"

    assert match_kind(chunk, "encoder has six layers") == WHOLE


def test_a_run_of_half_the_evidence_words_is_a_partial_match() -> None:
    # The chunk ends in the middle of the evidence: it holds 4 of the 8 words, in a row.
    chunk = "some text before alpha beta gamma delta"

    assert match_kind(chunk, EIGHT_WORDS) == PARTIAL


def test_a_run_shorter_than_half_is_no_match() -> None:
    chunk = "some text before alpha beta gamma"  # 3 of 8

    assert match_kind(chunk, EIGHT_WORDS) is None


def test_odd_length_evidence_needs_the_bigger_half() -> None:
    evidence = "alpha beta gamma delta epsilon"  # 5 words: 3 needed

    assert match_kind("x alpha beta y", evidence) is None
    assert match_kind("x gamma delta epsilon y", evidence) == PARTIAL


def test_a_partial_run_is_compared_word_by_word() -> None:
    # "alpha beta" is inside "xalpha beta" as characters, but not as two whole words.
    assert match_kind("xalpha beta", "alpha beta gamma delta") is None


def test_evidence_words_scattered_over_the_chunk_do_not_count() -> None:
    chunk = "alpha x beta x gamma x delta x epsilon x zeta x eta x theta"

    assert match_kind(chunk, EIGHT_WORDS) is None


def test_blank_evidence_matches_nothing() -> None:
    assert match_kind("any chunk text", "   ") is None


def test_a_chunk_is_relevant_if_any_evidence_entry_matches() -> None:
    chunk = "the model was trained for ten epochs"

    assert is_relevant(chunk, ["not in the chunk at all", "trained for ten epochs"])
    assert not is_relevant(chunk, ["not in the chunk at all", "nor is this one"])
    assert not is_relevant(chunk, [])


def test_first_relevant_rank_counts_from_one() -> None:
    evidence = ["trained for ten epochs"]
    chunks = ["unrelated text", "it was trained for ten epochs", "trained for ten epochs again"]

    assert first_relevant_rank(chunks, evidence) == 2
    assert first_relevant_rank(["unrelated text", "more of it"], evidence) is None


def test_evidence_status_tells_whole_split_and_missing_apart() -> None:
    first_half = "intro words alpha beta gamma delta"
    second_half = "epsilon zeta eta theta and more"

    assert evidence_status([first_half, f"all of it: {EIGHT_WORDS}"], [EIGHT_WORDS]) == WHOLE
    assert evidence_status([first_half, second_half], [EIGHT_WORDS]) == SPLIT
    assert evidence_status(["nothing useful", "here either"], [EIGHT_WORDS]) == MISSING


def test_prefix_words_do_not_make_a_chunk_relevant() -> None:
    chunk = "Training setup details\n\nAdam is used"
    evidence = "training setup details adam"

    # With the prefix the chunk would hold the whole evidence; its own text holds 1 word of 4.
    assert match_kind(chunk, evidence) == WHOLE
    body = strip_heading_prefix(chunk, {"Training setup details"})
    assert body == "Adam is used"
    assert match_kind(body, evidence) is None


def test_text_that_starts_with_no_known_prefix_is_left_alone() -> None:
    two_paragraphs = "First paragraph\n\nSecond paragraph"

    assert strip_heading_prefix(two_paragraphs, {"Intro"}) == two_paragraphs
    assert strip_heading_prefix(two_paragraphs, set()) == two_paragraphs
    assert strip_heading_prefix("Intro", {"Intro"}) == "Intro"  # no body after it


def test_heading_prefixes_lists_each_section_path_and_no_empty_one() -> None:
    doc = make_doc(
        paragraph("text before any heading"),
        heading("3 Method"),
        heading("3 Method", "3.2 Training"),
        paragraph("body"),
    )

    assert heading_prefixes(doc) == {"3 Method", "3 Method > 3.2 Training"}


def test_stripping_gives_back_the_text_under_each_heading() -> None:
    doc = make_doc(
        paragraph("before the first heading"),
        heading("Intro"),
        paragraph("intro body"),
        heading("Method"),
        paragraph("method body"),
    )
    chunks = SectionAwareStrategy(FakeEmbedder()).chunk(doc)
    prefixes = heading_prefixes(doc)

    assert [chunk.text for chunk in chunks][1] == "Intro\n\nintro body"
    assert [strip_heading_prefix(chunk.text, prefixes) for chunk in chunks] == [
        "before the first heading",
        "intro body",
        "method body",
    ]
