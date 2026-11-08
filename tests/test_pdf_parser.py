"""Tests for the PDF parser. PDF tests build their own tiny PDFs; no sample files needed."""

import logging
from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm
import pytest

from adaptrag.parsing.blocks import BlockType, ParsedBlock
from adaptrag.parsing.pdf import (
    PARSER_VERSION,
    CorruptPDFError,
    EncryptedPDFError,
    ScannedPDFError,
    blocks_from_markdown,
    choose_page_text,
    clean_text,
    count_words,
    parse_pdf,
)


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a small PDF with one text string per page ("" means a blank page)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, page.rect.height / 2), text)
    doc.save(path)
    doc.close()
    return path


def make_paragraph_pdf(path: Path, markers: list[str]) -> Path:
    """Write a PDF with one body paragraph per marker word ("" means a blank page).

    A real paragraph, because the layout model drops a page's only tiny line as a
    header or footer, and parse_pdf removes those.
    """
    doc = pymupdf.open()
    for marker in markers:
        page = doc.new_page()
        if marker:
            text = (
                f"The {marker} experiment compares several training methods on a shared "
                "benchmark. Each method runs with the same settings so that the results can "
                "be compared fairly. We report the average score over repeated runs and "
                "discuss where the methods differ most in practice."
            )
            page.insert_textbox(pymupdf.Rect(72, 200, 540, 500), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def one_page(markdown: str) -> list[ParsedBlock]:
    return blocks_from_markdown([(1, markdown)])


def test_clean_text_joins_lines_and_extra_spaces() -> None:
    assert clean_text("permission to\nreproduce  the\n") == "permission to reproduce the"


def test_clean_text_of_only_whitespace_is_empty() -> None:
    assert clean_text(" \n\t ") == ""


def test_count_words_counts_the_cells_of_a_glued_row() -> None:
    assert count_words("|0.91|0.87|0.93|") == 3


@pytest.mark.parametrize("text", ["---", "|", "|---|---|", "| --- | --- |"])
def test_count_words_ignores_separators_and_lone_pipes(text: str) -> None:
    assert count_words(text) == 0


def test_count_words_counts_normal_prose() -> None:
    assert count_words("We train for 10 epochs.") == 5


# --- blocks_from_markdown: pure rules, no PDF ---


def test_heading_level_is_the_number_of_hashes() -> None:
    blocks = one_page("# One\n\n### Three")
    assert [(b.block_type, b.level, b.text) for b in blocks] == [
        (BlockType.HEADING, 1, "One"),
        (BlockType.HEADING, 3, "Three"),
    ]


def test_seven_hashes_is_not_a_heading() -> None:
    blocks = one_page("####### Too deep")
    assert [b.block_type for b in blocks] == [BlockType.PARAGRAPH]
    assert blocks[0].level is None


def test_heading_stack_runs_across_pages_and_a_new_h1_resets_it() -> None:
    blocks = blocks_from_markdown(
        [
            (1, "# A\n\n## B"),
            (2, "Under B.\n\n# C\n\nUnder C."),
        ]
    )
    assert [(b.page, b.text, b.heading_path) for b in blocks] == [
        (1, "A", ("A",)),
        (1, "B", ("A", "B")),
        (2, "Under B.", ("A", "B")),
        (2, "C", ("C",)),
        (2, "Under C.", ("C",)),
    ]


def test_heading_of_the_same_level_replaces_its_sibling() -> None:
    blocks = one_page("# A\n\n## B\n\n## C")
    assert blocks[-1].heading_path == ("A", "C")


def test_paragraph_under_a_heading_gets_the_path() -> None:
    blocks = one_page("# Method\n\n## Training\n\nWe train for 10 epochs.")
    assert blocks[-1].block_type is BlockType.PARAGRAPH
    assert blocks[-1].heading_path == ("Method", "Training")


def test_text_before_the_first_heading_has_an_empty_path() -> None:
    blocks = one_page("Preamble.\n\n# First")
    assert blocks[0].text == "Preamble."
    assert blocks[0].heading_path == ()


def test_table_lines_become_one_table_block_with_pipes_kept() -> None:
    table = "| **a** | b |\n|---|---|\n| 1 | 2 |"
    blocks = one_page(f"# Results\n\nBefore.\n{table}\nAfter.")
    assert [b.block_type for b in blocks] == [
        BlockType.HEADING,
        BlockType.PARAGRAPH,
        BlockType.TABLE,
        BlockType.PARAGRAPH,
    ]
    assert blocks[2].text == "| **a** | b |\n| --- | --- |\n| 1 | 2 |"
    assert blocks[2].heading_path == ("Results",)
    assert blocks[2].level is None


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("|0.91|0.87|", "| 0.91 | 0.87 |"),
        ("|---|---|", "| --- | --- |"),
        ("|  two   words |x|  ", "| two words | x |"),
    ],
)
def test_table_lines_are_normalised_to_spaced_rows(line: str, expected: str) -> None:
    blocks = one_page(line)
    assert [(b.block_type, b.text) for b in blocks] == [(BlockType.TABLE, expected)]


def test_blank_line_splits_paragraphs_and_lines_are_joined_with_a_space() -> None:
    blocks = one_page("one\ntwo\n\nthree")
    assert [b.text for b in blocks] == ["one two", "three"]


def test_heading_ends_a_paragraph_without_a_blank_line() -> None:
    blocks = one_page("Before.\n# Title\nAfter.")
    assert [b.text for b in blocks] == ["Before.", "Title", "After."]


def test_paragraph_does_not_continue_on_the_next_page() -> None:
    blocks = blocks_from_markdown([(1, "End of page"), (2, "start of next")])
    assert [(b.page, b.text) for b in blocks] == [(1, "End of page"), (2, "start of next")]


def test_picture_markers_are_dropped_and_the_picture_text_stays() -> None:
    markdown = (
        "Before.\n\n"
        "<!-- Start of picture text -->\n"
        "Encoder<br>Decoder<br><!-- End of picture text -->\n\n"
        "After."
    )
    assert [b.text for b in one_page(markdown)] == ["Before.", "Encoder Decoder", "After."]


def test_code_fence_lines_are_dropped() -> None:
    blocks = one_page("```\nprint(1)\n```")
    assert [b.text for b in blocks] == ["print(1)"]


def test_lines_empty_after_cleaning_are_dropped() -> None:
    assert one_page("   \n<sup></sup>\n# <u></u>") == []


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("**bold** word", "bold word"),
        ("*italic* word", "italic word"),
        ("_italic_ word", "italic word"),
        ("**_both_** word", "both word"),
        ("`code` word", "code word"),
        ("~~gone~~ word", "gone word"),
        ("[the text](https://example.com/a) word", "the text word"),
        ("word<sup>1</sup> <u>under</u> <mark>marked</mark>", "word1 under marked"),
    ],
)
def test_markup_is_removed_and_its_text_kept(markdown: str, expected: str) -> None:
    assert [b.text for b in one_page(markdown)] == [expected]


def test_markup_is_removed_from_headings_and_their_paths() -> None:
    blocks = one_page("## **3.1 Attention**")
    assert blocks[0].text == "3.1 Attention"
    assert blocks[0].heading_path == ("3.1 Attention",)


@pytest.mark.parametrize(
    "text",
    [
        "the snake_case_name field",
        "the __init__ method",
        "2 * 3 * 4",
        "2*3*4",
        "citation [12] here",
    ],
)
def test_text_that_only_looks_like_markup_is_kept(text: str) -> None:
    assert [b.text for b in one_page(text)] == [text]


# --- choose_page_text: pure rules, no PDF ---


def words(count: int) -> str:
    return " ".join(["word"] * count)


def test_page_below_the_ratio_is_replaced_and_listed() -> None:
    pages, replaced = choose_page_text([(1, words(10))], {1: [words(60), words(40)]})
    assert pages == [(1, f"{words(60)}\n\n{words(40)}")]
    assert replaced == (1,)


def test_page_just_below_the_ratio_is_replaced() -> None:
    _, replaced = choose_page_text([(1, words(69))], {1: [words(100)]})
    assert replaced == (1,)


@pytest.mark.parametrize("markdown_words", [70, 120])
def test_page_at_or_above_the_ratio_is_kept(markdown_words: int) -> None:
    markdown = words(markdown_words)
    pages, replaced = choose_page_text([(1, markdown)], {1: [words(100)]})
    assert pages == [(1, markdown)]
    assert replaced == ()


def test_page_of_glued_table_rows_with_the_same_cells_is_not_replaced() -> None:
    cells = [f"cell{n}" for n in range(60)]
    rows = ["|" + "|".join(cells[start : start + 6]) + "|" for start in range(0, 60, 6)]
    markdown = "\n".join(rows)
    pages, replaced = choose_page_text([(1, markdown)], {1: [" ".join(cells)]})
    assert pages == [(1, markdown)]
    assert replaced == ()


def test_page_with_too_few_plain_words_is_never_replaced() -> None:
    pages, replaced = choose_page_text([(1, "")], {1: [words(49)]})
    assert pages == [(1, "")]
    assert replaced == ()


def test_page_with_exactly_the_minimum_plain_words_can_be_replaced() -> None:
    _, replaced = choose_page_text([(1, "")], {1: [words(50)]})
    assert replaced == (1,)


def test_only_the_lossy_page_changes_and_the_order_stays() -> None:
    markdown_pages = [(1, words(60)), (2, "x"), (3, words(60))]
    plain_pages = {1: [words(60)], 2: [words(80)], 3: [words(60)]}
    pages, replaced = choose_page_text(markdown_pages, plain_pages)
    assert pages == [(1, words(60)), (2, words(80)), (3, words(60))]
    assert replaced == (2,)


@pytest.mark.parametrize("first", ["# not a heading", "| not | a table |"])
def test_plain_text_can_never_become_a_heading_or_table(first: str) -> None:
    pages, _ = choose_page_text([(1, "x")], {1: [f"{first} {words(60)}"]})
    blocks = blocks_from_markdown(pages)
    assert [(b.block_type, b.level) for b in blocks] == [(BlockType.PARAGRAPH, None)]
    assert blocks[0].text == f"{first} {words(60)}"


def test_replaced_page_keeps_the_heading_path_from_the_previous_page() -> None:
    pages, _ = choose_page_text([(1, "# Appendix"), (2, "x")], {1: ["Appendix"], 2: [words(60)]})
    blocks = blocks_from_markdown(pages)
    assert blocks[-1].page == 2
    assert blocks[-1].block_type is BlockType.PARAGRAPH
    assert blocks[-1].heading_path == ("Appendix",)


# --- parse_pdf: real PDFs ---
# The next three tests check page handling, not scanned detection, so they switch the
# scanned check off with min_chars_per_page=0.


def test_page_numbers_are_one_based(tmp_path: Path) -> None:
    pdf = make_paragraph_pdf(tmp_path / "two.pdf", ["alpha", "beta"])
    doc = parse_pdf(pdf, min_chars_per_page=0)
    assert doc.page_count == 2
    assert [b.page for b in doc.blocks if "alpha" in b.text] == [1]
    assert [b.page for b in doc.blocks if "beta" in b.text] == [2]
    assert [b.page for b in doc.blocks] == sorted(b.page for b in doc.blocks)
    assert doc.fallback_pages == ()


def test_blank_page_counts_but_gives_no_blocks(tmp_path: Path) -> None:
    pdf = make_paragraph_pdf(tmp_path / "blank.pdf", ["alpha", ""])
    doc = parse_pdf(pdf, min_chars_per_page=0)
    assert doc.page_count == 2
    assert {b.page for b in doc.blocks} == {1}
    assert any("alpha" in b.text for b in doc.blocks)
    assert doc.fallback_pages == ()


def test_document_records_the_parser_version(tmp_path: Path) -> None:
    pdf = make_paragraph_pdf(tmp_path / "one.pdf", ["hello"])
    doc = parse_pdf(pdf, min_chars_per_page=0)
    assert PARSER_VERSION == "0.2.0"
    assert doc.parser_version == PARSER_VERSION
    assert any("hello" in b.text for b in doc.blocks)
    assert doc.fallback_pages == ()


def test_pdf_with_heading_paragraph_and_table_gives_structured_blocks(tmp_path: Path) -> None:
    html = """
    <h1>Experimental Results</h1>
    <p>We evaluated the proposed model on three public datasets and report the accuracy
    of every run in the table below. Each experiment was repeated five times.</p>
    <table>
      <tr><th>Dataset</th><th>Accuracy</th><th>Runs</th></tr>
      <tr><td>Alpha</td><td>0.91</td><td>5</td></tr>
      <tr><td>Beta</td><td>0.87</td><td>5</td></tr>
      <tr><td>Gamma</td><td>0.93</td><td>5</td></tr>
    </table>
    """
    css = "table, th, td {border: 1px solid black; border-collapse: collapse; padding: 4px;}"
    path = tmp_path / "structured.pdf"
    document = pymupdf.open()
    document.new_page().insert_htmlbox(pymupdf.Rect(72, 72, 540, 720), html, css=css)
    document.save(path)
    document.close()

    blocks = parse_pdf(path, min_chars_per_page=0).blocks

    headings = [b for b in blocks if b.block_type is BlockType.HEADING and b.page == 1]
    tables = [b for b in blocks if b.block_type is BlockType.TABLE and b.page == 1]
    paragraphs = [b for b in blocks if b.block_type is BlockType.PARAGRAPH and "datasets" in b.text]
    assert headings
    assert tables
    assert "|" in tables[0].text
    assert paragraphs
    assert "Experimental Results" in paragraphs[0].heading_path


def test_parse_pdf_asks_for_per_page_markdown_without_images_or_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = make_pdf(tmp_path / "two.pdf", ["first page", "second page"])
    calls: list[dict[str, Any]] = []

    def fake_to_markdown(doc: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return [
            {"metadata": {"page_number": 1}, "text": "# Title"},
            {"metadata": {"page_number": 2}, "text": "Body."},
        ]

    monkeypatch.setattr(pymupdf4llm, "to_markdown", fake_to_markdown)
    doc = parse_pdf(pdf, min_chars_per_page=0)

    assert calls == [
        {
            "page_chunks": True,
            "write_images": False,
            "use_ocr": False,
            "header": False,
            "footer": False,
        }
    ]
    assert [(b.page, b.text) for b in doc.blocks] == [(1, "Title"), (2, "Body.")]
    assert doc.fallback_pages == ()


def test_page_that_lost_its_text_falls_back_to_plain_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    expected = [f"w{n}" for n in range(70)]
    lines = [" ".join(expected[start : start + 10]) for start in range(0, 70, 10)]
    pdf = make_pdf(tmp_path / "lossy.pdf", ["\n".join(lines)])

    def almost_nothing(doc: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"metadata": {"page_number": 1}, "text": "stub"}]

    monkeypatch.setattr(pymupdf4llm, "to_markdown", almost_nothing)
    caplog.set_level(logging.WARNING, logger="adaptrag.parsing.pdf")
    doc = parse_pdf(pdf)

    assert doc.fallback_pages == (1,)
    assert " ".join(b.text for b in doc.blocks).split() == expected
    assert all(b.block_type is BlockType.PARAGRAPH for b in doc.blocks)
    warnings = [r.getMessage() for r in caplog.records if r.name == "adaptrag.parsing.pdf"]
    assert len(warnings) == 1
    assert "[1]" in warnings[0]


def test_library_failure_becomes_a_parse_error_with_the_cause_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = make_pdf(tmp_path / "one.pdf", ["hello"])
    original = RuntimeError("layout model crashed")

    def crash(doc: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise original

    monkeypatch.setattr(pymupdf4llm, "to_markdown", crash)
    with pytest.raises(CorruptPDFError, match="could not be read") as caught:
        parse_pdf(pdf, min_chars_per_page=0)
    assert caught.value.__cause__ is original


def test_corrupt_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not a real pdf")
    with pytest.raises(CorruptPDFError):
        parse_pdf(path)


def test_encrypted_pdf_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="owner")
    doc.close()
    with pytest.raises(EncryptedPDFError):
        parse_pdf(path)


def test_pdf_without_text_is_rejected_as_scanned(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "scan.pdf", ["", ""])
    with pytest.raises(ScannedPDFError, match="scanned"):
        parse_pdf(pdf)


def test_scanned_pdf_is_rejected_before_markdown_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = make_pdf(tmp_path / "scan.pdf", ["", ""])

    def must_not_run(doc: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("to_markdown must not run on a scanned PDF")

    monkeypatch.setattr(pymupdf4llm, "to_markdown", must_not_run)
    with pytest.raises(ScannedPDFError):
        parse_pdf(pdf)


def test_page_with_enough_text_is_not_scanned(tmp_path: Path) -> None:
    text = ("word " * 10 + "\n") * 5  # 5 lines, about 250 characters
    pdf = make_pdf(tmp_path / "enough.pdf", [text])
    doc = parse_pdf(pdf)
    assert doc.page_count == 1
