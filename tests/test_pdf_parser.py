"""Tests for parser v0. Each test builds its own tiny PDF; no sample files needed."""

from pathlib import Path

import pymupdf
import pytest

from adaptrag.parsing.blocks import BlockType
from adaptrag.parsing.pdf import (
    PARSER_VERSION,
    CorruptPDFError,
    EncryptedPDFError,
    ScannedPDFError,
    clean_text,
    parse_pdf,
)


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a small PDF with one text string per page ("" means a blank page)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def test_clean_text_joins_lines_and_extra_spaces() -> None:
    assert clean_text("permission to\nreproduce  the\n") == "permission to reproduce the"


def test_clean_text_of_only_whitespace_is_empty() -> None:
    assert clean_text(" \n\t ") == ""


# The next three tests check page handling, not scanned detection. Their tiny PDFs
# hold only a few words, so they switch the scanned check off with min_chars_per_page=0.


def test_page_numbers_are_one_based(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "two.pdf", ["first page", "second page"])
    doc = parse_pdf(pdf, min_chars_per_page=0)
    assert doc.page_count == 2
    assert [(b.page, b.text) for b in doc.blocks] == [(1, "first page"), (2, "second page")]


def test_blank_page_counts_but_gives_no_blocks(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "blank.pdf", ["only text", ""])
    doc = parse_pdf(pdf, min_chars_per_page=0)
    assert doc.page_count == 2
    assert [b.page for b in doc.blocks] == [1]


def test_v0_marks_every_block_as_paragraph(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "one.pdf", ["hello"])
    doc = parse_pdf(pdf, min_chars_per_page=0)
    assert doc.parser_version == PARSER_VERSION
    assert all(b.block_type is BlockType.PARAGRAPH for b in doc.blocks)


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


def test_page_with_enough_text_is_not_scanned(tmp_path: Path) -> None:
    text = ("word " * 10 + "\n") * 5  # 5 lines, about 250 characters
    pdf = make_pdf(tmp_path / "enough.pdf", [text])
    doc = parse_pdf(pdf)
    assert doc.page_count == 1
