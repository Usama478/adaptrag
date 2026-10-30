"""Parser v0: turn a PDF into a ParsedDocument using PyMuPDF.

This is the only module that imports PyMuPDF (decision note 0006).
v0 keeps it simple: every text block becomes a PARAGRAPH, in the order
PyMuPDF returns it (note 0007). Headings and tables come in Stage 3.
Unusable PDFs raise a ParseError subclass with a message safe to show users.
"""

from pathlib import Path

import pymupdf

from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument

PARSER_VERSION = "0.1.0"

TEXT_BLOCK = 0  # PyMuPDF block type for text; 1 means image
MIN_CHARS_PER_PAGE = 100  # below this on average, treat the PDF as scanned


class ParseError(Exception):
    """The PDF can't be used. The message is short and safe to show users."""


class CorruptPDFError(ParseError):
    """The file is broken or has no pages."""


class EncryptedPDFError(ParseError):
    """The file is password-protected."""


class ScannedPDFError(ParseError):
    """The file has (almost) no extractable text, e.g. scanned page images."""


def clean_text(raw: str) -> str:
    """Join a block's lines into one line, with single spaces."""
    return " ".join(raw.split())


def parse_pdf(path: Path, min_chars_per_page: int = MIN_CHARS_PER_PAGE) -> ParsedDocument:
    """Parse the PDF at `path` into text blocks with 1-based page numbers."""
    try:
        # PyMuPDF's Document() has no type hints; strict mypy refuses untyped calls.
        doc = pymupdf.open(path)  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise CorruptPDFError("The file could not be opened as a PDF.") from exc

    blocks: list[ParsedBlock] = []
    with doc:
        if doc.needs_pass:
            raise EncryptedPDFError("The PDF is password-protected.")
        page_count = doc.page_count
        if page_count == 0:
            raise CorruptPDFError("The PDF has no pages.")
        for page_index, page in enumerate(doc):
            for raw_block in page.get_text("blocks"):
                if raw_block[6] != TEXT_BLOCK:
                    continue
                text = clean_text(raw_block[4])
                if not text:
                    continue
                blocks.append(
                    ParsedBlock(
                        text=text,
                        page=page_index + 1,
                        block_type=BlockType.PARAGRAPH,
                    )
                )

    total_chars = sum(len(block.text) for block in blocks)
    if total_chars < min_chars_per_page * page_count:
        raise ScannedPDFError(
            "No extractable text found. The PDF may be scanned; OCR is not supported yet."
        )

    return ParsedDocument(
        page_count=page_count,
        parser_version=PARSER_VERSION,
        blocks=tuple(blocks),
    )
