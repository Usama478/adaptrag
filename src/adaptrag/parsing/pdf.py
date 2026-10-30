"""Parser v0: turn a PDF into a ParsedDocument using PyMuPDF.

This is the only module that imports PyMuPDF (decision note 0006).
v0 keeps it simple: every text block becomes a PARAGRAPH, in the order
PyMuPDF returns it. Reading order (Step 7), headings (Step 8) and tables
(Step 9) come later.
"""

from pathlib import Path

import pymupdf

from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument

PARSER_VERSION = "0.1.0"

TEXT_BLOCK = 0  # PyMuPDF block type for text; 1 means image


def clean_text(raw: str) -> str:
    """Join a block's lines into one line, with single spaces."""
    return " ".join(raw.split())


def parse_pdf(path: Path) -> ParsedDocument:
    """Parse the PDF at `path` into text blocks with 1-based page numbers."""
    blocks: list[ParsedBlock] = []
    # PyMuPDF's Document() has no type hints; strict mypy refuses untyped calls.
    with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
        page_count = doc.page_count
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
    return ParsedDocument(
        page_count=page_count,
        parser_version=PARSER_VERSION,
        blocks=tuple(blocks),
    )
