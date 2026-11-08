"""Parser: turn a PDF into a ParsedDocument using pymupdf4llm's per-page Markdown.

This is the only module that imports pymupdf or pymupdf4llm (notes 0006 and 0018).
Two steps: pymupdf4llm writes one Markdown string per page, then the pure function
blocks_from_markdown() turns those strings into HEADING, PARAGRAPH and TABLE blocks.
Unusable PDFs raise a ParseError subclass with a message safe to show users.
"""

import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm  # type: ignore[import-untyped]

from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument

logger = logging.getLogger(__name__)

PARSER_VERSION = "0.2.0"

TEXT_BLOCK = 0  # PyMuPDF block type for text; 1 means image
MIN_CHARS_PER_PAGE = 100  # below this on average, treat the PDF as scanned
FALLBACK_MIN_RATIO = 0.7  # Markdown with less than this share of the plain words lost text
FALLBACK_MIN_PLAIN_WORDS = 50  # pages with fewer plain words are never replaced

HEADING_LINE = re.compile(r"^(#{1,6})\s+(.+)$")

# What pymupdf4llm writes around a picture when images are not saved. The text found
# inside the picture stays (parser 0.1.0 kept it too); only these markers go.
PICTURE_TEXT_START = "<!-- Start of picture text -->"
PICTURE_TEXT_END = "<!-- End of picture text -->"
PICTURE_LINE_BREAK = "<br>"
CODE_FENCE = "```"

# (pattern, replacement), applied in this order. Every rule keeps the inner text.
MARKUP_RULES = (
    (re.compile(r"</?(?:sup|u|mark)>"), ""),  # style tags pymupdf4llm emits
    (re.compile(r"\[([^\]]+)\]\([^)\s]+\)"), r"\1"),  # [text](url)
    (re.compile(r"`([^`]+)`"), r"\1"),  # `code`
    (re.compile(r"~~(.+?)~~"), r"\1"),  # ~~strikeout~~
    (re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*"), r"\1"),  # **bold**
    # *italic*: the stars must touch the text, so "2 * 3 * 4" and "2*3*4" stay.
    (re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])"), r"\1"),
    # _italic_: no word character outside the underscores, so snake_case stays.
    (re.compile(r"(?<!\w)_(?=[^\s_])(.+?)(?<=[^\s_])_(?!\w)"), r"\1"),
)


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


def count_words(text: str) -> int:
    """Count words, treating "|" as a separator; a word needs a letter or a digit."""
    tokens = text.replace("|", " ").split()
    return sum(1 for token in tokens if any(char.isalnum() for char in token))


def _normalise_table_line(line: str) -> str:
    """Put one space around every pipe, so "|0.91|0.87|" becomes "| 0.91 | 0.87 |"."""
    return " ".join(line.replace("|", " | ").split())


def _without_placeholders(line: str) -> str:
    """Remove what pymupdf4llm adds that is not document text (empty if nothing is left)."""
    if line.strip() == CODE_FENCE:
        return ""
    line = line.replace(PICTURE_TEXT_START, "").replace(PICTURE_TEXT_END, "")
    return line.replace(PICTURE_LINE_BREAK, " ")


def _plain_text(markdown: str) -> str:
    """Remove emphasis, code and link markup, then normalise the whitespace."""
    for pattern, replacement in MARKUP_RULES:
        markdown = pattern.sub(replacement, markdown)
    return clean_text(markdown)


def _page_items(markdown: str) -> Iterator[tuple[BlockType, str, int | None]]:
    """Yield (block type, text, heading level) for one page's Markdown, in order."""
    paragraph: list[str] = []
    table: list[str] = []

    for raw_line in markdown.split("\n"):
        if raw_line.startswith("|"):
            if paragraph:
                yield BlockType.PARAGRAPH, " ".join(paragraph), None
                paragraph = []
            table.append(_normalise_table_line(raw_line))
            continue
        if table:
            yield BlockType.TABLE, "\n".join(table), None
            table = []

        line = _without_placeholders(raw_line)
        heading = HEADING_LINE.match(line)
        text = _plain_text(heading.group(2) if heading else line)
        if heading is None and text:
            paragraph.append(text)
            continue

        # A heading, a blank line or a dropped line ends the paragraph.
        if paragraph:
            yield BlockType.PARAGRAPH, " ".join(paragraph), None
            paragraph = []
        if heading is not None and text:
            yield BlockType.HEADING, text, len(heading.group(1))

    if paragraph:
        yield BlockType.PARAGRAPH, " ".join(paragraph), None
    if table:
        yield BlockType.TABLE, "\n".join(table), None


def blocks_from_markdown(pages: Sequence[tuple[int, str]]) -> list[ParsedBlock]:
    """Turn (1-based page number, page Markdown) pairs into blocks with heading paths.

    The heading stack runs across pages. Paragraphs and tables do not: a paragraph that
    continues on the next page becomes two blocks, so each block has exactly one page.
    """
    blocks: list[ParsedBlock] = []
    stack: list[tuple[int, str]] = []  # (level, text) of the headings currently open

    for page, markdown in pages:
        for block_type, text, level in _page_items(markdown):
            if level is not None:  # only headings have a level
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, text))
            blocks.append(
                ParsedBlock(
                    text=text,
                    page=page,
                    block_type=block_type,
                    heading_path=tuple(heading for _, heading in stack),
                    level=level,
                )
            )
    return blocks


def _fallback_text(texts: Sequence[str]) -> str:
    """Join plain texts so that none of them can be read as a heading or a table."""
    return "\n\n".join(f" {text}" if text.startswith(("#", "|")) else text for text in texts)


def choose_page_text(
    markdown_pages: Sequence[tuple[int, str]],
    plain_pages: Mapping[int, Sequence[str]],
    min_ratio: float = FALLBACK_MIN_RATIO,
    min_plain_words: int = FALLBACK_MIN_PLAIN_WORDS,
) -> tuple[list[tuple[int, str]], tuple[int, ...]]:
    """Use a page's plain text where its Markdown lost too many words.

    Returns the (page number, text) pairs, in the same order, and the replaced pages.
    """
    pages: list[tuple[int, str]] = []
    replaced: list[int] = []
    for page, markdown in markdown_pages:
        texts = plain_pages.get(page, ())
        plain_words = sum(count_words(text) for text in texts)
        if plain_words >= min_plain_words and count_words(markdown) < min_ratio * plain_words:
            pages.append((page, _fallback_text(texts)))
            replaced.append(page)
        else:
            pages.append((page, markdown))
    return pages, tuple(replaced)


def _plain_pages(doc: Any) -> dict[int, list[str]]:
    """Return each page's cleaned text blocks: {1-based page: [text of each block]}."""
    plain_pages: dict[int, list[str]] = {}
    for page_index, page in enumerate(doc):
        texts = (
            clean_text(raw_block[4])
            for raw_block in page.get_text("blocks")
            if raw_block[6] == TEXT_BLOCK
        )
        plain_pages[page_index + 1] = [text for text in texts if text]
    return plain_pages


def _markdown_pages(doc: Any) -> list[tuple[int, str]]:
    """Run pymupdf4llm and return (1-based page number, page Markdown) pairs."""
    # use_ocr=False: scanned PDFs are rejected, so OCR must never add text to a page.
    # header/footer=False: drops running headers and page numbers the layout model finds.
    page_chunks = pymupdf4llm.to_markdown(
        doc,
        page_chunks=True,
        write_images=False,
        use_ocr=False,
        header=False,
        footer=False,
    )
    # metadata["page_number"] is already 1-based (pymupdf4llm stores page.number + 1).
    return [(chunk["metadata"]["page_number"], chunk["text"]) for chunk in page_chunks]


def parse_pdf(path: Path, min_chars_per_page: int = MIN_CHARS_PER_PAGE) -> ParsedDocument:
    """Parse the PDF at `path` into heading, paragraph and table blocks."""
    try:
        # PyMuPDF's type hints are incomplete (Document() is untyped, iterating it is
        # mistyped), so `doc` is Any. Safe: only this module touches PyMuPDF.
        doc: Any = pymupdf.open(path)  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise CorruptPDFError("The file could not be opened as a PDF.") from exc

    with doc:
        if doc.needs_pass:
            raise EncryptedPDFError("The PDF is password-protected.")
        page_count = doc.page_count
        if page_count == 0:
            raise CorruptPDFError("The PDF has no pages.")

        plain_pages = _plain_pages(doc)
        total_chars = sum(len(text) for texts in plain_pages.values() for text in texts)
        if total_chars < min_chars_per_page * page_count:
            raise ScannedPDFError(
                "No extractable text found. The PDF may be scanned; OCR is not supported yet."
            )

        try:
            markdown_pages = _markdown_pages(doc)
        except Exception as exc:
            logger.exception("pymupdf4llm failed on %s", path.name)
            raise CorruptPDFError("The PDF could not be read.") from exc

    pages, fallback_pages = choose_page_text(markdown_pages, plain_pages)
    if fallback_pages:
        logger.warning("%s: plain text used for pages %s", path.name, list(fallback_pages))

    return ParsedDocument(
        page_count=page_count,
        parser_version=PARSER_VERSION,
        blocks=tuple(blocks_from_markdown(pages)),
        fallback_pages=fallback_pages,
    )
