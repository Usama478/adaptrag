"""Document profile: a small record of signals measured from a parsed document.

The router will read these signals. This module only measures; it holds no rules.
"""

import re
from dataclasses import asdict, dataclass

from adaptrag.parsing.blocks import BlockType, ParsedDocument
from adaptrag.parsing.pdf import count_words

DECIMALS = 3

# A section number at the start of a heading, followed by whitespace. Requiring the
# whitespace is what keeps "3D models are hard" out. A letter or a roman numeral needs
# its dot, so "A Survey of Models" is not numbered.
SECTION_NUMBER = re.compile(
    r"(?:"
    r"\d+(?:\.\d+)*\.?"  # 3, 3.1, 3.1.2, and the same with a final dot
    r"|[IVX]+\."  # III.
    r"|[A-Z]\."  # A.
    r")\s"
)


@dataclass(frozen=True, slots=True)
class DocumentProfile:
    """Signals for one document. Shares and rates are rounded to 3 decimals.

    words: count_words over all blocks, so heading and table words are included.
    numbered_heading_share: headings that start with a section number / headings.
    table_word_share: words in TABLE blocks / words.
    fallback_page_share: pages built from plain text / page_count.
    """

    page_count: int
    words: int
    headings: int
    numbered_heading_share: float
    headings_per_1k_words: float
    tables: int
    table_word_share: float
    fallback_pages: int
    fallback_page_share: float
    words_per_page: float

    def to_dict(self) -> dict[str, int | float]:
        """Return the fields as a JSON-serialisable dict."""
        return asdict(self)


def _ratio(part: int, whole: int) -> float:
    """Return part / whole rounded, or 0.0 when there is nothing to divide by."""
    if whole == 0:
        return 0.0
    return round(part / whole, DECIMALS)


def profile_document(doc: ParsedDocument) -> DocumentProfile:
    """Measure the signals of a parsed document."""
    words = 0
    headings = 0
    numbered_headings = 0
    tables = 0
    table_words = 0

    for block in doc.blocks:
        block_words = count_words(block.text)
        words += block_words
        if block.block_type is BlockType.HEADING:
            headings += 1
            if SECTION_NUMBER.match(block.text):
                numbered_headings += 1
        elif block.block_type is BlockType.TABLE:
            tables += 1
            table_words += block_words

    fallback_pages = len(doc.fallback_pages)
    return DocumentProfile(
        page_count=doc.page_count,
        words=words,
        headings=headings,
        numbered_heading_share=_ratio(numbered_headings, headings),
        headings_per_1k_words=_ratio(headings * 1000, words),
        tables=tables,
        table_word_share=_ratio(table_words, words),
        fallback_pages=fallback_pages,
        fallback_page_share=_ratio(fallback_pages, doc.page_count),
        words_per_page=_ratio(words, doc.page_count),
    )
