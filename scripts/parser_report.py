"""Compare the old "plain" parser with the Markdown parser on dev PDFs (developer tool).

Usage:
    python scripts/parser_report.py data/samples
    python scripts/parser_report.py data/samples --pages paper_01.pdf

Engines: "plain" (old parser 0.1.0) and "markdown" (parse_pdf, including its page fallback).
Only for dev folders: any path inside data/test_set is refused (note 0017).
"""

import argparse
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from adaptrag.parsing.blocks import BlockType, ParsedBlock
from adaptrag.parsing.pdf import TEXT_BLOCK, ParseError, clean_text, count_words, parse_pdf

TEST_SET = (Path(__file__).resolve().parents[1] / "data" / "test_set").resolve()

# Glued words found in the Stage 2 chunks (.claude/PROGRESS.md).
GLUED_TOKENS = ("shownin", "Thedecoder", "Anattention", "withkernel", "transformermodel")
HYPHEN_BREAK = re.compile(r"\b[A-Za-z]{2,}- [a-z]{2,}")
DIGITS_ONLY = re.compile(r"\d+")
HEADINGS_SHOWN = 8
LOSS_RATIO = 0.7  # a page is flagged when markdown keeps less than this share of plain's words


@dataclass(frozen=True)
class Stats:
    words: int
    headings: int
    tables: int
    glued: int
    hyphen_breaks: int
    digit_only: int
    seconds: float

    def line(self) -> str:
        return (
            f"words={self.words} headings={self.headings} tables={self.tables} "
            f"glued={self.glued} hyphen_breaks={self.hyphen_breaks} "
            f"digit_only={self.digit_only} seconds={self.seconds:.2f}"
        )


def measure(texts: list[str], headings: int, tables: int, seconds: float) -> Stats:
    return Stats(
        words=sum(count_words(text) for text in texts),
        headings=headings,
        tables=tables,
        glued=sum(text.count(token) for text in texts for token in GLUED_TOKENS),
        hyphen_breaks=sum(len(HYPHEN_BREAK.findall(text)) for text in texts),
        digit_only=sum(1 for text in texts if DIGITS_ONLY.fullmatch(text)),
        seconds=seconds,
    )


def measure_blocks(blocks: list[ParsedBlock], seconds: float) -> Stats:
    headings = sum(1 for b in blocks if b.block_type is BlockType.HEADING)
    tables = sum(1 for b in blocks if b.block_type is BlockType.TABLE)
    return measure([b.text for b in blocks], headings, tables, seconds)


def total(stats: list[Stats]) -> Stats:
    return Stats(
        words=sum(s.words for s in stats),
        headings=sum(s.headings for s in stats),
        tables=sum(s.tables for s in stats),
        glued=sum(s.glued for s in stats),
        hyphen_breaks=sum(s.hyphen_breaks for s in stats),
        digit_only=sum(s.digit_only for s in stats),
        seconds=sum(s.seconds for s in stats),
    )


def parse_plain(path: Path) -> list[tuple[int, str]]:
    """The old parser (0.1.0): (1-based page, cleaned text) per PyMuPDF text block."""
    items: list[tuple[int, str]] = []
    with pymupdf.open(path) as doc:
        for page_index, page in enumerate(doc):
            for raw_block in page.get_text("blocks"):
                if raw_block[6] != TEXT_BLOCK:
                    continue
                text = clean_text(raw_block[4])
                if text:
                    items.append((page_index + 1, text))
    return items


def words_per_page(items: list[tuple[int, str]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for page, text in items:
        counts[page] += count_words(text)
    return counts


def page_items(blocks: list[ParsedBlock]) -> list[tuple[int, str]]:
    return [(b.page, b.text) for b in blocks]


def print_pages(
    page_count: int,
    plain: list[tuple[int, str]],
    markdown: list[ParsedBlock],
) -> None:
    plain_words = words_per_page(plain)
    markdown_words = words_per_page(page_items(markdown))
    for page in range(1, page_count + 1):
        loss = markdown_words[page] < LOSS_RATIO * plain_words[page]
        print(
            f"page {page}  plain={plain_words[page]}  markdown={markdown_words[page]}"
            + ("  <-- LOSS" if loss else "")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PDF parsers on a dev folder.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--pages", metavar="NAME", help="report only this PDF, with per-page words")
    args = parser.parse_args()

    folder = args.folder.resolve()
    if folder.is_relative_to(TEST_SET):
        print("Error: this report is for dev documents only; data/test_set is not allowed.")
        sys.exit(1)

    # Matching against the folder listing means --pages cannot reach outside the folder.
    paths = sorted(folder.glob("*.pdf"))
    if args.pages is not None:
        paths = [path for path in paths if path.name == args.pages]
        if not paths:
            print(f"Error: no PDF named {args.pages} in {folder}.")
            sys.exit(1)

    all_plain: list[Stats] = []
    all_markdown: list[Stats] = []
    for path in paths:
        start = time.perf_counter()
        try:
            parsed = parse_pdf(path)
        except ParseError as exc:
            print(f"{path.name}: REJECTED ({type(exc).__name__}) {exc}")
            continue
        markdown_seconds = time.perf_counter() - start

        start = time.perf_counter()
        plain_items = parse_plain(path)
        plain_seconds = time.perf_counter() - start

        markdown_blocks = list(parsed.blocks)
        plain = measure([text for _, text in plain_items], 0, 0, plain_seconds)
        markdown = measure_blocks(markdown_blocks, markdown_seconds)
        all_plain.append(plain)
        all_markdown.append(markdown)

        print(path.name)
        print(f"  plain    {plain.line()}")
        print(f"  markdown {markdown.line()} fallback={list(parsed.fallback_pages)}")
        print(f"  first {HEADINGS_SHOWN} headings:")
        headings = [b for b in markdown_blocks if b.block_type is BlockType.HEADING]
        for heading in headings[:HEADINGS_SHOWN]:
            print(f"    h{heading.level} {heading.text}")
        if args.pages is not None:
            print_pages(parsed.page_count, plain_items, markdown_blocks)

    print(f"TOTAL plain    {total(all_plain).line()}")
    print(f"TOTAL markdown {total(all_markdown).line()}")


if __name__ == "__main__":
    main()
