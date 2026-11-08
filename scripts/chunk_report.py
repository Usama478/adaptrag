"""Compare every registered chunking strategy on dev PDFs (developer tool).

Usage:
    python scripts/chunk_report.py data/samples
    python scripts/chunk_report.py data/samples --show paper_01.pdf

Strategies are built with FakeEmbedder: chunk() never embeds, so no model loads.
Only for dev folders: any path inside data/test_set is refused (note 0017).

Counts are per chunk, so text repeated in an overlap is counted once in each chunk.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from adaptrag.chunking.chunks import TextChunk
from adaptrag.chunking.fixed import DEFAULT_SIZE_WORDS
from adaptrag.chunking.splitter import word_count
from adaptrag.embedding.fake import FakeEmbedder
from adaptrag.parsing.pdf import ParseError, parse_pdf
from adaptrag.strategies.registry import build_strategy, strategy_names

TEST_SET = (Path(__file__).resolve().parents[1] / "data" / "test_set").resolve()

# Glued words found in the Stage 2 chunks (.claude/PROGRESS.md).
GLUED_TOKENS = ("shownin", "Thedecoder", "Anattention", "withkernel", "transformermodel")
PREVIEW_CHUNKS = 4
PREVIEW_CHARACTERS = 100


@dataclass(frozen=True)
class Stats:
    chunks: int
    words: int
    max_words: int
    over_limit: int
    multi_page: int
    table_chunks: int
    glued: int
    seconds: float

    def line(self) -> str:
        # Words over chunks, so a TOTAL is weighted by chunks and not a mean of means.
        mean_words = self.words / self.chunks if self.chunks else 0.0
        return (
            f"chunks={self.chunks} mean_words={mean_words:.1f} max_words={self.max_words} "
            f"over_{DEFAULT_SIZE_WORDS}={self.over_limit} multi_page={self.multi_page} "
            f"table_chunks={self.table_chunks} glued={self.glued} seconds={self.seconds:.4f}"
        )


def is_table_chunk(text: str) -> bool:
    return any(line.startswith("|") for line in text.split("\n"))


def measure(chunks: list[TextChunk], seconds: float) -> Stats:
    counts = [word_count(chunk.text) for chunk in chunks]
    return Stats(
        chunks=len(chunks),
        words=sum(counts),
        max_words=max(counts, default=0),
        over_limit=sum(1 for count in counts if count > DEFAULT_SIZE_WORDS),
        multi_page=sum(1 for chunk in chunks if chunk.page_end > chunk.page_start),
        table_chunks=sum(1 for chunk in chunks if is_table_chunk(chunk.text)),
        glued=sum(chunk.text.count(token) for chunk in chunks for token in GLUED_TOKENS),
        seconds=seconds,
    )


def total(stats: list[Stats]) -> Stats:
    return Stats(
        chunks=sum(s.chunks for s in stats),
        words=sum(s.words for s in stats),
        max_words=max((s.max_words for s in stats), default=0),
        over_limit=sum(s.over_limit for s in stats),
        multi_page=sum(s.multi_page for s in stats),
        table_chunks=sum(s.table_chunks for s in stats),
        glued=sum(s.glued for s in stats),
        seconds=sum(s.seconds for s in stats),
    )


def print_preview(name: str, chunks: list[TextChunk]) -> None:
    print(f"  first {PREVIEW_CHUNKS} chunks of {name}:")
    for chunk in chunks[:PREVIEW_CHUNKS]:
        preview = chunk.text[:PREVIEW_CHARACTERS].replace("\n", "\\n")
        print(
            f"    [{chunk.index}] pages {chunk.page_start}-{chunk.page_end} "
            f"words={word_count(chunk.text)} {preview}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunking strategies on a dev folder.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--show", metavar="NAME", help="also print this PDF's first chunks")
    args = parser.parse_args()

    folder = args.folder.resolve()
    if folder.is_relative_to(TEST_SET):
        print("Error: this report is for dev documents only; data/test_set is not allowed.")
        sys.exit(1)

    # Matching against the folder listing means --show cannot reach outside the folder.
    paths = sorted(folder.glob("*.pdf"))
    if args.show is not None and args.show not in {path.name for path in paths}:
        print(f"Error: no PDF named {args.show} in {folder}.")
        sys.exit(1)

    names = strategy_names()
    width = max(len(name) for name in names)
    per_strategy: dict[str, list[Stats]] = {name: [] for name in names}

    for path in paths:
        try:
            doc = parse_pdf(path)
        except ParseError as exc:
            print(f"{path.name}: REJECTED ({type(exc).__name__}) {exc}")
            continue

        print(path.name)
        chunks_by_strategy: dict[str, list[TextChunk]] = {}
        for name in names:
            strategy = build_strategy(name, FakeEmbedder())
            start = time.perf_counter()
            chunks = strategy.chunk(doc)
            stats = measure(chunks, time.perf_counter() - start)
            per_strategy[name].append(stats)
            chunks_by_strategy[name] = chunks
            print(f"  {name:<{width}} {stats.line()}")

        if path.name == args.show:
            for name in names:
                print_preview(name, chunks_by_strategy[name])

    for name in names:
        print(f"TOTAL {name:<{width}} {total(per_strategy[name]).line()}")


if __name__ == "__main__":
    main()
