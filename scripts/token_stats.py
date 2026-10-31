"""Count real model tokens in the fixed-size chunks of every PDF in a folder.

The chunker counts words; the embedding model counts tokens and cuts text off after
max_tokens. This script measures how many chunks would be cut off.

Usage:
    python scripts/token_stats.py data/samples
    python scripts/token_stats.py data/samples --show-over   # also list each chunk over
"""

import argparse
import statistics
from pathlib import Path

from adaptrag.chunking.fixed import chunk_fixed_size
from adaptrag.config import get_settings
from adaptrag.embedding.sentence_transformer import get_embedder
from adaptrag.parsing.pdf import ParseError, parse_pdf

PREVIEW_CHARS = 200


def summary(counts: list[int], max_tokens: int) -> str:
    over = sum(1 for c in counts if c > max_tokens)
    return (
        f"{len(counts)} chunks, tokens min {min(counts)} / median {statistics.median(counts)}"
        f" / max {max(counts)}, {over} over {max_tokens}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Token counts of fixed-size chunks per PDF.")
    parser.add_argument("folder", type=Path, help="folder with PDF files")
    parser.add_argument(
        "--show-over",
        action="store_true",
        help="after each file, print every chunk longer than max_tokens",
    )
    args = parser.parse_args()

    settings = get_settings()
    embedder = get_embedder(
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_query_instruction,
    )
    print(f"model {embedder.model_name}, max_tokens {embedder.max_tokens}")

    all_counts: list[int] = []
    for path in sorted(args.folder.glob("*.pdf")):
        try:
            doc = parse_pdf(path)
        except ParseError as exc:
            print(f"{path.name}: REJECTED ({exc})")
            continue
        chunks = chunk_fixed_size(doc)
        counts = [embedder.count_tokens(chunk.text) for chunk in chunks]
        if not counts:
            print(f"{path.name}: no chunks")
            continue
        all_counts.extend(counts)
        print(f"{path.name}: {summary(counts, embedder.max_tokens)}")

        if args.show_over:
            for chunk, tokens in zip(chunks, counts, strict=True):
                if tokens > embedder.max_tokens:
                    print(
                        f"  over: chunk {chunk.index}, pages {chunk.page_start}-{chunk.page_end},"
                        f" {tokens} tokens: {chunk.text[:PREVIEW_CHARS]}"
                    )

    if all_counts:
        print(f"OVERALL: {summary(all_counts, embedder.max_tokens)}")


if __name__ == "__main__":
    main()
