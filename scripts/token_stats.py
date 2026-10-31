"""Count real model tokens in the fixed-size chunks of every PDF in a folder.

The chunker counts words; the embedding model counts tokens and cuts text off after
max_tokens. This script measures how many chunks would be cut off.

Usage: python scripts/token_stats.py data/samples
"""

import statistics
import sys
from pathlib import Path

from adaptrag.chunking.fixed import chunk_fixed_size
from adaptrag.config import get_settings
from adaptrag.embedding.sentence_transformer import get_embedder
from adaptrag.parsing.pdf import ParseError, parse_pdf


def summary(counts: list[int], max_tokens: int) -> str:
    over = sum(1 for c in counts if c > max_tokens)
    return (
        f"{len(counts)} chunks, tokens min {min(counts)} / median {statistics.median(counts)}"
        f" / max {max(counts)}, {over} over {max_tokens}"
    )


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/token_stats.py FOLDER")
        raise SystemExit(2)

    settings = get_settings()
    embedder = get_embedder(
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_query_instruction,
    )
    print(f"model {embedder.model_name}, max_tokens {embedder.max_tokens}")

    all_counts: list[int] = []
    for path in sorted(Path(sys.argv[1]).glob("*.pdf")):
        try:
            doc = parse_pdf(path)
        except ParseError as exc:
            print(f"{path.name}: REJECTED ({exc})")
            continue
        counts = [embedder.count_tokens(chunk.text) for chunk in chunk_fixed_size(doc)]
        if not counts:
            print(f"{path.name}: no chunks")
            continue
        all_counts.extend(counts)
        print(f"{path.name}: {summary(counts, embedder.max_tokens)}")

    if all_counts:
        print(f"OVERALL: {summary(all_counts, embedder.max_tokens)}")


if __name__ == "__main__":
    main()
