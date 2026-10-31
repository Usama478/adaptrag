"""Chunk every PDF in a folder with the fixed-size chunker and print a summary.

Usage: python scripts/chunk_pdfs.py data/samples
"""

import sys
from pathlib import Path

from adaptrag.chunking.fixed import chunk_fixed_size
from adaptrag.parsing.pdf import ParseError, parse_pdf


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/chunk_pdfs.py FOLDER")
        raise SystemExit(2)

    folder = Path(sys.argv[1])
    for path in sorted(folder.glob("*.pdf")):
        try:
            doc = parse_pdf(path)
        except ParseError as exc:
            print(f"{path.name}: REJECTED ({exc})")
            continue

        chunks = chunk_fixed_size(doc)
        word_count = sum(len(block.text.split()) for block in doc.blocks)
        crossing = sum(1 for c in chunks if c.page_end > c.page_start)
        print(
            f"{path.name}: {doc.page_count} pages, {word_count} words, "
            f"{len(chunks)} chunks, {crossing} cross a page"
        )


if __name__ == "__main__":
    main()
