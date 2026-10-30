"""Parse every PDF in a folder and print the result (developer tool).

Usage:
    python scripts/check_pdfs.py data/samples
"""

import sys
from pathlib import Path

from adaptrag.parsing.pdf import ParseError, parse_pdf


def main() -> None:
    folder = Path(sys.argv[1])
    for path in sorted(folder.glob("*.pdf")):
        try:
            doc = parse_pdf(path)
        except ParseError as exc:
            print(f"{path.name}: REJECTED ({type(exc).__name__}) {exc}")
        else:
            print(f"{path.name}: OK, {doc.page_count} pages, {len(doc.blocks)} blocks")


if __name__ == "__main__":
    main()
