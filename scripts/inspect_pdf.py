"""Print PyMuPDF's raw output for one page of a PDF.

A developer tool for looking at what the PDF library returns before we
write parser code. Usage:
    python scripts/inspect_pdf.py data/samples/paper_01_single_column.pdf 1
"""

import sys

import pymupdf


def main() -> None:
    path = sys.argv[1]
    page_number = int(sys.argv[2])  # 1-based, like a PDF viewer

    doc = pymupdf.open(path)
    page = doc[page_number - 1]  # PyMuPDF counts from 0
    print(f"pages in document: {doc.page_count}")
    print(f"page size: {page.rect}")
    print()

    print("BLOCKS: number, x0, y0, x1, type, first 60 characters")
    for b in page.get_text("blocks"):
        print(b[5], round(b[0]), round(b[1]), round(b[2]), b[6], repr(b[4][:60]))
    print()

    print("FONTS: size, bold?, text  (first span of each line, text blocks only)")
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            span = line["spans"][0]
            bold = bool(span["flags"] & 16)
            print(round(span["size"], 1), bold, repr(span["text"][:60]))


if __name__ == "__main__":
    main()
