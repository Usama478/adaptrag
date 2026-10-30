# 0007: Reading order (keep PyMuPDF's stored order)

Date: 2026-10-30
Status: Accepted

## Context
Two-column papers must be read down the left column, then down the right.
Wrong order would glue unrelated paragraphs into one chunk.

## Evidence
- paper_02 pages 1-2: PyMuPDF's block order already goes down the left
  column (x 50-286), then down the right (x 309-545). Blocks spanning both
  columns (title, authors) come first.
- paper_02 is the only two-column sample so far; paper_03 turned out to be
  single-column and was renamed.
- PyMuPDF follows the order text is stored in the file; LaTeX stores it in
  column order.

## Decision
Keep PyMuPDF's stored block order in the parser. No re-sorting.

## Rejected
- Sorting by y, then x (PyMuPDF `sort=True`): breaks correct order. On paper_02
  page 2 the right column starts at y=75 but the left column's text at y=183
  (a figure sits above it), so sorting would interleave the columns.
- A custom column-detection algorithm now: no real failing example to build
  or test it against.

## Known limits
- Evidence is one two-column paper. Re-check when more two-column papers are
  added (for example an ACL or IEEE paper).
- Order depends on how the PDF was produced. Non-LaTeX tools may store text
  out of order. Revisit with a real failing document as the test case.
- Figure labels (axis numbers, diagram words), page numbers, footnotes and the
  arXiv margin stamp come through as small blocks.
