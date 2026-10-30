# 0006: PDF library (PyMuPDF)

Date: 2026-10-30
Status: Accepted

## Context
The parser needs, from each PDF page: text grouped into blocks, each block's
position (for two-column reading order), font size and boldness (for heading
detection), and help finding tables.

## Decision
- PyMuPDF, and only PyMuPDF. It is imported only inside the parser module;
  everything else reads `ParsedBlock` (note 0005).
- Chosen for: blocks with positions and fonts out of the box, and speed
  (we re-parse 15 to 25 papers many times during evaluation).

## Rejected
- pdfplumber: very good tables, MIT license, but slower and gives words rather
  than blocks. Would be the choice for a closed-source product.
- pypdf: mostly plain text, weak position info; can't support headings or
  reading order.
- ML layout tools (docling, unstructured, marker): heavy, slow, and they hide
  the heading and table logic that the profiler and router depend on.
- Two libraries at once: parser differences would mix into the router
  benchmark (we hold the parser fixed so we measure only routing); double
  code and tests; and a PDF must be parsed before we know which to pick.

## Known limits
- License is AGPL. Fine for this public open-source repo; a closed-source
  product would need a commercial license or a switch to pdfplumber.
- Table detection quality is unknown until Step 9. If it is poor on the
  table-heavy paper, revisit pdfplumber for tables only, with measurements.