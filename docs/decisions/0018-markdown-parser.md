# 0018: Markdown parser with pymupdf4llm

Status: Accepted
Date: 2026-11-08

## Context
Parser 0.1.0 made every PyMuPDF text block a PARAGRAPH. Section-aware chunking and the
router's "enough headings" rule need headings and tables. Note 0008 drafted our own
heading rules, but they missed IEEE-style headings and would be more code to own.

## Decision
- **Use `pymupdf4llm.to_markdown` with per-page output** (`page_chunks=True`). It is the
  same library family as PyMuPDF, is already installed (PyMuPDF 2.0 ships it), and gives
  headings and tables in one call.
- **Block conversion is a pure function.** `blocks_from_markdown()` takes
  (page number, Markdown) pairs and returns `ParsedBlock`s, so its rules are tested with
  plain strings. `parse_pdf()` keeps its signature and error types. Only `parsing/pdf.py`
  imports `pymupdf` or `pymupdf4llm`.
- **Line rules.** `#` to `######` is a HEADING with that level; a heading stack across
  pages gives every block its heading path. Consecutive `|` lines are one TABLE block.
  Other consecutive lines are one PARAGRAPH. Blocks never span pages.
- **Tables stay Markdown tables**, because the pipes carry the row and column structure.
  Table rows are normalised to spaced cells ('|0.91|0.87|' -> '| 0.91 | 0.87 |'); cell
  text is otherwise unchanged.
- **Page headers and footers are dropped** (header=False, footer=False). Measured with
  whitespace word counts on the 9 dev documents: digit-only blocks 60 -> 0, words
  63,507 -> 63,431, identical headings and tables.
- **Words are counted with count_words():** pipes become spaces and only tokens with a
  letter or digit count.
- **Per-page fallback:** if a page's Markdown keeps fewer than 70% of its plain words
  (count_words) and the page has at least 50 plain words, its plain text is used and the
  page is listed in ParsedDocument.fallback_pages. On MTEB, whitespace counting first
  suggested 80-96% loss on pages 15 and 20-24, but that was mostly table rows glued into
  one 'word' (when those pages fell back, tables dropped 14 -> 7). With count_words only
  pages 4, 20 and 22 fall back; page 21 keeps 1,758 of 1,903 words.
- **Emphasis is stripped** from heading and paragraph text (bold, italic, code,
  strikeout, links, and the `<sup>`, `<u>`, `<mark>` tags the library emits), so markup
  does not become part of search tokens. Underscores inside words stay.
- **Picture markers are dropped, picture text stays.** Parser 0.1.0 also kept text found
  inside figures; removing it is a page-furniture decision for later.
- **The scanned rule runs before Markdown and is unchanged** (same count of extractable
  characters, same threshold). OCR is switched off (`use_ocr=False`): otherwise an
  installed OCR engine could add text to scanned pages, and scanned PDFs must be rejected.
- **A pymupdf4llm failure becomes `CorruptPDFError`**, with the real cause logged and
  kept as the exception cause.
- **Page map deferred to Step 4.** Blocks already carry their page.
- **Glued words, hyphenation and page furniture are measured before anything is fixed.**
  `scripts/parser_report.py` compares both parsers on dev documents.
- `PARSER_VERSION` is `0.2.0`, so chunks and vectors from 0.1.0 are no longer current.

## Rejected
- **Docling, Marker:** much larger layout and OCR models, slower, extra dependencies;
  Marker is GPL. pymupdf4llm also uses a layout model, but a small one that ships with
  PyMuPDF 2.0.
- **Keeping plain PyMuPDF blocks:** no structure, so no section-aware strategy.
- **Our own heading rules (note 0008):** superseded by this note unless the evaluation
  shows a need.

## Library options checked (pymupdf4llm 2.0 source)
- No option keeps table boxes the model cannot build a grid for.
- Picture text is already kept (`force_text` defaults to True).
- `table_output="html"` and `use_layout(False)` exist but were not adopted.

## Known limits
- Heading detection relies on pymupdf4llm: in 2.0 a small ONNX layout model finds the
  heading boxes and the rounded font size picks the level. Neither is under our control,
  and levels can differ from the paper's numbering.
- The layout model loads when `pymupdf4llm` is imported, and runs per page on the CPU.
- Parse time: 14.63 s for the 9 dev documents (MTEB, 24 pages: 5.93 s); the layout
  model's load time at import is not measured.
- Report (count_words; old plain parser -> new): words 65,308 -> 63,518; headings
  0 -> 185; tables 0 -> 49; hyphen breaks 927 -> 15; digit-only blocks 133 -> 22 (all on
  MTEB fallback pages); fallback pages: paper_09 [4, 20, 22], others none.
- Rules are line-based. A code listing line such as `# compute loss`, or a paragraph
  starting with `# `, becomes a HEADING.
- Footnotes keep the library's `> ` prefix and list items its `- ` prefix.
- The fixed-size chunker splits on whitespace, so each | and --- counts as a word: a
  200-word window over a table holds fewer real words (revisit in Step 4).
- Fallback pages lose structure (MTEB tables 14 -> 12) and keep page numbers and hyphen
  breaks.
- The 70% / 50-word thresholds come from one document, and MTEB page 23 sits on the
  boundary (589 of 841 words, kept).
- A page whose only text is a short isolated line can be dropped as a header/footer and
  is never rescued (pages under 50 words), e.g. a one-line cover page.
- False headings: "end for" twice in GAN pseudo-code, "greatly benefited from very deep
  models." in ResNet.
- The Stage 2 glued words were found in neither parser's output, so their origin must be
  rechecked on chunks in Step 4.
- Stage 2 gate evidence strings were copied from 0.1.0 chunks and may no longer match the
  new text.
