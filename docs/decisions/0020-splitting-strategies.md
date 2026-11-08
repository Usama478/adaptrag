# 0020: Recursive and section-aware splitting strategies

Status: Accepted
Date: 2026-11-09

## Context
The router needs more than the fixed-size control to choose from. Step 4 adds a recursive
strategy that cuts at paragraph and sentence boundaries, and a section-aware strategy that
follows the document's headings. Both must still report the pages of every chunk, because
answers cite pages. The owner decided to use a well-tested library for splitting, not
custom chunkers; the fixed-size control stays our own code (note 0010).

## Decision
- **`langchain-text-splitters` (`RecursiveCharacterTextSplitter`) behind one wrapper.**
  `chunking/splitter.py` is the only module that imports it, so the library can be
  replaced without touching the strategies.
- **The wrapper computes offsets itself,** with `text.find` starting from the previous
  start + 1. The library's `add_start_index` assumes that `length_function` counts
  characters: it subtracts the overlap from a character position, and our overlap is in
  words, so its offsets would be wrong.
- **One 200/40-word budget for all strategies,** so the benchmark measures boundaries, not
  size.
- **Page map** (`chunking/text_map.py`). Blocks are joined into one string and each block's
  character range is remembered, so a piece's offsets give its first and last page.
- **Tables are their own chunks,** never joined to prose. A table over the budget is split
  into whole rows, with no overlap.
- **Shared helpers in `chunking/segments.py`** (`chunk_prose`, `chunk_table`,
  `chunk_blocks`), so both strategies apply the same prose and table rules.
- **Section-aware groups blocks by the parser's `heading_path`.** A new section starts at
  every HEADING block, and chunks never cross a section.
- **Each section-aware chunk starts with its heading path** ("3 Method > 3.2 Training"),
  which counts towards the budget. A path over 40 words is replaced by the innermost
  heading, cut to its first 40 words. The prefix has no page: pages come from the body text.
- **`"version"` in params.** Separators, table handling and prefix rules are code, not
  settings, but they change the chunks. Bumping `"version"` changes the cache key.

## Dependency cost
`langchain-text-splitters` brings `langchain-core` and with it `langsmith`, `jsonpatch`,
`tenacity`, `orjson`, `requests`, `zstandard`, `xxhash`, `uuid-utils` and
`langchain-protocol`. `langsmith` registers a pytest plugin. Its tracing only runs if its
environment variables are set; we set none.

## Rejected
- **`MarkdownHeaderTextSplitter`:** it reports no offsets, so the pages would be lost. The
  parser's `heading_path` already holds the same structure.
- **The library's `add_start_index`:** wrong offsets when sizes are in words (see above).

## Known limits
- Identical repeated text can map a piece to an earlier copy. The page is wrong only if the
  copies are on different pages.
- `>` and `|` count as words, so a prefix or a table holds fewer real words than its count.
- False headings create false sections (examples in note 0018).
- Section-aware differs from recursive in boundaries AND context (the prefix), so Stage 4
  cannot separate the two effects. A Stage 5 switch could.
- Chunk report (`scripts/chunk_report.py`, owner's run, TOTAL lines for the 9 dev documents):
  ```
  fixed_size    chunks=482 mean_words=198.2 max_words=200 over_200=0 multi_page=112 table_chunks=42 glued=0
  recursive     chunks=531 mean_words=153.0 max_words=200 over_200=0 multi_page=71 table_chunks=125 glued=0
  section_aware chunks=614 mean_words=141.7 max_words=200 over_200=0 multi_page=54 table_chunks=126 glued=0
  ```
  Chunking time for all 9 documents: about 0.004 s (fixed), 0.010 s (recursive), 0.009 s
  (section_aware).
- Correction: `glued=0` above covers only the 5 Stage 2 tokens, so it does not show that
  glued words are gone. "theWMT" was found in a paper_01 chunk in Step 6 (note 0022).
- Same 200-word budget, but respecting boundaries makes chunks smaller on average (198.2 vs
  153.0 vs 141.7 words), so the evaluation compares boundary placement and resulting chunk
  size together.
- `table_chunks` is not comparable for fixed_size (it joins words with spaces, so only a
  chunk starting with "|" is counted).
- Leftover markup in chunk text: author footnote markers like "Vaswani_∗_" and the "> "
  prefix of footnotes.
- The dependency list and `add_start_index` behaviour are confirmed by the owner's pip
  output and by the Step 4b reading of the installed library.
