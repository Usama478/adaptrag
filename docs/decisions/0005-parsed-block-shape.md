# 0005: Parser output shape (ParsedBlock, ParsedDocument)

Date: 2026-10-30
Status: Accepted

## Context
The parser reads PDFs with a third-party library. The profiler, chunkers and
citations all need its output. If they read the library's own objects, all of
them depend on that library, and every one of their tests needs a real PDF.

## Decision
- The parser returns our own shapes, defined in `src/adaptrag/parsing/blocks.py`:
  - `ParsedBlock`: text, page (1-based physical page), block_type, heading_path.
  - `ParsedDocument`: page_count, parser_version, blocks in reading order.
- `BlockType` is a StrEnum with three values: heading, paragraph, table.
- Both are frozen dataclasses. `__post_init__` rejects page < 1, blank text,
  page_count < 1, and any block whose page is beyond page_count.
- Only the parser imports the PDF library. Swapping libraries changes one file.

## Rejected
- dict: typos in keys fail silently; mypy can't check them.
- TypedDict: typed, but mutable and can't validate itself.
- NamedTuple: behaves like a tuple (index access, equal to plain tuples).
- pydantic: built for checking untrusted input at runtime. Parser output is our
  own code, which mypy checks. Pydantic stays at the API boundary.
- Printed journal page numbers: a PDF viewer opens physical pages.

## Known limits
- Types are not checked at runtime: `block_type="tabel"` is accepted. mypy
  catches this in `src/`, but not in interactive Python or outside code.
- No position on the page (bounding box) yet. Needed for Stage 8 highlighting.
- A heading block's heading_path ends with itself. The heading detector
  (Step 8) must follow this convention.
- Blocks are not saved to the database. Chunks will be (Step 12).