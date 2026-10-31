# 0011: Chunk storage

Status: Accepted
Update: vectors are stored with pgvector in Postgres, not Chroma (note 0013).
Date: 2026-10-31

## Context
Stage 2 Step 2 stores the chunks made by the fixed-size chunker (0010) during ingestion.
Later stages run several strategies on the same document and compare them, so a
document can have more than one way of being cut. Chunk text must survive re-embedding
and model changes, and "delete means delete" must remove every chunk.

## Decision
- Two tables:
  - `chunk_sets`: document_id (FK, ON DELETE CASCADE), strategy, params (JSONB),
    parser_version, created_at. Unique on (document_id, strategy, params, parser_version).
  - `chunks`: chunk_set_id (FK, ON DELETE CASCADE), chunk_index, text, page_start,
    page_end. Unique on (chunk_set_id, chunk_index).
- Postgres is the source of truth for chunk text. Chroma (Step 4) will hold only vectors
  keyed by chunk id, so vectors can be rebuilt from Postgres at any time.
- `save_chunk_set` checks for an identical set first and skips it; the unique
  constraint is the safety net if the check is ever bypassed.
- Parse, chunk and save run in one transaction; any error rolls it back, so a failed
  job leaves no partial chunk set.
- Deletion is handled by ON DELETE CASCADE: document -> chunk sets -> chunks.
- The `uq` naming convention now uses `column_0_N_name`, so multi-column unique
  constraints get names listing every column.
- No new document status yet ("parsed" stays the success status); a new status comes
  when vectors exist (Step 4).

## Rejected
- One flat `chunks` table with strategy and params on every row: repeats the settings
  on every chunk and cannot express "this set already exists" with one constraint.
- Chunks only in Chroma: the vector store would become the source of truth for text,
  and changing the embedding model would mean re-chunking.
- Extra indexes on the same columns as the unique constraints: Postgres already builds
  an index for each unique constraint, so they would only duplicate it.

## Known limits
- Documents ingested before this step have no chunks, and duplicate uploads are not
  re-processed, so they stay without chunks until a re-ingest path exists.
- Page rules (page_start >= 1, page_end >= page_start) are enforced only in
  `TextChunk`, not by database check constraints.
- Check-then-insert can race: two jobs for the same document could both pass the check.
  The unique constraint would then fail one job; data would not be duplicated.
