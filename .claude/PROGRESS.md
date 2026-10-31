# AdaptRAG Progress
 
Update this file at the end of every chat. Newest stage at the top of the log.
 
## Current status
- **Current stage:** Stage 2 in progress. Steps 1 and 2 DONE.
  **Next: Stage 2, Step 3** (embeddings with BAAI/bge-small-en-v1.5 behind an Embedder seam).
- **Last updated:** 2026-10-31
- **Repo:** public GitHub repo `adaptrag`, branch `main`, CI green (Postgres service in CI).
- **Tests:** 66 passing. Decision notes 0001-0011 (0008 is "Proposed").
- **Branch:** Step 2 committed on `stage2-step2-chunk-storage` (not pushed or merged yet).
- **Working mode (from Stage 2 Step 2):** Claude Code implements each step from a prompt
  on a branch; the owner reviews the diff in chat before merging; no fail-on-purpose
  experiments; the owner runs manual checks when asked.
## Stage checklist
| Stage | Name | Status |
| --- | --- | --- |
| 0 | Skeleton | Done |
| 1 | Upload, parsing, job status | Done (headings and tables moved to Stage 3) |
| 2 | Baseline RAG | In progress (Steps 1-2 done) |
| 3 | Strategy interface, profiler, router (+ headings, tables) | Not started |
| 4 | Mini evaluation | Not started |
| 5 | Hybrid, rerank, parent retrieval | Not started |
| 6 | Full evaluation | Not started |
| 7 | Notebooks and multi-document search | Not started |
| 8 | Frontend | Not started |
| 9 | Hardening, deploy, README | Not started |
 
Parallel task: evaluation question set. **Deferred to Stage 4 (owner's decision).**
`eval/questions/` is empty. `eval/question_format_example.json` shows the format
(made-up examples, not real questions). At Stage 4: LLM drafts questions in that
format, owner verifies each against the PDF and sets `"verified": true`.
 
## Stage 2 plan (approved)
1. `TextChunk` + fixed-size chunker, 200/40 words, bge-small chosen [Done, note 0010]
2. `chunk_sets` + `chunks` tables, chunks saved during ingestion [Done, note 0011]
3. Embedder seam with sentence-transformers BAAI/bge-small-en-v1.5 + measure real token
   counts of chunks [Next]
4. Chroma vector store + embedding cache key (file hash + parser version + strategy and
   params + model name); ingest embeds and stores; "ready to query" document status
5. Dense retrieval + `POST /documents/{id}/query` returning top chunks with pages and
   scores (no LLM yet); provisional "weak retrieval" threshold, labelled as a guess
6. LLM answer with page citations, "not found in your documents", injection-safe prompt,
   LLM behind a seam with a fake for tests (decision then: LLM provider)
7. Docker rebuild, end-to-end check, Stage 2 gate, decision notes, docs

## Stage 2 gate
- The owner writes 10 hand-verified questions for paper_01 in the eval format before
  Step 7 (option A, assumed).
- A page citation counts as correct when the evidence page is inside
  page_start..page_end of the cited chunk.

## Stage 2 log
1. **Fixed-size chunker:** `chunking/chunks.py` (`TextChunk`: index, text, page_start,
   page_end) and `chunking/fixed.py` (`chunk_fixed_size`, 200 words, 40 overlap, chunks
   may cross pages). Embedding model chosen: BAAI/bge-small-en-v1.5 (512 tokens), not
   installed yet. `scripts/chunk_pdfs.py` summarises a folder. Note 0010.
   Finding: chunks crossing a page boundary are common (paper_01 16/38, paper_02 17/62,
   paper_03 12/29, paper_04 18/64, paper_05 3/20). paper_05 "short" has 4 pages but
   3236 words (dense IEEE pages); relevant for the Stage 3 "very short" rule.
2. **Chunk storage:** `ChunkSet` (document, strategy, params JSONB, parser_version; unique
   on those four) and `Chunk` (chunk_set, chunk_index, text, pages; unique on set + index),
   both ON DELETE CASCADE. `save_chunk_set()` skips an identical set; parse + chunk + save
   in one transaction with rollback on error. `uq` naming convention now uses
   `column_0_N_name`. Migration 3b2dd0114f3e. `scripts/list_chunk_sets.py`. Real-data check:
   paper_03 -> 29 chunks. Note 0011.

## Stage 1 steps
| Step | What | Status |
| --- | --- | --- |
| 1 | Sample PDFs (5 good, 3 bad) | Done |
| 2 | `POST /documents`: extension, size, `%PDF-` magic bytes | Done (note 0002) |
| 3 | SHA-256 hash, save as `data/uploads/<hash>.pdf` | Done (note 0003) |
| 4 | Postgres, SQLAlchemy, Alembic, duplicate detection, CI | Done (note 0004) |
| 5 | `ParsedBlock` / `ParsedDocument` data shapes | Done (note 0005) |
| 6 | Parser v0 with PyMuPDF: text blocks + 1-based pages | Done (note 0006) |
| 7 | Reading order for two-column papers | Done, no code change needed (note 0007) |
| 8 | Heading detection | **Moved to Stage 3** (rules drafted in note 0008, status Proposed) |
| 9 | Table detection | **Moved to Stage 3** |
| 10 | Reject corrupt, encrypted, scanned PDFs | Done |
| 11 | Background ingest job + `GET /jobs/{id}` | Done (note 0009) |
| 12 | Docker rebuild, end-to-end check, gate, docs | Done |
| - | Chunk table | **Moved to Stage 2 Step 1** (add the table with the chunker that fills it) |
 
## Stage 1 gate
| Gate item | Result |
| --- | --- |
| Parse 5 papers; page numbers correct | Done: `scripts/check_pdfs.py data/samples` parses all 5; pages are 1-based physical pages (tested). |
| Headings match the real PDF | Moved to Stage 3 with heading detection. |
| Reading order on a two-column paper | Done on paper_02 (note 0007). Only one real two-column sample so far. |
| Duplicate upload: same document, no re-processing | Done (tests; duplicate returns existing document and its latest job). |
| Corrupt and scanned PDFs give clear errors, not crashes | Done: `bad/` files rejected with CorruptPDFError / ScannedPDFError; job stores the message. |
| Unit tests for parser and duplicate logic | Done (43 tests). |
| Upload -> background job -> `GET /jobs/{id}` works in Docker | Done in Step 12. |
 
## Stage 1 log
1. **Samples:** `data/samples/` holds paper_01 (single column, "Attention Is All You
   Need"), paper_02 (two column, ResNet), paper_03 (single column, GAN; renamed from
   `paper_03_two_column_b.pdf` after inspection showed one column), paper_04 (table-heavy),
   paper_05 (short, IEEE style); `bad/` holds corrupt, fake, scanned. `data/` is gitignored.
2. **Upload validation:** `routes/documents.py`. 415 wrong extension or not `%PDF-`,
   413 over `ADAPTRAG_MAX_UPLOAD_MB` (20). Note 0002.
3. **Hashing and storage:** `storage.py` with `compute_sha256()`, `upload_path()`,
   `save_upload()`; files saved by hash. Note 0003.
4. **Database:** Postgres 17 in compose, SQLAlchemy 2.0 + psycopg 3, Alembic,
   duplicate detection with IntegrityError race handling, tests on `adaptrag_test`
   (Alembic rebuild per run, TRUNCATE of all `Base` tables after each test), CI with
   Postgres service and `alembic check`. Note 0004.
5. **ParsedBlock:** `parsing/blocks.py`: frozen, slotted dataclasses. `BlockType`
   StrEnum (heading, paragraph, table). Rules in `__post_init__`: page >= 1, text not
   blank, page_count >= 1, no block beyond page_count. 12 tests. Note 0005.
6. **Parser v0:** PyMuPDF 2.0 (`pymupdf>=1.24`). Chose PyMuPDF over pdfplumber, pypdf,
   ML layout tools, and "both libraries" (parser held fixed so the benchmark measures
   routing only). `parsing/pdf.py`: `parse_pdf(path)` reads `page.get_text("blocks")`,
   skips images and blank blocks, `clean_text()` collapses whitespace, page = index + 1,
   `PARSER_VERSION = "0.1.0"`. Every block is PARAGRAPH for now. Tests build their own
   PDFs with PyMuPDF. `scripts/inspect_pdf.py` prints raw blocks (x0, y0, x1) and fonts.
   Note 0006.
7. **Reading order:** paper_02 already reads left column (x 50-286) then right
   (x 309-545). Sorting by y (PyMuPDF `sort=True`) would break it (right column starts at
   y=75, left text at y=183). Kept PyMuPDF's stored order. Note 0007.
8. **Headings (deferred):** inspection found sections 12.0 bold, subsections 10.0 bold
   (= body size), run-in bold labels inside paragraph blocks ("Encoder:"), arXiv stamp
   20.0 not bold, IEEE paper_05 has no bold headings at all. Rules drafted in note 0008
   (Proposed). In Stage 3, compare them with pymupdf4llm before accepting.
10. **Rejecting bad PDFs:** `ParseError` base class with `CorruptPDFError`,
    `EncryptedPDFError`, `ScannedPDFError`; user-safe messages, original cause kept with
    `from exc`. Scanned = fewer than 100 characters per page on average
    (`min_chars_per_page` parameter; small test PDFs pass 0). `scripts/check_pdfs.py`
    parses a whole folder.
11. **Background jobs:** `jobs` table (queued, running, succeeded, failed, error,
    created_at, finished_at; FK to documents ON DELETE CASCADE) and `documents.page_count`.
    `ingest.py` `run_ingest_job(engine, upload_dir, job_id)` is the seam. `POST /documents`
    flushes the document, adds a Job, commits both, schedules the task, returns `job_id`.
    `GET /jobs/{id}` returns status and error (404 if unknown). Note 0009.
12. **Docker and gate:** rebuilt the image with PyMuPDF; uploaded a sample and a bad file
    to the Docker app and checked their jobs. Stage 1 gate above.
## Problems hit and lessons learned
- ruff I001: imports sorted by module path; `python -m ruff check . --fix` sorts them.
- `HTTP/1.1 100 Continue` in curl upload output is normal.
- mypy strict caught a duplicated `data_dir` line in Settings (`no-redef`).
- Typing Python at the `>>>` prompt changes no files; leave with `exit()`.
- "Is the docker daemon running?" / "failed to connect to the docker API" means Docker
  Desktop is not open. `Connection refused` on 5432 in tests: check `docker compose ps` first.
- Postgres reads `POSTGRES_PASSWORD` only when the volume is first created.
- `docker compose down -v` deletes the database volume. Never use it on data you need.
- Alembic `fileConfig` must use `disable_existing_loggers=False`.
- SQLAlchemy prints URLs with `***`; use `render_as_string(hide_password=False)`.
- Read timestamps when diagnosing (a `200` came from the Docker app, not local uvicorn).
- `ERROR at setup` means a fixture broke (e.g. no database); `FAILED` means an assert was
  wrong. With many errors, read the first one; the rest are often knock-on errors.
- Files pasted from chat can lose (or double) the final newline. Fix: VS Code setting
  "Files: Insert Final Newline", and run `python -m ruff format .` locally before checks.
- Pasting several commands on separate lines runs all of them even if one fails. Chain
  checks and git with `&&` so nothing is committed unless everything passes.
- In zsh, `<name>` is input redirection: never type placeholders literally.
- `clear` is a shell command, not Python; inside Python use Ctrl+L.
- A multi-line block at `>>>` needs an empty line to finish before the next command.
- `git add <files>` stages only the named files; read `git status` for leftovers.
- pytest only collects files named `test_*.py` (a file named `test_jobs` was silently skipped).
- PyMuPDF type hints are incomplete: `Document()` is untyped and iterating it is mistyped.
  `doc: Any` plus `# type: ignore[no-untyped-call]` in `parsing/pdf.py` only.
- Alembic's default migration template used old typing; `migrations/script.py.mako`
  updated to `str | Sequence[str] | None` and sorted imports.
- FastAPI `TestClient` runs background tasks before `client.post()` returns, so tests can
  check job results immediately.
- PyMuPDF prints harmless `SwigPyPacked ... has no __module__` deprecation warnings.
- Model code was pasted into `tests/test_models.py` by accident; `git restore` brought
  back the committed test file.
- A migration autogenerated before `models.py` was saved came out empty (only `pass`).
  Save first, then read every autogenerated migration.
- Deleting that empty migration after the test database had applied it broke conftest
  ("Can't locate revision"); fixed by recreating `adaptrag_test`. A migration counts as
  applied once any database ran it.
- Docker `migrate` failed until the image was rebuilt (`docker compose up --build -d`),
  because the image holds its own copy of `migrations/`.
- `git diff` opens a pager: press `q`, or use `git --no-pager diff`.
## Decisions made so far
- FastAPI (rejected Flask, Django). PostgreSQL 17 (rejected SQLite, MySQL, MongoDB).
- SQLAlchemy 2.0 sync, psycopg 3; Alembic migrations with `alembic check` in CI;
  migrations as a separate one-off step; tests on a real Postgres test database.
- Duplicates return the existing document with 200. SHA-256, files named by hash.
- Parser output is our own `ParsedBlock`/`ParsedDocument` (frozen dataclasses), so only
  `parsing/pdf.py` depends on PyMuPDF (note 0005).
- PyMuPDF only, AGPL acceptable for this open-source repo (note 0006).
- Keep PyMuPDF's stored reading order (note 0007).
- Get the whole pipeline working first; headings and tables in Stage 3 (owner's choice).
- Write rules from known techniques and verify with tests plus one run over the sample
  PDFs; inspect real PDFs only when a decision cannot be made without data.
- Unusable PDFs raise `ParseError` subclasses with user-safe messages.
- FastAPI BackgroundTasks now, Redis queue later; separate `jobs` table (note 0009).
- Evaluation questions deferred to Stage 4 (owner's decision).
- Chroma for vectors (decided, not installed yet). Modular routers. `src/` layout;
  Python 3.11; strict mypy.
## Deferred / known limits
- Headings and tables: Stage 3. Compare note 0008 rules with pymupdf4llm first.
  IEEE-style headings (paper_05) not covered by the drafted rules.
- Only one real two-column sample (paper_02). Add an ACL or IEEE two-column paper.
- Page furniture (permission notices, footnotes, page numbers, arXiv stamp, figure
  labels) comes through as paragraphs. Hyphenated line breaks become "trans- former".
- Scanned threshold (100 chars/page) is a rule of thumb, not measured.
- Errors while reading individual pages (after opening) are not turned into ParseError.
- Jobs: a restart mid-job leaves it "running" (Stage 9: mark stale jobs failed on
  startup); no retries; jobs run inside the API process.
- Uploaded files in Docker are not on a volume yet (lost when the container is removed).
- `database_url` has a code default; make config required with a startup check (Stage 9).
- The race-condition branch in `POST /documents` has no automated test.
- Whole upload read into memory (fine at 20 MB).
- `StarletteDeprecationWarning` about httpx in tests: look into separately.
- Dev database documents 1 and 2 were uploaded before jobs existed (no job, status uploaded).
- Stage 3 Strategy interface: a strategy must receive its dependencies (an embedder, an
  LLM client) when it is created, and expose its params for the cache key.
- Stage 3: LLM-guided chunkers may only return boundary positions, never rewritten text.
## Future stretch ("Stage 10", after Stage 6, tuned on dev documents only)
- Semantic chunking (owner's own binary-search boundary method).
- LLM-guided chunking.
- Each one is one strategy file + one registry line.
## Explain-aloud questions (Stage 1, answer without notes)
1. What's the difference between `/health` and `/ready`?
2. Why does the rest of the app read `ParsedBlock` instead of PyMuPDF's own objects?
3. Why can't evaluation mark the right answer by chunk id?
4. Why do the parser tests build their own PDFs instead of using `data/samples/`?
5. Why would sorting blocks top to bottom break a two-column page?
6. Why PyMuPDF and not two PDF libraries at once?
7. Why `BackgroundTasks` now, and what would a Redis queue add?
8. Why `session.flush()` before creating the job, and why one final `commit()`?
9. Why does `run_ingest_job` catch `Exception` after catching `ParseError`?
10. Why does `test_page_must_be_one_or_more` test 0 and not only -1?
## Explain-aloud questions (Stage 2, answer without notes)
1. Why did deleting the empty migration break the test database but not the dev database?
2. Why does the chunker count words instead of the embedding model's tokens?
3. Why does `run_ingest_job` call `session.rollback()` before recording a failure, and
   what would be saved without it?

## Open items / notes for the next chat
- Review and merge `stage2-step2-chunk-storage`, then start Stage 2 Step 3.
- Documents ingested before Step 2 have no chunks (duplicates are not re-processed).
- Update this file at the end of the chat.
