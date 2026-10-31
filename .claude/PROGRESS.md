# AdaptRAG Progress
 
Update this file at the end of every chat. Newest stage at the top of the log.
 
## Current status
- **Current stage:** **Stage 2 DONE** (Steps 1-7; gate PASS 10/10, note 0016).
  **Next: Stage 3** (strategy interface, profiler, router, headings and tables).
- **Last updated:** 2026-11-01
- **Repo:** public GitHub repo `adaptrag`, branch `main`, CI green (pgvector Postgres
  service in CI from Step 4).
- **Tests:** Step 7: 163 passed, 7 deselected (normal run). Step 6: 136 passed,
  7 deselected; 7 passed with `-m slow` (5 embedder, 2 real OpenAI). Step 5: 112 passed,
  5 deselected. Step 4: 92 passed, 5 deselected; 5 passed with `-m slow`.
  Decision notes 0001-0016 (0008 is "Proposed").
- **Branch:** Step 7 on `stage2-step7-gate` (owner commits after checks).
- **Working mode (from Stage 2 Step 3):** Claude Code only creates and edits files; the
  owner runs every command (tests, linters, scripts, docker, git) and fills in
  `TO FILL from owner's run` placeholders. The owner reviews the diff in chat before merging.
## Stage checklist
| Stage | Name | Status |
| --- | --- | --- |
| 0 | Skeleton | Done |
| 1 | Upload, parsing, job status | Done (headings and tables moved to Stage 3) |
| 2 | Baseline RAG | Done (gate 10/10, note 0016) |
| 3 | Strategy interface, profiler, router (+ headings, tables) | Not started |
| 4 | Mini evaluation | Not started |
| 5 | Hybrid, rerank, parent retrieval | Not started |
| 6 | Full evaluation | Not started |
| 7 | Notebooks and multi-document search | Not started |
| 8 | Frontend | Not started |
| 9 | Hardening, deploy, README | Not started |
 
Parallel task: evaluation question set. **Deferred to Stage 4 (owner's decision).**
`eval/questions/paper_01.json` holds the 10 Stage 2 gate questions (evidence filled and
verified by the owner; used for the Stage 2 gate, note 0016).
`eval/question_format_example.json` shows the format (made-up examples, not real questions). At Stage 4: LLM drafts questions in that
format, owner verifies each against the PDF and sets `"verified": true`.
 
## Stage 2 plan (approved)
1. `TextChunk` + fixed-size chunker, 200/40 words, bge-small chosen [Done, note 0010]
2. `chunk_sets` + `chunks` tables, chunks saved during ingestion [Done, note 0011]
3. Embedder seam with sentence-transformers BAAI/bge-small-en-v1.5 + measure real token
   counts of chunks [Done, note 0012]
4. Vector storage with pgvector in Postgres (changed from Chroma, owner's decision) +
   embedding cache key (file hash + parser version + strategy and params + model name);
   ingest embeds and stores; "ready" document status; log a warning and count chunks
   over max_tokens per document [Done, note 0013]
5. Dense retrieval (exact search) + `POST /documents/{id}/query` returning top chunks with
   pages and scores (no LLM yet). Changed: no provisional threshold; the "weak retrieval"
   threshold moves to Step 6, chosen from measured scores [Done, note 0014]
6. LLM answer with page citations, "not found in your documents" (threshold chosen from
   measured scores), injection-safe prompt, LLM behind a seam with a fake for tests
   (owner chose OpenAI gpt-4o-mini, threshold 0.55) [Done, note 0015]
7. Docker rebuild, end-to-end check, Stage 2 gate, decision notes, docs; measure the
   first-upload model load time after a restart (note 0013 known limit) [Done, note 0016]

## Stage 2 gate
| Gate item | Result |
| --- | --- |
| >= 8/10 hand-verified questions cite the right page | Done: 10/10 cited_ok, 10/10 evidence in top-5, PASS (note 0016) |
| Off-topic question answered "not found" | Done |
| Prompt injection not followed | Done: PWNED slow test |
| Embedding cache: no re-embedding of stored work | Done: cache tests |
| Upload -> ingest -> answer works | Done: integration test |
| Full stack in Docker | Done: Docker end-to-end check |

- A page citation counts as correct when the evidence page is inside
  page_start..page_end of the cited chunk.
- Questions: `eval/questions/paper_01.json` (q01-q10; q10 has two evidence entries),
  evidence filled and `"verified": true` set by the owner.
- Check: `python scripts/gate_check.py eval/questions/paper_01.json --document-id ID --k 5`.
  Rules in `src/adaptrag/evaluation/gate.py` (tests in `tests/test_gate.py`):
  cited_ok = found AND a citation's page range holds an evidence page (the gate counts
  this); retrieved_ok = an evidence text is inside a top-k source after casefold and
  whitespace collapsing (reported, not part of the gate). Unverified entries are
  skipped; verified entries with a null page or "TO FILL" text are refused. PASS needs
  at least 10 scored questions and >= 8/10 cited_ok.
- Gate result (owner's run; gpt-4o-mini, k=5, min_score 0.55, document 1 = paper_01):
  10/10 cited the right page, 10/10 evidence in top-5, PASS. q10 cited pages 6 and 9.
- Latency: answer mean 1.79 s, max 2.80 s (`gate_check.py`, local, model load
  excluded). First query after an API restart 3.23 s vs 0.035 s afterwards (Docker,
  `/query`), i.e. the model load costs about 3.2 s once per process.
  Tokens per answer: 1567-1999 input, 14-129 output. Money cost: owner's OpenAI usage page.

## Stage 2 failure-analysis notes
- Rank 1 is not always the answering chunk (GAN question: rank 3).
- Retrieval scores measure topic similarity, not answer presence: in-domain
  unanswerable questions score 0.61-0.70, close to answerable ones.
- Top-k compensated a retrieval miss: for the GAN question the answering chunk was
  rank 3, and the LLM cited source 3.
- Parser text quality: words are glued at line breaks ("shownin", "Thedecoder",
  "Anattention", "withkernel", "transformermodel"), and page-footer numbers and figure
  captions stay inside chunk text. This matters for BM25 (Stage 5: glued words are
  different tokens) and for evidence matching. Stage 3 parser work.
- Gate caveats (note 0016): one clean dev paper; 10 mostly single-fact questions drafted
  by an LLM; evidence phrases copied from our own parsed chunks, so retrieved_ok is easy
  here; lenient rule (any citation's page range containing an evidence page counts,
  extra citations not penalised; q02 cited three chunks); one run. Stage 4 must build
  questions and evidence more independently and measure citation precision.

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
3. **Embedder:** `embedding/base.py` (`Embedder` protocol), `embedding/sentence_transformer.py`
   (`SentenceTransformerEmbedder`, cached `get_embedder()`; only module importing
   sentence_transformers/torch), `embedding/fake.py` (`FakeEmbedder`, hash-based, dim 8).
   New dependency `sentence-transformers>=6.2`. Device cpu; CPU-only torch in Docker and CI.
   Settings `embedding_model`, `embedding_device`, `embedding_query_instruction`. Queries get
   the bge instruction prefix, chunks don't; vectors normalised. Real-model tests marked
   `slow` (excluded by default, not in CI). `scripts/token_stats.py`. Note 0012.
   Token measurement (fixed-size 200/40, data/samples):
   `OVERALL: 213 chunks, tokens min 73 / median 313 / max 654, 3 over 512`
   (paper_03: 1, paper_05: 2). Owner chose option A: keep 200/40, record the overflow as
   a known limit, Step 4 ingest warns and counts chunks over max_tokens.
   `--show-over` flag lists the chunks over the limit.
   api image: 2.33 GB on disk, 523 MB content size; `torch 2.15.0+cpu`, no CUDA.
   Part A: `.claude/claude.md` renamed to `.claude/CLAUDE.md`; CONTEXT_HANDOFF.md
   references removed.
4. **Vector storage (pgvector):** owner chose pgvector in Postgres instead of Chroma
   (chunks and vectors in one transaction, cascade delete, Alembic schema, one tool
   fewer). New dependency `pgvector`; Postgres image `pgvector/pgvector:pg17` in compose
   and CI. `ChunkEmbedding` (`chunk_embeddings`: chunk_id FK CASCADE, model_name,
   vector(384); PK chunk_id + model_name). Hand-written migration 5c8e1a7d4f2b (creates
   the `vector` extension); `migrations/env.py` registers "vector" for reflection.
   `vectorstore/pgvector.py`: `embeddings_exist`, `save_embeddings` (length and
   dimension checks). Structural cache key: document -> chunk set -> model name.
   `save_chunk_set` returns `(chunk_set_id, created)`. `run_ingest_job` takes an
   `embedder`, warns about chunks over max_tokens, embeds and saves in the same
   transaction; success status is now "ready". `embedding/provider.py`
   `get_app_embedder` (lazy torch import); tests override it with
   `FakeEmbedder(dimension=384)` (FakeEmbedder now supports any dimension).
   Docker: volumes `app-data` (/app/data) and `hf-cache` (/app/.cache/huggingface,
   `HF_HOME`). `scripts/reingest.py`; `list_chunk_sets.py` prints embedding counts.
   Note 0013. Tests: 92 passed, 5 deselected (normal run); 5 passed (`-m slow`).
   Real-data check: `scripts/reingest.py 5` (paper_03): job succeeded, 29 chunks,
   29 embeddings, WARNING "1 of 29 chunks exceed 512 tokens" (matches the Step 3
   measurement: paper_03 has 1 chunk over 512). Docker: `hf-cache` and `app-data`
   volumes created, `/ready` returned `{"status":"ready","database":"ok"}`. CI green on
   the pgvector image.
5. **Dense retrieval:** `retrieval/results.py` (`ScoredChunk`, frozen and slotted),
   `vectorstore/pgvector.py` `search()` (exact search, pgvector cosine distance, filtered
   by chunk set and model name, `ORDER BY distance, chunk_index LIMIT k`,
   score = 1 - distance; ValueError for a wrong-length vector or k < 1; all vector SQL
   stays here). `retrieval/dense.py`: `current_chunk_set_id()` (fixed_size,
   `FIXED_SIZE_PARAMS`, `PARSER_VERSION`, the same constants ingest uses),
   `retrieve_dense()` (embed_query, then search), shared limits and messages.
   `routes/query.py`: `POST /documents/{id}/query` (422 invalid body, 404 unknown,
   409 not ready, 409 needs re-ingestion, 200 top-k best first). No LLM, no threshold
   (owner's decision: the threshold is chosen in Step 6 from measured scores).
   Scripts: `query.py`, `list_documents.py`, `reingest.py --file PATH` (SHA-256 checked
   with storage.py, restores a missing upload). Note 0014.
   Tests: 112 passed, 5 deselected (normal run).
   Real-data check (`scripts/query.py`, bge-small, k=3, top score):
   - answerable: paper_01 "attention heads" 0.7589 (p.5 multi-head section, rank 1);
     paper_01 "optimizer" 0.7357 (Adam chunk, rank 1); paper_03 "two models trained at
     the same time" 0.7159 (the abstract chunk that answers it is rank 3 at 0.6925).
   - off-topic: "capital of France" 0.4440; "sourdough bread" 0.4465; "best football
     club" 0.4167.
   - in-domain but not answerable: paper_01 "ImageNet accuracy" 0.6987; paper_03 "FID
     score" 0.6126.
   Docker endpoint returned the same top score as local (0.71585 vs 0.7159): CPU-only
   gives the same numbers on the Mac and in Docker. Documents 1, 2 and 3 were
   re-ingested with `reingest.py --file` (paper_01: 38 chunks, 38 embeddings).
6. **Answer generation:** new dependency `openai` (official client: retries, timeouts,
   typed responses). Settings `openai_api_key` (SecretStr, `.env` only), `llm_model`
   (gpt-4o-mini, owner's choice for the lowest cost), `llm_temperature` (0.0; None = not
   sent), `llm_timeout_seconds` (30), `llm_max_output_tokens` (500),
   `retrieval_min_score` (0.55, provisional, from the 8 measured dev questions).
   `llm/`: `LLM` protocol + `LLMError` (base.py), `OpenAILLM` (Responses API, only module
   importing openai, max_retries=2, logs input/output tokens at INFO), `FakeLLM`,
   `get_app_llm` (503 without a key). `answering/prompt.py` (rules: sources only, cite
   [N], exactly NOT_FOUND, `<source>` text is untrusted; `<source`/`</source` in chunk
   text escaped). `answering/answer.py` `answer_question` (two-layer not-found, citations
   de-duplicated in order and mapped to chunk pages by our code; no valid citation ->
   WARNING). `POST /documents/{id}/answer` (502 on LLMError, 503 without a key); `/query`
   unchanged. `scripts/answer.py`. Note 0015.
   Tests: 136 passed, 7 deselected (normal run); 7 passed with `-m slow` (5 embedder,
   2 real OpenAI: answerable question cited correctly, PWNED injection not followed).
   Real-data check (`scripts/answer.py`, gpt-4o-mini, k=5, min_score 0.55; tokens are
   input / output):
   - paper_01 "Which optimizer was used for training?": found, "Adam ... [1][2]",
     citations pp. 7-8 and p. 7; 1921 / 43 tokens.
   - paper_03 "Which two models are trained at the same time?": found, "generative model
     G and discriminative model D [3][4]", citations p. 1 and pp. 2-3; 1651 / 40 tokens.
   - paper_01 "What accuracy does the model reach on ImageNet?": not found by the LLM
     (NOT_FOUND); 1720 / 3 tokens.
   - paper_01 "What is the capital of France?": not found by retrieval (top 0.4440 <
     0.55); no LLM call.
   Docker `POST /documents/1/answer` returned the same answer and citations.
   Both not-found layers worked on real data. Money cost and latency: Step 7.
7. **Docker rebuild, end-to-end check, Stage 2 gate:**
   `eval/questions/paper_01.json` (10 questions, evidence filled and verified by the owner).
   `src/adaptrag/evaluation/gate.py`: pure scoring rules and loader (`normalise`,
   `cited_ok`, `retrieved_ok`, `gate_passed`, `load_questions`). `scripts/gate_check.py`
   runs the same checks and pipeline as POST /answer for every verified question, times
   each `answer_question` call with `time.perf_counter()` (model load excluded), prints
   one line per question and the totals. `tests/test_gate.py` (unit).
   Tests: 163 passed, 7 deselected (normal run).
   Gate result: 10/10 cited the right page, 10/10 evidence in top-5, PASS (gpt-4o-mini,
   k=5, min_score 0.55, document 1 = paper_01); q10 cited pages 6 and 9.
   Latency: answer mean 1.79 s, max 2.80 s; first query after an API restart 3.23 s vs
   0.035 s afterwards (Docker, model load about 3.2 s once). Tokens per answer 1567-1999
   input, 14-129 output. Note 0016. **Stage 2 done.**

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
- A deselected test file is still imported during collection. A top-level import of the
  real embedder made the normal run import torch, so it moved inside a function in
  `tests/test_embedder_real.py`.
- `.env` values with a trailing space must be quoted (`embedding_query_instruction`).
- Appending to a file without a final newline glued two lines together (.env.example).
  Check the last line before appending.
- curl right after `docker compose up -d` can fail because uvicorn is not listening yet;
  re-run it (use `;` instead of `&&` for checks that may fail).
- The local app and the Docker app share one database but not one upload folder
  (data/uploads vs the app-data volume), so a document may exist without its file locally.
  `scripts/list_documents.py` shows which; `scripts/reingest.py ID --file PATH` restores it.
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
- Embedder seam with bge-small on CPU, CPU-only torch in Docker/CI, query instruction on
  queries only, normalised vectors, model loaded once per process; real-model tests are
  `slow` and not run in CI (note 0012).
- pgvector in Postgres for vectors instead of Chroma (owner's decision, note 0013):
  exact search, no ANN index, fixed 384 dimensions, structural cache key, embedder via
  a FastAPI dependency, "ready" document status, Docker volumes for uploads and the
  model cache. Modular routers. `src/` layout; Python 3.11; strict mypy.
- Dense retrieval (note 0014): `POST /documents/{id}/query` with a JSON body, current
  chunk set only (409 asks for re-ingestion otherwise), cosine distance with
  score = 1 - distance, exact search ordered by distance then chunk_index, vector SQL only
  in `vectorstore/pgvector.py`, no threshold until Step 6 measures scores.
- Answer generation (note 0015): OpenAI gpt-4o-mini behind an `LLM` seam (owner's
  choice for the lowest cost; rejected gpt-5.4-mini for its price at this stage),
  two-layer not-found (score < 0.55, then the model's NOT_FOUND), numbered sources
  mapped to pages by our code, untrusted-source prompt with tag escaping, `/answer`
  separate from `/query`, 503 without a key, 502 on LLM failure.
- Stage 2 gate (note 0016): cited_ok scored, retrieved_ok reported only, >= 10 scored
  questions and >= 80% cited_ok, only verified entries, an LLM failure stops the run.
## Deferred / known limits
- Headings and tables: Stage 3. Compare note 0008 rules with pymupdf4llm first.
  IEEE-style headings (paper_05) not covered by the drafted rules. pymupdf4llm 2.0 is
  already installed as a dependency of pymupdf 2.0, so the comparison needs no new
  dependency.
- Only one real two-column sample (paper_02). Add an ACL or IEEE two-column paper.
- Page furniture (permission notices, footnotes, page numbers, arXiv stamp, figure
  labels) comes through as paragraphs. Hyphenated line breaks become "trans- former".
- Scanned threshold (100 chars/page) is a rule of thumb, not measured.
- Errors while reading individual pages (after opening) are not turned into ParseError.
- Jobs: a restart mid-job leaves it "running" (Stage 9: mark stale jobs failed on
  startup); no retries; jobs run inside the API process.
- `database_url` has a code default; make config required with a startup check (Stage 9).
- The race-condition branch in `POST /documents` has no automated test.
- Whole upload read into memory (fine at 20 MB).
- 3 of 213 sample chunks exceed 512 tokens and are truncated by the model (note 0012).
- Dev database documents 1 and 2 were uploaded before jobs existed (no job, status uploaded).
- A model with a different number of dimensions needs a migration (vector(384), note 0013).
- The first upload after a restart loads the model during the request (measured about
  3.2 s: first `/query` 3.23 s vs 0.035 s afterwards; the dependency runs before the
  endpoint, even for rejected or duplicate uploads).
- The model downloads from Hugging Face on first use into the `hf-cache` volume (first
  upload in a fresh Docker setup needs network access).
- Exact search reads every vector: fine only at small scale (a few hundred chunks).
- Query searches only the fixed-size chunk set until the router exists (Stage 3); one
  document per query until notebooks (Stage 7).
- The first query after a restart also loads the model (dependency runs first, even for
  404/409 answers).
- `FakeEmbedder` scores are meaningless: query tests check shape and order only; ranking
  quality is measured with the real model in Stage 4.
- Documents ingested before Step 4 have no embeddings until `scripts/reingest.py` runs.
- Stage 3 Strategy interface: a strategy must receive its dependencies (an embedder, an
  LLM client) when it is created, and expose its params for the cache key.
- Stage 3: LLM-guided chunkers may only return boundary positions, never rewritten text.
- Not-found threshold 0.55 comes from the 8 measured dev questions only; recalibrate in
  Stage 4. Answers may vary between runs. Gate run: 1567-1999 input and 14-129 output
  tokens per answer, mean 1.79 s and max 2.80 s per answer; money cost on the owner's
  OpenAI usage page.
- gpt-4o-mini may follow the citation and NOT_FOUND rules less reliably than newer
  models (watch the slow test and the gate questions); availability and price must be
  checked by the owner on OpenAI's site.
- Prompt-injection defence reduces but does not eliminate the risk.
- The app does not configure logging yet, so INFO lines (token usage) may not show under
  uvicorn; `scripts/answer.py` configures it. Stage 9 hardening.
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
4. Why does the query get an instruction prefix but the chunks don't?
5. Why pgvector instead of Chroma once we chose Postgres?
6. Why exact search instead of an HNSW index for the benchmark?
7. Why cosine distance, and why score = 1 - distance?
8. Why choose the not-found threshold from measured scores instead of a default?
9. Why does the not-found check need two layers?
10. Why does our code map citations to pages instead of the model?
11. Why is /answer separate from /query?
12. Why does the gate score cited_ok but only report retrieved_ok?
13. How were the gate questions built, and why might they be biased toward this system?

## Open items / notes for the next chat
- Stage 2 is done. Owner: commit Step 7 on `stage2-step7-gate`, review, merge.
- Next: Stage 3 (strategy interface, profiler, router, headings and tables). Start with
  a Stage 3 plan for approval; decide the dev/test document split before router rules.
  Parser fixes for glued words and page furniture belong here (failure-analysis notes).
- Model in Docker: decided, `hf-cache` volume (note 0013); baking it in is a Stage 9
  option.
- Chunks over 512 tokens: decided, option A (note 0012); ingest now warns per document.
- Documents ingested before Step 2/4 lack chunks or embeddings; use `scripts/reingest.py`.
- Update this file at the end of the chat.
