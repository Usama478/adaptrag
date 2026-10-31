# CLAUDE.md: AdaptRAG project guide
 
For any coding agent or assistant working in this repo. Read this first.
Also read `CONTEXT_HANDOFF.md` (how to teach the owner) and `PROGRESS.md` (current state).
 
## What this project is
AdaptRAG is a document-aware RAG notebook built with FastAPI. It profiles each
uploaded document (research-paper PDFs first), picks a chunking and retrieval
strategy using explicit rules, and shows with a benchmark that routing beats any
single fixed strategy. It is a portfolio and learning project, built to
production standards; the owner must be able to explain and defend every module.
 
**Headline claim (do not write until the evaluation produces it):**
"Router improves Recall@5 by X points over the best single strategy on N documents."
Never invent or estimate numbers. Report honest and mixed results. Do not claim the
app "handles many users" unless a load test measured it.
 
## Owner and working agreement
- The owner is learning. Prefer small, readable changes with explanations over
  clever code. Explain what changed and why.
- Work stage by stage, one step at a time. Don't build ahead of the current step.
- Do not add dependencies or tools without a stated reason.
- Each module needs a short decision note in `docs/decisions/` (drafted by the
  assistant, reviewed by the owner).
- Write rules from known techniques and verify them with tests plus one run over
  the sample PDFs. Inspect real PDFs only when a decision can't be made without data.
## Design principle: build simple, leave seams
Decide per component by the cost of changing it later.
- Do properly now (cheap now, expensive later): PostgreSQL, Alembic migrations.
- Simple for the MVP, behind a seam for later upgrade:
  - Background jobs: FastAPI BackgroundTasks now; a Redis-backed queue later.
    Job logic lives in the plain function `ingest.run_ingest_job(engine, upload_dir, job_id)`.
  - File storage: local disk now (`data/uploads/`); S3-style object storage later.
    All file paths go through `storage.py` (`upload_path()`, `save_upload()`).
  - PDF library: only `parsing/pdf.py` imports PyMuPDF; everything else reads
    `ParsedBlock` / `ParsedDocument`.
  - Vector store: Chroma in-process now; Chroma server later (client config change).
## Tech stack
- Python 3.11, FastAPI, Uvicorn, pydantic-settings, python-multipart
- PostgreSQL 17 (Docker), SQLAlchemy 2.0 (sync, typed `Mapped` style),
  psycopg 3 (`psycopg[binary]`), Alembic (migrations)
- PyMuPDF (`pymupdf>=1.24`, currently 2.0) for PDF parsing (AGPL; fine for this public repo)
- Tests: pytest (TestClient), against a real Postgres test database
- Lint/format: ruff. Types: mypy (strict, `files = src`)
- Container: Docker + docker compose. CI: GitHub Actions with a Postgres service
- Planned later: sentence-transformers (embeddings), Chroma, BM25, cross-encoder
  reranker, an LLM for answers; possibly pymupdf4llm (Stage 3, compare first)
## Repo layout
```
adaptrag/
├── .github/workflows/ci.yml      # lint, format, mypy, migrations, alembic check, tests
├── alembic.ini                   # NO database URL here (env.py reads settings)
├── migrations/
│   ├── env.py                    # URL from settings, or config.attributes (tests)
│   ├── script.py.mako            # migration template (modern typing, sorted imports)
│   └── versions/                 # 2eded5ee7afc documents, 1b7cc351503c jobs + page_count,
│                                 # 3b2dd0114f3e chunk_sets + chunks
├── docker/postgres/init/01-create-test-db.sql   # creates adaptrag_test on first init
├── docs/decisions/               # 0001 stack, 0002 upload validation, 0003 hashing/storage,
│                                 # 0004 Postgres/SQLAlchemy/Alembic, 0005 parsed block shape,
│                                 # 0006 PDF library, 0007 reading order,
│                                 # 0008 heading rules (Proposed), 0009 background jobs,
│                                 # 0010 fixed-size chunking, 0011 chunk storage
├── eval/
│   ├── question_format_example.json   # format only (made-up examples)
│   └── questions/                # real questions, one JSON per paper (Stage 4)
├── scripts/
│   ├── inspect_pdf.py            # raw PyMuPDF blocks (x0, y0, x1) and fonts for one page
│   ├── check_pdfs.py             # parse every PDF in a folder, print OK / REJECTED
│   ├── chunk_pdfs.py             # chunk every PDF in a folder (fixed-size), print a summary
│   └── list_chunk_sets.py        # one line per stored chunk set with its chunk count
├── src/adaptrag/
│   ├── __init__.py               # __version__
│   ├── config.py                 # Settings (env prefix ADAPTRAG_), get_settings()
│   ├── db.py                     # get_engine(url) (cached), get_session() dependency
│   ├── models.py                 # Base (naming convention), Document, Job, ChunkSet, Chunk
│   ├── storage.py                # compute_sha256(), upload_path(), save_upload()
│   ├── ingest.py                 # run_ingest_job(): parse, chunk, save_chunk_set(), job outcome
│   ├── chunking/
│   │   ├── chunks.py             # TextChunk (in-memory chunk shape)
│   │   └── fixed.py              # chunk_fixed_size() (control strategy, 200/40 words)
│   ├── parsing/
│   │   ├── blocks.py             # BlockType, ParsedBlock, ParsedDocument
│   │   └── pdf.py                # parse_pdf(), clean_text(), ParseError family, PARSER_VERSION
│   ├── main.py                   # create_app() factory; includes routers
│   └── routes/
│       ├── health.py             # GET /health (liveness)
│       ├── ready.py              # GET /ready (readiness: database reachable)
│       ├── documents.py          # POST /documents
│       └── jobs.py               # GET /jobs/{id}
├── tests/
│   ├── conftest.py               # test DB URL, migrated engine, TRUNCATE all Base tables, make_client
│   ├── test_documents.py  test_health.py  test_jobs.py  test_models.py
│   ├── test_parsed_blocks.py  test_pdf_parser.py  test_ready.py  test_storage.py
│   ├── test_text_chunk.py  test_fixed_chunker.py  test_chunk_storage.py
├── .dockerignore  .env.example  .gitignore
├── CLAUDE.md  CONTEXT_HANDOFF.md  PROGRESS.md
├── docker-compose.yml            # api, migrate (one-off), db (Postgres + volume)
├── Dockerfile  pyproject.toml  README.md
```
`.env`, `data/` (sample PDFs, uploads) are gitignored.
 
## Commands (always use `python -m`, never the bare tool)
```bash
source .venv/bin/activate                          # new terminal: activate first
pip install -e ".[dev]"                            # install
docker compose up -d db                            # start Postgres only (Docker Desktop must be open)
python -m alembic upgrade head                     # apply migrations to the dev database
python -m uvicorn adaptrag.main:app --reload       # run the app locally
docker compose up --build -d                       # full stack in the background: db -> migrate -> api
docker compose stop api                            # free port 8000 for local uvicorn
docker compose down                                # stop (NEVER add -v unless you mean to delete data)
python scripts/check_pdfs.py data/samples          # parse every sample PDF
python scripts/inspect_pdf.py <pdf> <page>         # raw blocks and fonts for one page
python scripts/chunk_pdfs.py data/samples          # chunk every sample PDF, print a summary
python scripts/list_chunk_sets.py                  # stored chunk sets and chunk counts
```
**The routine before every commit** (stops at the first failure; nothing is pushed
unless everything passes):
```bash
python -m ruff format . && python -m ruff check . --fix && python -m mypy && python -m alembic check && python -m pytest && git add -A && git commit -m "message" && git push
```
Locally `ruff format .` fixes formatting; CI only checks (`ruff format --check .`).
 
Migrations:
```bash
python -m alembic revision --autogenerate -m "describe the change"   # draft a migration
python -m alembic current                          # which revision this database is at
python -m alembic check                            # fails if models have unmigrated changes
python -m alembic downgrade -1                     # undo the last migration
```
Why `python -m`: pyenv and conda also exist on this machine, and a bare `uvicorn`
once resolved to the wrong Python.
 
## Conventions
- **Routers:** one file per feature in `src/adaptrag/routes/`, each exposing
  `router = APIRouter()`. Import and register it in `create_app()`. Keep `main.py` as wiring only.
- **Settings:** all config via env vars with prefix `ADAPTRAG_`, read through
  `get_settings()`. Never hardcode secrets. Add every new variable to `.env.example`.
- **Database URL:** one setting, `ADAPTRAG_DATABASE_URL`. Not in `alembic.ini`.
  Never log the URL (it holds the password).
- **Endpoints that use the database are plain `def`, not `async def`** (sync SQLAlchemy).
- **Background tasks open their own session** from `get_engine(settings.database_url)`;
  the request session is closed when the response is sent.
- **Migrations:**
  - Every model change gets a migration; `alembic check` in CI enforces it.
  - Always read an autogenerated migration before applying it (renames show up as
    drop + add and would lose data).
  - Never edit a migration that has been applied and pushed; add a new one.
  - Keep `downgrade()` working. Naming convention lives on `Base.metadata`.
  - Save `models.py` before autogenerating, then read every autogenerated migration.
    An empty migration (only `pass`) means the model changes were not seen.
  - A migration counts as applied once ANY database ran it, including `adaptrag_test`
    (conftest migrates it on every test run). Don't delete it; add a new one.
  - After adding a migration, rebuild Docker with `docker compose up --build -d`
    (the migrate service runs the migrations baked into the image).
- **Parsing:**
  - Only `parsing/pdf.py` imports PyMuPDF. PyMuPDF's types are incomplete, so the
    document is typed `Any` there, with `# type: ignore[no-untyped-call]` on `open()`.
  - `ParsedBlock.page` is the 1-based physical page (PyMuPDF index + 1).
  - Unusable PDFs raise a `ParseError` subclass; its message is shown to users.
  - Bump `PARSER_VERSION` whenever parser output changes (it is part of the cache key).
- **Tests:** add tests with each feature. Database tests use the `db_engine` or
  `make_client` fixtures (test database, TRUNCATE of every `Base` table after each
  test). Pure unit tests must not need the database. Parser tests build their own PDFs
  with PyMuPDF; never depend on `data/samples/` (gitignored, copyrighted).
  Every new checker or test gets a "make it fail on purpose" check.
- **Typing:** every function typed; mypy strict must pass.
- **Style:** ruff rules E, F, I, B, UP; line length 100. Files end with exactly one newline.
- **Secrets:** `.env` is gitignored and dockerignored. CI uses a throwaway password.
- **Commits:** small and meaningful. CI must stay green on `main`; experiment on branches.
## Rules for coding agents
- One step per task. Don't build ahead.
- Work on a branch named `stageN-stepM-short-name`; commit on that branch.
- Never push, merge or touch `main` unless told to.
- Never run `docker compose down -v`; never drop or recreate the `adaptrag` database.
- Never edit a pushed migration; never force-push.
- Never add a dependency without a stated reason.
- Never invent benchmark numbers.
- Always `python -m <tool>`.
- Run the full check chain before committing:
  `python -m ruff format . && python -m ruff check . --fix && python -m mypy && python -m alembic check && python -m pytest`
- Stop and ask when a decision is not covered by the instructions.
- End every task with this report (exact headings):
  1. Files changed (path + one line each)
  2. Check chain output: the last line of each tool, and the pytest count line
  3. Fail-on-purpose: failing test name and assertion line, then confirmation it was restored
  4. Real-data check: commands run and their output
  5. `git log --oneline -3` and `git --no-pager diff main --stat`
  6. Anything you were unsure about or did differently from the instructions
## Architecture (planned request flow)
1. Upload: validate type and size, hash file, skip duplicates (done)
2. Parse: blocks with page, heading path, block type (done for text and pages;
   headings and tables in Stage 3)
3. Profile: small record of document signals
4. Route: rules map profile to a strategy name plus a reason string
5. Chunk and embed: store embeddings with metadata (doc id, page, section, strategy)
6. Query: each document searched with its own strategy, results merged
7. Rerank: cross-encoder re-scores all candidates on one scale
8. Generate: LLM answers only from retrieved chunks, with page citations
Ingestion runs as a background job with a status endpoint (done).
 
## API
Done:
- `GET /health` (liveness), `GET /ready` (readiness, 503 if db unreachable)
- `POST /documents`: 415 wrong type, 413 too large, 202 new (with `job_id`),
  200 duplicate (existing document and its latest `job_id`, or null)
- `GET /jobs/{id}`: `{job_id, document_id, status, error}`; 404 if unknown
Planned: GET /documents/{id}, DELETE /documents/{id}, POST /notebooks,
POST /notebooks/{id}/query, POST /query/compare.
## Data model
Done:
- Document: id, file_hash (unique), filename, size_bytes, status
  (uploaded / parsed / failed), page_count (nullable), created_at
- Job: id, document_id (FK, ON DELETE CASCADE), status (queued / running /
  succeeded / failed), error (nullable), created_at, finished_at (nullable)
- ChunkSet: id, document_id (FK, ON DELETE CASCADE), strategy, params (JSONB),
  parser_version, created_at; unique on (document_id, strategy, params, parser_version)
- Chunk: id, chunk_set_id (FK, ON DELETE CASCADE), chunk_index, text, page_start,
  page_end; unique on (chunk_set_id, chunk_index). Postgres holds chunk text; Chroma
  will hold vectors keyed by chunk id (note 0011)
Add columns only when they can be filled: title (parser), profile, strategy,
reason (router). Planned: Notebook, QueryLog.
## Strategy interface (Stage 3)
Every strategy implements `chunk(parsed_doc) -> list[Chunk]` and
`retrieve(query, doc_id, k) -> list[ScoredChunk]`. A registry maps names to classes.
Adding a strategy = one new file + one registry line. Starting strategies:
fixed-size (control), recursive with overlap, section-aware with parent retrieval.
 
## Rules the code must respect
- Treat document text as untrusted; the system prompt tells the model to ignore
  instructions found inside documents (prompt injection).
- If retrieval is empty or weak, answer "not found in your documents"; don't guess.
- Reject scanned PDFs (no extractable text), encrypted and corrupt files with clear errors.
- Embedding cache key = file hash + parser version + strategy name and params +
  embedding model name. Changing any of these must trigger re-embedding.
- "Delete means delete": remove file, chunks, vectors, and jobs.
- Rate limit the public demo; cap uploads per user.
- Users see short error messages; the server logs the real cause.
## Evaluation rules (Stage 4 and 6)
- Decide the dev/test document split BEFORE writing router rules; write it in the
  repo. Never tune on test documents. The 5 current samples are dev documents.
- Choose documents for structural diversity (sectioned, short, table-heavy, poorly
  structured), not subject. Keep scanned PDFs out of the benchmark.
- Question format: see `eval/question_format_example.json` (answer, evidence list of
  page + exact text, verified flag). A chunk is relevant if it contains the evidence text.
- Compare: fixed-size dense, recursive dense, section-aware dense, section-aware
  hybrid, hybrid+rerank, the router, best single strategy (chosen on dev only),
  and an oracle (upper bound only).
- Metrics: Recall@k, MRR, answer correctness, faithfulness, latency, cost.
- Ablations: remove rerank, hybrid, parent retrieval.
- Use cluster bootstrap (resample documents, not questions) for confidence intervals.
- Report router pick frequencies and a written failure analysis.
- Hand-verify every question. One command (e.g. `make eval`) reproduces results.
## Stage plan and current status
Stages 0-9 (skeleton, upload/parse, baseline RAG, strategy/router, mini eval,
hybrid/rerank, full eval, notebooks, frontend, hardening/deploy).
**Status: Stage 2 Step 2 done (chunk storage). Next: Stage 2 Step 3 (embeddings with
BAAI/bge-small-en-v1.5 behind an Embedder seam).**
See `PROGRESS.md`.
 
If time runs short, cut in this order: streaming, document-level index experiment,
compare mode, evidence-panel polish. Never cut evaluation, reliability checks
(not-found, injection test, delete test), or deployment.
