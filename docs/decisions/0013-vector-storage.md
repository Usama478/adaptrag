# 0013: Vector storage

Status: Accepted
Date: 2026-10-31

## Context
Stage 2 Step 4 makes ingestion embed every chunk and store the vectors, so Step 5 can
search them. The plan (notes 0011, 0012) said Chroma. Since then Postgres already
holds documents, jobs, chunk sets and chunk text, and "delete means delete" must remove
vectors too. The embedding cache key (CLAUDE.md) is file hash + parser version +
strategy and params + model name: changing any of them must trigger re-embedding, and
nothing else should.

## Decision
- **pgvector in Postgres, not Chroma (owner's decision).** New table `chunk_embeddings`:
  chunk_id (FK chunks.id, ON DELETE CASCADE), model_name String(200),
  embedding vector(384); primary key (chunk_id, model_name). The `vector` extension is
  created by the migration (`CREATE EXTENSION IF NOT EXISTS vector`). New dependency
  `pgvector` (reason: SQLAlchemy column type for the vector extension).
- **Postgres image `pgvector/pgvector:pg17`** in docker-compose.yml and in CI: Postgres
  17 with the extension installed. The data folder format is the same as `postgres:17`,
  so the existing `pgdata` volume is reused.
- **pgvector imports** only in `models.py` (column type), `vectorstore/pgvector.py`,
  and the migration side (`migrations/env.py`, the migration file).
- **Exact search, no ANN index.** Step 5 compares the query with every stored vector.
  Exact search returns the same results every run, which the benchmark needs, and the
  scale is a few hundred chunks. No HNSW or IVFFlat index now.
- **Fixed 384 dimensions** (`EMBEDDING_DIMENSION` in models.py, bge-small). A typed
  column lets Postgres reject a vector of the wrong length; `save_embeddings` checks the
  length first and gives a clear ValueError.
- **Structural cache key.** No hash string: the key is the chain of rows.
  document (file hash, unique) -> chunk_set (parser version, strategy, params, unique per
  document) -> chunk_embeddings.model_name. "Already embedded" = rows exist for this
  chunk set and this model (`embeddings_exist`). A new parser version or new params make
  a new chunk set, which has no embeddings yet; a new model name has no rows yet.
- **One transaction** for chunks and vectors: parse, chunk, save, embed, save vectors,
  then commit. Any error rolls back all of it, so a failed job leaves no chunks and no
  vectors.
- **Over-limit chunks (option A, note 0012).** Before embedding, ingest counts chunks
  with more than `max_tokens` tokens and logs one WARNING per document with the count.
- **Embedder via a FastAPI dependency.** `embedding/provider.py` `get_app_embedder`
  imports the real embedder inside the function, so importing the app never imports
  torch. `POST /documents` passes the embedder to the background job. Tests override
  the dependency with `FakeEmbedder(dimension=384)`; `FakeEmbedder` now supports any
  dimension by joining several SHA-256 digests.
- **New document status "ready"** (parsed, chunked and embedded) replaces "parsed" as
  the success status. Statuses: uploaded / ready / failed.
- **Docker volumes.** `app-data` at /app/data (uploads) and `hf-cache` at
  /app/.cache/huggingface with `HF_HOME` set to it, on the api service. The Dockerfile
  creates both folders owned by appuser, because a new named volume copies the ownership
  of the image folder it is mounted on. The migrate service needs no volumes.
- **Re-ingest path.** `scripts/reingest.py DOCUMENT_ID` creates a new job and runs
  ingestion with the real model; stored work is skipped, missing work is done.

## Comparison
| | pgvector in Postgres (chosen) | Chroma in-process (rejected) |
| --- | --- | --- |
| Transactions | Chunks and vectors commit or roll back together | Two stores; a crash between them leaves them out of sync |
| Delete | ON DELETE CASCADE removes vectors with the document | Separate delete call; easy to forget, can fail separately |
| Schema | Alembic migration, checked by `alembic check` in CI | Collections managed in code, outside migrations |
| Tools | No new server or store; one fewer thing to run and back up | Another store with its own files, volume and backup |
| Metadata filters | Plain SQL joins (document, chunk set, model) | Its own filter syntax on copied metadata |
| Search features | Exact search now; HNSW/IVFFlat indexes exist if needed | Built for vector search; ANN index by default |
| Tests | Same Postgres test database as everything else | Needs its own temporary store per test |
| Lock-in | Tied to Postgres (already chosen, note 0004) | Easy to swap for another vector DB |

## Rejected
- **Chroma:** see the comparison. Its strengths (a dedicated vector API, ANN by default)
  do not matter at a few hundred chunks; the costs (two stores to keep in sync, a
  separate delete path) do.
- **An HNSW index now:** approximate search can return different neighbours from exact
  search; at this scale exact search is enough and keeps benchmark results stable.
  Add one only if a measurement shows search is too slow.
- **An untyped `vector` column (any length):** would accept vectors of mixed sizes from
  different models in one column, and would make an HNSW index impossible later.
- **Baking the model into the image:** a bigger image and a rebuild for every model
  change. The cache volume keeps the model across restarts. Revisit in Stage 9 if
  deployment needs it (for example, no network at start-up).

## Known limits
- A model with a different number of dimensions needs a new migration (new column or
  new table).
- The first upload after a restart loads the model during the request, because the
  dependency runs before the endpoint. Measured in the Stage 2 Step 7 Docker check
  (owner's run): the first `POST /documents/{id}/query` after an API restart took
  3.23 s, later ones 0.035 s, so the model load costs about 3.2 s once per process.
  `/query` and `POST /documents` use the same `get_app_embedder` dependency, so the
  first upload pays the same load. Any `POST /documents` triggers the load, even one
  that is then rejected or is a duplicate.
- The model is downloaded from Hugging Face on first use into the `hf-cache` volume;
  the first upload in a fresh Docker setup needs network access.
- Exact search reads every vector; fine only at small scale.
- Documents ingested before this step have no embeddings until `scripts/reingest.py`
  is run for them.
- Real-model ingestion speed has not been measured.
