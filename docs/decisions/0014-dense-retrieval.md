# 0014: Dense retrieval and the query endpoint

Status: Accepted
Date: 2026-10-31

## Context
Stage 2 Step 4 stores one vector per chunk (pgvector, note 0013). Step 5 makes them
searchable: given a ready document and a question, return the top-k most similar chunks
with their pages and scores. There is no LLM yet (Step 6) and no evaluation set yet, so
there is no measured basis for deciding when retrieval is "too weak".

## Decision
- **Endpoint:** `POST /documents/{document_id}/query`, plain `def` (sync SQLAlchemy).
  Body `{"question": str, "k": int = 5}`. The embedder comes from
  `Depends(get_app_embedder)`, as in `POST /documents`.
- **Validation (422):** the question is stripped of surrounding whitespace, then must
  have 1-1000 characters (so a blank question is rejected); `k` must be 1-20.
- **Errors:**
  - 404 "Document not found." if the document does not exist.
  - 409 "Document is not ready (status: <status>)" if its status is not "ready"
    (uploaded, failed, or an old "parsed" row).
  - 409 "Document needs re-ingestion for the current settings" if the document has no
    current chunk set, or that set has no vectors from the embedder's model.
  409 (conflict with the resource's state), not 404, because the document exists; the
  message tells the user what to do.
- **Response (200):** `{document_id, chunk_set_id, model_name, results: [{chunk_id,
  chunk_index, page_start, page_end, score, text}]}`, best first. `chunk_set_id` and
  `model_name` show exactly which stored vectors answered, which matters once several
  strategies and models exist.
- **Current chunk set only:** `retrieval/dense.py` `current_chunk_set_id` finds the set
  ingestion would create now: strategy `fixed_size`, `FIXED_SIZE_PARAMS` from
  `ingest.py`, `PARSER_VERSION`. Using the same constants as ingestion means the two can
  not drift apart.
- **Cosine distance, score = 1 - distance.** pgvector's `cosine_distance` (`<=>`) is
  `1 - cosine similarity`, from 0 (same direction) to 2 (opposite). Postgres sorts by
  distance (smaller is closer); the API reports `1 - distance`, which is the cosine
  similarity, so a higher score means a better match and 1.0 is a perfect match.
- **Exact search with a deterministic tie-break:** `ORDER BY distance, chunk_index
  LIMIT k`, no ANN index (note 0013). The same query always returns the same chunks in
  the same order, which the benchmark needs.
- **SQL only in `vectorstore/pgvector.py`:** `search(session, chunk_set_id, model_name,
  query_vector, k)` filters by chunk set and model name, joins `chunks`, and returns
  `ScoredChunk` objects (`retrieval/results.py`, frozen and slotted). It raises
  ValueError if the query vector does not have `EMBEDDING_DIMENSION` numbers or k < 1.
  `retrieve_dense` only embeds the question (`embed_query`, with the bge query
  instruction) and calls `search`.
- **No "weak retrieval" threshold in this step.** Every request returns its top k,
  whatever the scores. Step 6 chooses the "not found in your documents" threshold from
  scores measured on real questions.
- **Scripts:** `scripts/query.py` runs the same checks with the real model and prints
  the results; `scripts/list_documents.py` shows which documents have their upload file
  locally; `scripts/reingest.py --file PATH` restores a missing upload after checking
  its SHA-256.

## Rejected
- **GET with a query string** (`GET /documents/{id}/query?q=...`): questions can be long
  and contain any characters; URLs get logged by proxies and servers; the body will grow
  (filters, options) in later stages. POST with a JSON body has none of these problems.
- **Inner product** (`<#>`): for normalised vectors it ranks the same as cosine, but
  pgvector returns it negated, so the score would need a sign flip and would silently
  become wrong if a model ever returned vectors that are not normalised. Cosine
  distance stays correct either way and reads naturally as 1 - similarity.
- **Searching any existing chunk set** (for example, the newest one): a document can hold
  sets from older params or parser versions. Searching one of those would silently mix
  old and new behaviour into the benchmark. Asking for re-ingestion is explicit.
- **A guessed threshold** (for example, "score < 0.5 means not found"): bge scores have
  their own range, and no measurement exists yet. A guessed number would look like a
  decision without being one. Step 6 measures first.

## Known limits
- Only the fixed-size chunk set is searched until the router exists (Stage 3).
- Exact search reads every vector of the chunk set; fine only at small scale (note 0013).
- The first query after a restart loads the model during the request, because the
  embedder dependency runs before the endpoint (even for a 404 or 409 answer).
- `FakeEmbedder` scores carry no meaning, so tests check shape, status codes, order,
  limits and the score formula, never that the "right" chunk wins. Ranking quality is
  measured in the Stage 4 evaluation with the real model.
- One document per query; multi-document search comes with notebooks (Stage 7).
