# 0012: Embedder

Status: Accepted
Date: 2026-10-31

## Context
Stage 2 Step 3 adds the part that turns text into vectors. Note 0010 chose
BAAI/bge-small-en-v1.5 (384 dimensions, 512-token input limit). The chunker counts
words, not model tokens, so we also need to measure how many real tokens our chunks
have, and whether any are longer than the model can read. Embeddings are not stored
yet, and ingestion does not call the embedder yet (Step 4).

## Decision
- **A seam.** `embedding/base.py` defines an `Embedder` protocol: `model_name`,
  `dimension`, `max_tokens`, `embed_passages(texts)`, `embed_query(text)`,
  `count_tokens(text)` (including the model's special tokens, [CLS] and [SEP]). The rest
  of the app depends only on this protocol.
- **Real implementation.** `embedding/sentence_transformer.py` holds
  `SentenceTransformerEmbedder` (sentence-transformers) and is the only module that
  imports sentence_transformers or torch. New dependency `sentence-transformers>=6.2`,
  reason: local, free, reproducible embeddings (note 0010).
- **Fake implementation.** `embedding/fake.py` holds `FakeEmbedder`: vectors from a
  SHA-256 hash of the text, normalised, dimension 8; `count_tokens` = word count + 2.
  Deterministic, no model, no network. Used by normal tests now and by ingest tests in
  Step 4.
- **Device `cpu` everywhere**, so numbers are the same on the Mac and in Docker.
- **CPU-only PyTorch in Docker and CI**, installed from
  https://download.pytorch.org/whl/cpu before the project is installed. pip then sees
  torch as already installed and does not fetch the default build. In the Dockerfile,
  torch has its own layer before the source code is copied, so code changes do not
  reinstall it. On the Mac a plain install is fine (macOS wheels have no CUDA).
- **Settings** (prefix `ADAPTRAG_`): `embedding_model = "BAAI/bge-small-en-v1.5"`,
  `embedding_device = "cpu"`, `embedding_query_instruction = "Represent this sentence
  for searching relevant passages: "`.
- **Query instruction on queries only.** bge models were trained on short queries
  paired with long passages, with this instruction in front of the query. Chunks
  (passages) are embedded as plain text. `embed_query` adds the prefix itself; we do not
  rely on prompts stored in the model's config.
- **Normalised vectors** (`normalize_embeddings=True`): every vector has length 1, so
  cosine similarity equals the dot product.
- **Load once.** `get_embedder(model_name, device, query_instruction)` is cached with
  `lru_cache`: the model loads on first use and is reused for the life of the process.
- **Plain lists in the interface** (`list[float]`), not numpy arrays.
- **Slow tests.** Tests that load the real model are marked `@pytest.mark.slow` and
  excluded by default (`addopts = "-m 'not slow'"`); `python -m pytest -m slow` runs
  them. CI never runs them and never downloads the model.
- **Measurement.** `scripts/token_stats.py FOLDER` chunks every PDF with the
  `chunk_fixed_size` defaults (200/40 words) and counts tokens per chunk with the real
  tokenizer, without truncation. `--show-over` also lists each chunk over max_tokens
  (index, pages, token count, first 200 characters).
- **Chunks over the token limit (owner's choice, option A).** Keep the fixed-size
  defaults at 200/40 words unchanged, so the control strategy stays fixed. Record the
  overflow as a measured known limit. In Step 4, ingest logs a warning and counts the
  chunks over max_tokens per document.

## Measurement (data/samples, fixed-size 200/40)
Exact output of `python scripts/token_stats.py data/samples`:
```
model BAAI/bge-small-en-v1.5, max_tokens 512
paper_01_single_column.pdf: 38 chunks, tokens min 211 / median 299.5 / max 442, 0 over 512
paper_02_two_column.pdf: 62 chunks, tokens min 73 / median 319.5 / max 474, 0 over 512
paper_03_single_column_b.pdf: 29 chunks, tokens min 195 / median 305 / max 517, 1 over 512
paper_04_table_heavy.pdf: 64 chunks, tokens min 98 / median 322.0 / max 448, 0 over 512
paper_05_short.pdf: 20 chunks, tokens min 235 / median 272.5 / max 654, 2 over 512
OVERALL: 213 chunks, tokens min 73 / median 313 / max 654, 3 over 512
```

What it means: 3 of 213 chunks (1.4%) exceed 512 tokens (paper_03: 1, paper_05: 2;
the worst is 654 tokens, about 3.3 tokens per word on dense IEEE pages). The model
silently ignores text past 512 tokens, so the end of those chunks is stored but not
embedded. Note 0010 argued that 200 words would fit; the measurement shows it does for
98.6% of chunks, not all.

Slow tests (`python -m pytest -m slow`): 5 passed.

Docker: `adaptrag-api` is 2.33 GB on disk, 523 MB content size. Verified inside the
image: `torch 2.15.0+cpu`, `torch.cuda.is_available()` is False, and no nvidia/cuda
packages are installed.

## Rejected
- **Default PyTorch build in Docker/CI:** on Linux x86_64 it pulls in CUDA/GPU
  libraries we can never use (no GPU in Docker on the Mac or in CI), which makes the
  image and the CI install much bigger and slower.
- **`mps` device (Apple GPU):** faster on the Mac, but Docker and CI can't use it, and
  results could differ slightly between machines. Reproducible numbers matter more for
  the benchmark than speed here.
- **Downloading the model in every test run:** slow, needs network, and makes CI
  depend on Hugging Face being up. The fake covers the logic; slow tests cover the model.
- **Loading the model per call:** reads the weights from disk and builds the model
  again on every call; load time and memory have not been measured, but both are paid
  once per call instead of once per process.
- **No query instruction:** the bge model card recommends it for short queries
  against passages; leaving it out would use the model differently from how it was trained.
- **numpy arrays in the interface for now:** would make every caller depend on numpy
  types. Lists are enough for Chroma (Step 4); revisit if conversion cost shows up in a
  measurement.
- **B: a smaller chunk size for the overflow:** changes a benchmark decision after
  seeing data, and at 3.3 tokens per word even 150 words is not guaranteed to fit.
- **C: splitting chunks over the token limit:** ties the chunker to the model's
  tokenizer, which note 0010 rejected.

## Known limits
- 3 of 213 sample chunks (1.4%) exceed 512 tokens; the model ignores the text past
  512 tokens, so the end of those chunks is not embedded. Kept on purpose (option A):
  the 200/40 control stays fixed, and from Step 4 ingest warns about and counts these
  chunks per document.
- The model is downloaded from Hugging Face on first use (into the user's cache). The
  Docker image does not contain it yet; Step 4 decides whether to bake it in or mount
  a cache volume.
- `get_embedder` is cached per process. Several Uvicorn workers would each load a copy.
- Embedding is single-process on the CPU; speed has not been measured.
- The Hugging Face tokenizer prints a "longer than the specified maximum sequence
  length" warning when `count_tokens` sees a long text. It is expected (we count
  without truncating on purpose) and harmless.
- The api image is 2.33 GB on disk (523 MB content size) with CPU-only torch, and it
  does not include the model yet.
