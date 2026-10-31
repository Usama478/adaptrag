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
  tokenizer, without truncation.

## Measurement (data/samples, fixed-size 200/40)
TO FILL from owner's run: full output of `python scripts/token_stats.py data/samples`
(one line per PDF with chunks, min / median / max tokens and chunks over max_tokens,
then the OVERALL line).

What it means: TO FILL from owner's run: how many chunks exceed 512 tokens and in which
papers. Text past 512 tokens is cut off silently by the model and never embedded.

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

## Known limits
- Chunks over 512 tokens are truncated by the model (see Measurement). The chunker
  still counts words; whether to cap chunks by tokens is a Step 4 or Stage 3 question.
- The model is downloaded from Hugging Face on first use (into the user's cache). The
  Docker image does not contain it yet; Step 4 decides whether to bake it in or mount
  a cache volume.
- `get_embedder` is cached per process. Several Uvicorn workers would each load a copy.
- Embedding is single-process on the CPU; speed has not been measured.
- The Hugging Face tokenizer prints a "longer than the specified maximum sequence
  length" warning when `count_tokens` sees a long text. It is expected (we count
  without truncating on purpose) and harmless.
- api Docker image size with CPU torch: TO FILL from owner's run: SIZE column for
  `adaptrag-api` in `docker images`.
