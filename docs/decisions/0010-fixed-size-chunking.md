# 0010: Fixed-size chunking (control strategy) and embedding model

Status: Accepted
Date: 2026-10-31

## Context
Stage 2 needs a baseline RAG pipeline. The fixed-size chunker is the control that the
router must beat in the benchmark, so it must be simple, structure-blind and reasonable
(not a strawman). Chunk size depends on the embedding model's input limit, because
longer input is silently cut off.

## Decision
- Embedding model: BAAI/bge-small-en-v1.5 (512-token limit). Installed in Stage 2 Step 3.
- Unit: words (split on whitespace), not characters or model tokens.
- Size 200 words, overlap 40 words (20%). Defaults in `chunking/fixed.py`, protected by a test.
- Chunks may cross page boundaries; each chunk stores page_start and page_end.
- In-memory shape is `TextChunk`; the database model (Step 2) will be `Chunk`.
- Our own ~30-line function, not a framework text splitter.

## Rejected
- all-MiniLM-L6-v2: 256-token limit is too tight for 200 technical words.
- e5-small-v2: close alternative; no requirement separates it from bge-small.
- nomic-embed-text: needs trust_remote_code; larger.
- Paid embedding API: per-call cost, needs key and network, benchmark runs many times.
- Counting in model tokens: needs the tokenizer now and ties chunks to the model, so
  swapping models would change the chunks and confound the benchmark.
- Restarting chunks at page breaks: tiny leftover chunks; adds structure to the control.
- LangChain/LlamaIndex splitters: large dependency for little code; no page tracking.

## Known limits
- Words are not tokens. Real token counts are measured in Step 3, not assumed.
- Chunks cut mid-sentence on purpose (structure-blind control).
- Page furniture (headers, footnotes, arXiv stamp) is included in chunks.
- Size and overlap are not tuned; Stage 4 may compare 2-3 sizes on dev documents only.
