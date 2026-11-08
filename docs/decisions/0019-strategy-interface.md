# 0019: Strategy interface and registry

Status: Accepted
Date: 2026-11-08

## Context
Until now ingestion and retrieval were hard-wired to the fixed-size chunker. Stage 3 adds
more strategies and a router that picks one per document, so ingest, `/query` and
`/answer` have to work with any strategy without knowing which one it is.

## Decision
- **`Strategy` is an `abc.ABC`** (`strategies/base.py`). Strategies are our own family and
  share code: `retrieve()` has one default, dense search, until Stage 5.
- **The embedder is passed in at construction.** A strategy never builds its own, so
  tests pass `FakeEmbedder`.
- **The registry is an explicit dict** (`STRATEGIES` in `strategies/registry.py`): the
  full list can be read in one place and importing it has no side effects. Adding a
  strategy is one new file in `strategies/` plus one line in the dict.
- **Params belong to the strategy.** `name`, `params` and the parser version identify a
  chunk set, so they form the cache key together.
- **`DEFAULT_STRATEGY = "fixed_size"`** until the router chooses per document (Step 5).
- **`current_chunk_set_id` in `retrieval/dense.py` takes a name and params, and `dense.py`
  never imports `ingest`.** Otherwise an import loop would form: ingest -> registry ->
  strategies -> base -> dense -> ingest. `tests/test_import_order.py` guards this by
  importing each module in a fresh interpreter.

## Proof of extensibility
`tests/test_strategy_extensibility.py` runs ingest and `/query` with a strategy registered
only in the test, with no change to production code.

## Rejected
- **`typing.Protocol`:** suits outside implementations, such as the `Embedder` and `LLM`
  seams. A Protocol would not give every strategy the shared `retrieve()`.
- **A registration decorator:** registration would become a side effect of importing a
  module, and the full list of strategies could not be read in one place.

## Known limits
- `scripts/query.py`, `scripts/answer.py` and `scripts/gate_check.py` pair
  `DEFAULT_STRATEGY` with the fixed-size `PARAMS` for the early chunk-set check, made
  before the model loads. To be fixed in Step 5, when each document stores its own
  strategy.
- `STRATEGY_NAME` still lives in `chunking/fixed.py`.
- All strategies retrieve densely until Stage 5.
- A strategy object is built per request. This is cheap: it only holds the embedder.
