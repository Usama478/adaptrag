# 0025: Mini evaluation design and results

Status: Accepted (code written; results are `TO FILL` until the owner's run)
Date: 2026-11-16

## Context
Stage 3 built three strategies and router rule v1 as a hypothesis (note 0021). Stage 4
measures, on the 9 dev documents only, how well each strategy and the router find the
right chunk, and measures the known issues from note 0022 (glued words, table chunks over
512 tokens, the "very short" rule). Everything that depends on a result is written as a
rule BEFORE the run (note 0023), and the script prints each rule's outcome.

## Decision
- **One command:** `python scripts/mini_eval.py` (no arguments, no LLM calls). It builds
  any missing chunk sets and embeddings through the existing strategy code (real
  bge-small, exact search), retrieves the top 10 for every question under every variant,
  prints the tables and writes `eval/results/stage4/results.json` and `summary.md`.
  Nothing in the output depends on the clock, so the same inputs give the same files.
- **Relevance rule** (`evaluation/relevance.py`). A chunk is relevant to an evidence entry
  if its normalised text contains the whole normalised evidence, OR a run of consecutive
  evidence words at least half the evidence long (rounded up). The second part exists
  because a chunk boundary can cut an evidence quote in two; without it, a strategy would
  be punished twice for one cut (no chunk "contains" the evidence, although the answer is
  retrievable). With "at least half", the chunk that holds the larger part of a cut quote
  counts (both chunks, for an even cut), and a chunk that only shares a few words with
  the quote does not. A question is answered at rank r if the chunk at rank r is relevant
  to any of its evidence entries. The section-aware heading prefix is removed before
  matching, so it counts neither for nor against a chunk.
- **Split evidence is reported:** per variant, how many questions have no chunk holding an
  evidence entry whole (split), and how many have no relevant chunk at all (missing).
- **Metrics** (`evaluation/metrics.py`): Recall@1, Recall@5, Recall@10 (share of
  questions with a relevant chunk in the top k) and MRR cut at 10. 95% intervals for
  Recall@5 and MRR by cluster bootstrap: documents are resampled, not questions, 10,000
  resamples, fixed seed, standard library `random`. With 9 documents the intervals will be
  wide; that is the honest picture.
- **Rows:** fixed_size, recursive, section_aware, router (each document's stored
  strategy), best single strategy and oracle (best strategy per document). Best single and
  oracle are chosen on these same dev numbers, so both are optimistic; the oracle is an
  upper bound only. Strategies are compared by Recall@5 hits, then MRR; the control
  (fixed_size) wins an exact tie. The owner-verified paper_01 set has its own rows (one
  document, so no interval).
- **Also printed:** per question type, per document, paired outcomes per question between
  strategies, a TIE line for strategies within 0.01 on Recall@5 with the questions that
  differ, every question the router failed (not in the top 5) with its evidence and top 3
  chunks, side-by-side chunk starts for one section_aware-routed document and doc_08, and
  the top-1 score distribution for answerable questions and 20 fixed off-topic questions
  with the threshold that separates them best. `retrieval_min_score` is NOT changed.

## Rules fixed before the run
- **Table token budget.** New parameter `table_token_budget` on recursive and
  section_aware (`chunking/segments.py`): a table is cut into pieces of whole rows that
  fit both the word budget and N model tokens, and every piece starts with the header row
  (and its divider). For section_aware the heading prefix counts towards the tokens. It is
  off by default and absent from the params then, so existing chunk sets keep their cache
  key; with a value it is part of the params. Rule: **ADOPT** a budget of 512 if table
  chunks over 512 tokens drop to 0 AND hits@5 on table questions drop by no more than 1
  question, for both strategies; otherwise **KEEP** the current behaviour. Changing the
  default is a follow-up after the owner's run.
- **"Very short" rule.** No such router rule unless at least 2 dev documents are under
  2,000 words.
- **Router rule v1.** Propose a rule v2 only if the routed strategy loses on at least 3 of
  the 9 documents (another strategy has more hits@5 there) AND one profile threshold
  separates those documents from the rest. The router is not changed in this stage.
- **Glued words** are measured, not fixed: a token is a candidate if it is not in the bge
  tokenizer's vocabulary and splits into two vocabulary words of at least 2 letters, or if
  a lowercase letter is followed by an uppercase one. Candidates include real compounds
  and product names, so the report shows examples.
- **Section-aware: boundaries or prefix?** Stage 4 cannot separate them: the strategy
  differs from recursive in both. The Stage 5 switch will.

## Results (owner's run)
- Results table (6 rows, R@1 / R@5 / R@10 / MRR / intervals / N): TO FILL from owner's
  run: `eval/results/stage4/summary.md`.
- Owner-verified paper_01 rows: TO FILL from owner's run.
- Best single strategy: TO FILL from owner's run.
- Router vs best single vs oracle, in one sentence: TO FILL from owner's run.
- Ties: TO FILL from owner's run.
- Per type and per document: TO FILL from owner's run.
- Evidence split / missing per variant: TO FILL from owner's run.
- Glued-word candidates per document, and questions with glued evidence: TO FILL from
  owner's run.
- Table token budget: table chunks over 512 without / with, table hits@5 without / with,
  outcome ADOPT or KEEP: TO FILL from owner's run.
- Very short: dev word counts and outcome: TO FILL from owner's run.
- Router: documents where the routed strategy loses, outcome KEEP v1 or candidate
  threshold: TO FILL from owner's run.
- Top-1 scores and threshold proposal: TO FILL from owner's run.
- Failure analysis (written by the owner from the failed-question list): TO FILL from
  owner's run.

## Known limits
- 9 documents and about 10 questions each: small. Differences of a few questions between
  strategies are inside the noise; the intervals and the paired outcomes say how much.
- The questions are machine-verified (note 0024); the absolute level is optimistic.
- Half of a split quote counts as relevant, but that chunk may hold the half without the
  answer. The split counts show how often the rule was needed.
- Retrieval only: no answer correctness, faithfulness, latency or cost here (Stage 6).
- The off-topic questions are the easy case for a threshold; in-domain questions without
  an answer score much higher (note 0015), so the proposal is a lower bound, not a fix.
- The router loses or wins per document on about 10 questions; the rule's "at least 3 of
  9 documents" guards against reacting to one noisy document, not against noise overall.
