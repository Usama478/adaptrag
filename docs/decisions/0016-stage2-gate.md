# 0016: Stage 2 gate

Status: Accepted
Date: 2026-11-01

## Context
Stage 2 built the baseline pipeline: fixed-size chunks (note 0010), bge-small
embeddings (note 0012), pgvector exact search (notes 0013, 0014) and gpt-4o-mini answers
with page citations and a two-layer not-found (note 0015). Before Stage 3 adds
strategies and a router, the baseline must be shown to work end to end on real data:
for a set of hand-checked questions, does the answer cite the page that holds the
evidence? This is a pass/fail check, not the benchmark (that is Stage 4).

## Decision
- **Questions.** `eval/questions/paper_01.json`: 10 questions on paper_01 ("Attention Is
  All You Need"), in the format of `eval/question_format_example.json` (answer, evidence
  list of page + exact text, `verified` flag). q10 has two evidence entries.
- **Only verified entries are scored.** An entry with `"verified": false` is skipped. A
  verified entry whose evidence still has a null page or a "TO FILL" text is refused:
  it cannot be scored honestly. Malformed files (missing keys, repeated ids, empty
  evidence) stop the loader with `QuestionFileError`.
- **cited_ok (the gate counts this).** The answer was found AND at least one citation's
  page_start..page_end contains at least one evidence page.
- **retrieved_ok (reported only).** At least one top-k source contains at least one
  evidence text, after both are normalised (casefold, whitespace runs collapsed). It is
  printed next to cited_ok so a miss can be traced to retrieval or to the LLM, but it is
  not part of the verdict: the gate asks whether the user gets the right page, and
  retrieved_ok depends on how the evidence text was chosen (see caveats).
- **Pass rule.** At least 10 scored questions and at least 80% cited_ok (8/10). Fewer
  than 10 scored questions is a FAIL, whatever the score.
- **An LLM failure stops the run.** `LLMError` on any question ends `gate_check.py` with
  the cause. Counting it as a miss would mix outages with answer quality; skipping it
  would drop a question silently.
- **Same pipeline as the API.** `scripts/gate_check.py` runs the same checks as
  `POST /documents/{id}/answer` (document ready, current chunk set, embeddings for the
  model) and calls `answer_question` with the settings' embedder, LLM and
  `retrieval_min_score`. Each answer is timed with `time.perf_counter()`; the model is
  loaded before the first question, so load time is not in the per-answer numbers.
- **Rules are pure functions** in `src/adaptrag/evaluation/gate.py` (`normalise`,
  `cited_ok`, `retrieved_ok`, `gate_passed`, `load_questions`), unit-tested in
  `tests/test_gate.py` without a database, model or LLM.

## Result (owner's run)
Settings: document 1 = paper_01, gpt-4o-mini, k=5, `retrieval_min_score` 0.55.

| Check | Result |
| --- | --- |
| Cited the right page (cited_ok) | 10/10 |
| Evidence in top-5 (retrieved_ok) | 10/10 |
| Gate (>= 8/10) | PASS |

q10 (two evidence entries) cited pages 6 and 9.

Other Stage 2 gate items, all done: off-topic question answered "Not found in your
documents."; PWNED prompt-injection test not followed; cache tests; integration test;
Docker end-to-end check. Normal test run: 163 passed, 7 deselected.

**Latency and tokens.**
- Answer (`gate_check.py`, local): mean 1.79 s, max 2.80 s per answer (query embedding,
  search and LLM call; model load excluded).
- Model load (Docker, `POST /documents/{id}/query`): first query after an API restart
  3.23 s, later queries 0.035 s, so loading the model costs about 3.2 s once per process
  (note 0013).
- Tokens per answer: 1567-1999 input, 14-129 output.
- Money cost: see the owner's OpenAI usage page (not recorded here).

## Caveats
The PASS shows the pipeline works; it does not show how well it works in general.
- **One clean dev paper.** paper_01 is single-column and well structured; no two-column,
  table-heavy or poorly structured document was gated.
- **Easy questions, drafted by an LLM.** 10 mostly single-fact questions, drafted by an
  LLM and then checked by the owner. Few need two passages or reasoning across pages.
- **Evidence copied from our own chunks.** Evidence phrases were taken from our parsed
  chunk text, not independently from the PDF, so they match the parser's output
  (including its glued words). retrieved_ok is therefore easy here and says little.
- **Lenient citation rule.** Any citation whose page range contains an evidence page
  counts; extra citations are not penalised (q02 cited three chunks), and a chunk that
  spans pages gets credit for any of them. Citation precision was not measured.
- **One run.** Answers can vary between runs (note 0015); no repeat runs, no
  confidence interval.

## What Stage 4 changes
- Several dev documents chosen for structural diversity, with a dev/test split written
  down before router rules exist (CLAUDE.md evaluation rules).
- Questions and evidence built more independently of the system: evidence text taken
  from the PDF, not from our chunks, and more multi-fact and unanswerable questions.
- Citation precision measured (share of citations that hold evidence), not only "any
  citation hits".
- Recall@k and MRR per strategy, answer correctness and faithfulness, latency and cost;
  cluster bootstrap over documents for confidence intervals.
- Recalibrate `retrieval_min_score` (0.55 is provisional).

## Rejected
- **Scoring retrieved_ok in the gate:** with evidence copied from our own chunks it is
  close to guaranteed, so it would add a number without adding a check.
- **Counting an LLM error as a miss:** would turn an outage into a quality result.
- **Scoring unverified entries:** a number built on unchecked evidence is not honest.
- **A stricter citation rule now (every citation must hold evidence):** right for
  Stage 4, but the gate only asks whether the user can find the right page.
