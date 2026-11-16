# Stage 4: Mini evaluation (code-only)

Branch: `stage4-mini-evaluation` (created by the owner). Rules: `.claude/AUTONOMOUS.md`.
Dev documents only (`data/samples/`, the 9 dev entries in `eval/corpus.json`). No new
dependencies are expected: the bge tokenizer comes with the installed
sentence-transformers / transformers.

## Purpose
Write everything needed to measure how well each strategy (fixed_size, recursive,
section_aware) and the router (rule v1) find the right chunk on the dev documents, with
questions built automatically and checked by code, and to measure the known issues. The
owner runs it afterwards.

## 1. Question set (machine-verified, owner's decision)
- `scripts/generate_questions.py` (no arguments: all 9 dev documents) writes
  `eval/questions/generated/<corpus id>.json`. Keep `eval/questions/paper_01.json` (10
  owner-verified questions) unchanged as a separate set.
- Passages come from the parser output (`parse_pdf`), never from any strategy's chunks, so
  no strategy is favoured. A passage is a window of consecutive blocks on one page (about
  150-300 words). Spread passages over the whole document. Table questions use table
  blocks. Multi-section questions use two passages from pages far apart. Pick passages
  with a fixed seed.
- Per document draft about 14 questions (about 8 factual, about 4 table or number if the
  document has tables, about 2 multi-section) and keep up to 10 that pass verification.
- The generator call (through the `LLM` seam, temperature 0) gets one passage (two for
  multi-section) and returns JSON: question, short answer, type, and an evidence quote
  copied exactly from the passage (at most 25 contiguous words). The question must
  paraphrase, must not quote the passage, and must not mention pages or section numbers.
  Invalid JSON means the draft is rejected and counted.
- Machine verification. Each check is a pure, unit-tested function; the script prints the
  rejection count per check:
  1. The evidence quote is found in the parsed document text after `normalise()`.
  2. Evidence pages are computed by code (every occurrence of the quote), never taken from
     the LLM.
  3. More evidence locations: find every page that contains the short answer string. A
     separate judge call asks for each candidate passage "does this passage alone answer
     the question?". Each yes becomes another evidence entry. Cap the candidates per
     question at 5.
  4. Judge call (separate prompt): the question is answerable from the evidence, has one
     correct answer within the document, and needs the document. Reject on any no.
  5. Lexical overlap: reject a question when 80% or more of its content words appear in
     its evidence quote. Print the overlap distribution.
  6. Remove near-duplicate questions within a document.
- Record per question: id, question, answer, type, evidence list (page + text),
  `verified: true`, `verification: "machine"`, the checks passed, the generator model,
  and the prompt version. Keep `load_questions()` working for `paper_01.json`.
- Calibration: the script also runs checks 1, 4 and 5 on the 10 owner-verified paper_01
  questions and prints how many it accepts.
- The generator and verifier prompts and code are frozen after this stage. Stage 6 builds
  test questions with exactly this pipeline, untuned.
- Targets, printed by the script as met or not met: at least 70 accepted questions, at
  least 15 table or number questions, at least 10 multi-section questions. If a target is
  missed, the script offers one more generation round (`--extra-round`) and never loosens
  the checks.

## 2. Relevance rule
- Pure function in `src/adaptrag/evaluation/`: a chunk is relevant to an evidence entry if
  its normalised text contains the whole normalised evidence, OR (the evidence is split
  across a chunk boundary) it contains a contiguous run of at least half of the evidence
  words. The section_aware heading prefix does not count either way.
- A question is answered at rank r if the chunk at rank r is relevant to any of its
  evidence entries.
- Report per strategy how many questions had their evidence split across chunks.

## 3. Metrics and the one command
- Pure functions with hand-computed unit tests: Recall@1, Recall@5, Recall@10, MRR (cut
  at 10), and a cluster bootstrap 95% interval (resample documents, not questions; fixed
  seed; 10,000 resamples; standard library `random`).
- `scripts/mini_eval.py` (no arguments; dev only). It builds any missing chunk sets and
  embeddings for all three strategies on the 9 dev documents through the existing strategy
  and registry code (real bge-small, exact search). It retrieves the top 10 for every
  question under every strategy and prints one table with these rows: fixed_size,
  recursive, section_aware, router (each document's stored strategy), best single
  strategy (chosen on these dev numbers), and oracle (best strategy per document, upper
  bound only). Columns: Recall@1, Recall@5, Recall@10, MRR, 95% intervals for Recall@5
  and MRR, N questions. The owner-verified paper_01 set is its own row.
- It also prints breakdowns per question type and per document, and paired wins/losses per
  question between strategies. It writes `eval/results/stage4/results.json` and
  `eval/results/stage4/summary.md`. Same inputs give the same output.
- `summary.md` is generated automatically and includes:
  - side-by-side chunk boundaries for 2 documents (one section_aware-routed, the recursive
    one, doc_08)
  - a TIE line for any two strategies within 0.01 on Recall@5, with the per-question data
    that explains it
  - every failed question under the router: the evidence, the top 3 retrieved chunks
    (rank, score, pages, first 200 characters), and which strategy found it if any
  - the top-1 score distribution for answerable dev questions and for 20 fixed off-topic
    questions written in the script, with the threshold proposal that separates them best.
    Do NOT change `retrieval_min_score` (owner decides).

## 4. Known issues (measured by the scripts)
- Glued words: a general detector, not a fixed list. A token is a candidate if it is not in
  the bge tokenizer's vocabulary and it splits exactly into two vocabulary words of at
  least 2 letters, or it has a lowercase-to-uppercase join like "theWMT". `mini_eval.py`
  prints counts and examples per document, and how many questions' evidence contains a
  glued token. Measure only; no parser change in this stage.
- Table chunks over 512 tokens: add a `table_token_budget` parameter to the shared table
  chunking (split a large table into chunks of at most N model tokens, repeating the
  header row). It is part of the strategy params, so the cache key changes. Keep the
  current behaviour as the default (no budget). `mini_eval.py` evaluates recursive and
  section_aware both without the budget and with a budget of 512, counts table chunks over
  512 tokens for each, and prints the adoption rule's outcome. The rule, fixed now: ADOPT
  if table chunks over 512 tokens drop to 0 AND Recall@5 on table questions drops by no
  more than 1 question for either strategy; otherwise KEEP. Changing the default happens
  in a follow-up prompt after the owner's run.
- "Very short" rule: `mini_eval.py` prints dev word and page counts. Rule, fixed now: add
  no "very short" rule unless at least 2 dev documents are under 2,000 words. Print the
  outcome.
- Section-aware boundaries vs prefix: state in `summary.md` that Stage 4 cannot separate
  them (the Stage 5 switch will).
- Router rule v1: `mini_eval.py` prints, per document, the routed strategy and the best
  strategy. Rule, fixed now: propose a rule v2 only if the routed strategy loses on at
  least 3 of 9 documents AND one simple profile threshold separates those documents.
  Print "KEEP v1" or the candidate threshold. Do not change the router in this stage.

## 5. Tests (normal run: no real model, no real LLM)
Unit tests for: each metric (hand-computed example), relevance (whole match, split
evidence, no match, prefix ignored), bootstrap (same seed gives the same interval;
document-level resampling), each verification check, JSON parsing of generator replies
(valid, invalid, missing fields), the usage-budget guard, the glued-word detector (with a
small fake vocabulary), the table token budget (header row repeated, no chunk over the
budget, with a fake token counter), and the three decision rules (ADOPT/KEEP, very short,
router). Generator and judge flows are tested with `FakeLLM`.

## 6. Docs (written now, numbers as TO FILL)
- Decision notes: 0024 question set and verification (state the credibility cost plainly:
  no human checked the questions; the questions are worded from the parser's text, which
  favours retrieval; the judge is the same model as the generator; the verifier's
  acceptance on the owner set is TO FILL), 0025 mini-eval design and results (TO FILL).
- In CLAUDE.md, replace "Hand-verify every question" with the rule from note 0024. Update
  the layout, commands and status ("Stage 4 code written, awaiting the owner's run").
- PROGRESS.md: Stage 4 plan, log entry (code written), and the explain-aloud questions,
  including: "Why does the relevance rule accept half of a split evidence quote?"

## Done when
Every item in sections 1-6 exists as code, tests or docs; every changed file has been
re-read once; and the final message is the final report with the headings from
`AUTONOMOUS.md`, including the run checklist:
1. `docker compose up -d db && python -m alembic upgrade head`
2. the checks chain from CLAUDE.md, with the expected test count
3. `python scripts/generate_questions.py` (costs money)
4. `python scripts/mini_eval.py`
