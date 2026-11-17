# Stage 4 close-out (code-only)

Branch: `stage4-mini-evaluation`. Rules: `.claude/AUTONOMOUS.md` (run nothing). Do the part
the owner names (Part A or Part B), then end with the final report from AUTONOMOUS.md.

Owner's decisions (2026-11-18), from the Stage 4 results:
1. Table token budget: ADOPT (rule outcome) -> 512 becomes the default.
2. Not-found threshold: keep `retrieval_min_score` at 0.55 (the 0.567 proposal is not taken).
3. Glued-word detector: recorded as inconclusive; a better method is a Stage 5 item.

## Part A: table token budget on by default (code)
- `RecursiveStrategy` and `SectionAwareStrategy`: the default `table_token_budget` becomes
  512 (one named constant, e.g. `DEFAULT_TABLE_TOKEN_BUDGET = 512`, defined once and imported).
  `None` stays allowed and turns it off. The registry builds the default, so routed
  documents now use the budget. The params now include `table_token_budget: 512`, so the
  cache key changes: this is intended (routed dev documents need re-ingestion; the owner
  does that at the start of Stage 5).
- `scripts/mini_eval.py` must still reproduce the Stage 4 run exactly: build its base rows
  with `table_token_budget=None` and its "+tb512" rows with 512, explicitly. Row labels stay
  the same. Add a short comment saying why.
- fixed_size is unchanged. No migration, no new dependency, no new setting.
- Tests: update every test that assumed the old default (no budget). Add tests that the
  registry-built recursive and section_aware strategies have `table_token_budget` 512 in
  their params, and that `None` leaves it out of the params. In the final report give the
  exact number of new tests and the expected total (currently 538 passed, 7 deselected).

## Part B: docs (fill every `TO FILL from owner's run` for Stage 4)
Read `eval/results/stage4/summary.md` and `eval/results/stage4/results.json` for the
evaluation numbers. The question-generation numbers are not saved in a file; they are here
(owner's runs, prompt version stage4-v2):

- v2 round 0: drafted 180, passed 95, kept 84; table 4, multi-section 13; 177,521 input and
  14,179 output tokens.
- After the extra round (final set): drafted 360, passed 157, kept 90 (10 per document);
  table 6 (target 15: NOT MET), multi-section 24 (met), accepted 90 (met).
- Rejections, final: quote_length 99, near_duplicate 40, quote_not_found 28,
  lexical_overlap 26, judge_not_answerable 5, invalid_json 3, missing_fields 2, others 0.
- Lexical overlap of 228 drafts reaching check 5: 0.0-0.2: 32, 0.2-0.4: 48, 0.4-0.6: 67,
  0.6-0.8: 55, 0.8-1.0: 26 (rejected).
- Extra evidence locations: 66 confirmed of 106 candidates.
- Answerable without the document (label only): 37 of 90.
- Calibration on the 10 owner-verified paper_01 questions: accepted 6 of 10
  (judge_not_answerable 2, quote_not_found 1, lexical_overlap 1). v1 was 1 of 10.
- Extra round: 158,514 input and 13,503 output tokens. Stage 4 total (llm_usage.jsonl):
  442,030 input and 37,375 output tokens (about $0.09 at $0.15 / $0.60 per million).

Write:
1. **Note 0024:** fill the v2 results and calibration. State plainly: the verifier is strict
   (it rejected 4 of 10 owner-verified questions) and also lets bad questions through: at
   least 2 of the 14 router failures are invalid questions ("highest score recorded in the
   first passage" leaks the generator's view; "number of layers in ResNet-34" has figure-axis
   numbers as evidence). They stay in the set because no question is removed after scoring.
   Table questions stayed at 6; a table-question pipeline (v3) is decided before Stage 6, on
   dev only.
2. **Note 0025:** fill the results from summary.md and add the interpretation:
   - No strategy separates on dev: Recall@5 fixed_size 0.867, recursive 0.844,
     section_aware 0.844, router 0.844, intervals overlapping; Recall@5 and MRR disagree on
     the best strategy (fixed_size vs recursive); paired wins cancel (4-7 each way).
   - The oracle adds only 3 questions over the best single strategy (0.900 vs 0.867):
     routing has little to gain on dev. Router rule v1 is KEEP v1 under the rule, but it is
     not supported by dev data; revisit after Stage 5.
   - The questions are easy (84-87% Recall@5, high lexical overlap), which compresses
     differences.
   - Finding: section_aware puts the full heading path, starting with the DOCUMENT TITLE, on
     every chunk. "What does MTEB stand for?" fails because every chunk starts with "MTEB:
     Massive Text Embedding Benchmark". This is the hypothesis for the Stage 5 prefix switch
     (full prefix / no prefix / prefix without the title).
   - Failure patterns (14 under the router): invalid questions (2+), exact numbers and table
     cells (dense search is weak at them; BM25 in Stage 5), a reference-list question (the
     References heading was not detected, so references sit under "7 Conclusion"), the
     title prefix.
   - Split evidence is 0 for every variant because chunks overlap by 40 words and quotes
     are at most 25 words, so every quote fits whole in some chunk.
   - Glued-word detector: inconclusive. Its top candidates are real words the tokenizer
     splits into pieces (Transformer, decoder, dataset). A better method is a Stage 5 item
     (BM25 is sensitive to glued words).
   - Table token budget: ADOPT (table chunks over 512 tokens 25 -> 0 and 30 -> 0, table hits
     unchanged at 2/6 and 3/6; weak evidence with 6 table questions, but text past 512
     tokens is never embedded). Very short rule: NO RULE. Threshold: kept at 0.55 by the
     owner (0.567 was fitted on the same data and fixes 3 cases; an off-topic pass reaches
     the LLM, which answers NOT_FOUND, while a higher threshold could refuse real questions:
     the lowest answerable score was 0.548).
3. **New note 0026-stage4-gate.md**, like note 0022: the gate table (one command prints the
   table; chunk difference: paper_01 43/45/56, doc_08 52/56/57; tie recursive vs
   section_aware explained by 4 different questions each; one failure explained in full:
   the MTEB question). Gate: PASS.
4. **CLAUDE.md:** status "Stage 4 done (gate PASS, note 0026). Next: Stage 5"; fix the stale
   layout label "(FROZEN, stage4-v1)" to stage4-v2; the table budget default in Conventions;
   `retrieval_min_score` "0.55, kept after the Stage 4 measurement (note 0025)"; add
   0026 to the notes list.
5. **PROGRESS.md:** Stage 4 marked Done in the checklist and Current status (test count from
   the owner's run after Part A, TO FILL if unknown); Stage 4 gate table; log entries (v1
   run, v2 fix, extra round, mini eval, close-out); lessons learned:
   - A `/goal` condition must be checkable from the chat transcript ("every file re-read"
     is not, so the checker looped and used up the usage limit).
   - "Connection refused" on port 5432 means the database container is not running.
   - "Needs the document" is an answer-evaluation criterion, not a retrieval one.
   - Markdown table symbols counted as words made table quotes look too long.
   - Overlap at least as long as the longest quote means no quote is ever split.
   - A detector that flags real words measures the tokenizer, not the parser.
   - The title in every section_aware prefix makes all chunks look alike.
   Explain-aloud questions to add: "Why does a 40-word overlap guarantee that no 25-word
   quote is split, and what changes with a 10-word overlap?", "Why is 'needs the document' a
   check for answer evaluation but not for retrieval evaluation?", "Why did the title in the
   prefix make 'What does MTEB stand for?' fail?", and the unanswered "Why does the bootstrap
   resample documents, not questions?" if it is not already there.
   Open items for Stage 5: re-ingest the routed dev documents (new cache key); BM25 + RRF,
   reranker, parent expansion scored on the child chunk; prefix switch with 3 settings; a
   working glued-word detector; latency per query. For Stage 6: table-question pipeline v3
   (dev only, before test questions); test questions generated with the frozen pipeline.
