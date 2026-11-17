# 0024: Question set and machine verification

Status: Accepted (v2 code written 2026-11-18; v2 numbers are `TO FILL` until the owner's run)
Date: 2026-11-16 (v1), revised 2026-11-18 (v2)

## v1 run (history)
The first pipeline (prompt version `stage4-v1`) was run by the owner on 2026-11-18, before
any strategy was scored. It missed every target. Real numbers from that run:
- Drafted 126, passed 25, kept 25 (paper_01 1, paper_02 1, paper_03 2, paper_04 3,
  paper_05 5, paper_06 3, paper_07 7, doc_08 2, paper_09 1).
- Rejections: quote_length 44, judge_no_document_needed 33, quote_not_found 10,
  lexical_overlap 9, missing_fields 3, invalid_json 1, judge_not_single_answer 1, the
  other reasons 0.
- Lexical overlap of the 68 drafts that reached check 5: 0.0-0.2: 7, 0.2-0.4: 14,
  0.4-0.6: 22, 0.6-0.8: 16, 0.8-1.0: 9 (rejected).
- Extra evidence locations: 10 confirmed of 17 candidates.
- Calibration on the 10 owner-verified paper_01 questions: accepted 1 of 10
  (judge_not_answerable 7, quote_not_found 1, lexical_overlap 1).
- Targets 25 / 70, table 4 / 15, multi-section 6 / 10: none met.
- Usage: 210 calls, 105,995 input and 9,693 output tokens.

Why v1 failed, and what v2 changes (decided by the owner on 2026-11-18, before any
strategy was scored, so none of the changes can favour a strategy):
1. **"Needs the document" is not a retrieval criterion** (33 rejections). Stage 4 measures
   whether the right chunk is retrieved, not whether a model knows the answer from
   training. The dev papers are famous, so the judge rejected good questions. The judge
   still answers `needs_document`, but it no longer rejects: the question stores
   `answerable_without_document` (true when needs_document is false), and the script prints
   the count. Stage 6 answer evaluation will use the label.
2. **The judge saw only the quote** (calibration: 7 of 10 owner questions "not
   answerable"). The owner's quotes are location markers that stop before the answer (q01
   ends at "a stack of"). Now the judge also sees, for each quote, the parsed block that
   contains it (a quote spanning blocks gets those blocks joined, at most 3), at most 300
   words, cut around the quote. Same for generated questions and for calibration.
3. **Quote length counted Markdown symbols as words** (44 rejections; paper_09: 9 of 14).
   `quote_words()` ignores tokens made only of `|`, `-`, `:`, `=`, `+`, `_`. The limits
   stay 8 to 25 words.
4. **Short quotes are extended by code, not rejected.** A quote under 8 words becomes the
   sentence (prose: cut after `. `, `? `, `! ` or at a line break) or the table row that
   contains it; over 25 words it is cut to 25 centred on the original quote. A quote still
   under 8 words after that, or not inside one sentence or row, is rejected as before. The
   text keeps the document's own words, so check 1 always finds it.
5. **More table questions.** A document with at least 1 usable table gets
   min(4, 2 x usable tables) table drafts (v1 needed 4 usable tables). A table used twice
   tells the generator that a question was already drafted from it (with temperature 0
   the same message would give the same question twice). The near-duplicate check stays.
6. **20 drafts per document** (v1: 14): 3 multi-section, the table drafts of change 5, and
   factual drafts for the rest. At most 10 questions are kept per document, as before.

Unchanged: checks 1, 2, 3, 5 and 6, the lexical overlap limit (0.8), the judge's other two
answers (answerable, single answer) still reject, temperature 0, fixed seed, one extra
round, targets 70 / 15 / 10.

An observation about the v1 overlap distribution: most questions share 40-80% of their
content words with their evidence (38 of 68 drafts in the 0.4-0.8 buckets). Retrieval on
these questions is easier than on real user questions, which use their own words. This
adds to the optimism already listed under "The credibility cost".

## Context
Stage 4 needs enough questions to compare strategies: the 10 owner-verified paper_01
questions (note 0016) cover one document and were written from our own chunk text. The
owner decided on 2026-11-16 that the larger set is built by an LLM and verified by code
plus an LLM judge, with no manual checking. This note records how, and what that choice
costs in credibility.

## Decision
- **One pipeline, frozen after this stage.** `scripts/generate_questions.py` with the rules
  in `src/adaptrag/evaluation/passages.py` and `generation.py` (prompt version
  `stage4-v2`; v2 is the version that is frozen). Stage 6 builds the test questions with
  exactly this code and these prompts, untuned. Any change bumps the prompt version and
  gets its own note. The script stops if a file in `eval/questions/generated/` has another
  prompt version (it never mixes versions); delete such files to regenerate.
- **Passages come from the parser, not from chunks.** A prose passage is a window of
  consecutive paragraph blocks on one page inside one section (about 150-300 words, never
  a heading; reference and acknowledgement sections are skipped). A table passage is one
  table block cut to whole rows, with its "Table ..." caption when the next block is one.
  A multi-section question gets two prose passages at least 2 pages (or a third of the
  document) apart. Passages are spread over the document (one from each equal stretch)
  with a fixed seed (`stage4:<corpus id>:<round>`), so no strategy's boundaries are favoured.
- **20 drafts per document, up to 10 kept:** 3 multi-section, min(4, 2 x usable tables)
  table drafts, and factual drafts for the rest (17 minus the table drafts). If more than
  10 pass, multi-section and table questions are kept first, because they are the rarer
  types and have their own targets.
- **The generator** (through the `LLM` seam, temperature 0) returns JSON: question, short
  answer, type, and one evidence quote per passage. The quote must be 8 to 25 words
  (`quote_words()`: table symbols do not count) copied exactly; a shorter quote is first
  extended by code (v2). The upper limit is from the stage file; the lower limit is ours,
  because the relevance rule accepts half a quote (note 0025) and half of a very short
  quote would match almost anything.
- **Verification** (each step is a pure, unit-tested function; the script prints the
  rejection count per step):
  1. The quote is found in the parsed document text after `normalise()`.
  2. Evidence pages are computed by our code (every page that holds the quote), never
     taken from the LLM.
  3. More evidence locations: other pages whose text holds the answer string as whole
     words (at most one place per page, at most 5 per question). A separate judge call
     asks "does this passage alone answer the question?"; each yes becomes one more
     evidence entry (the answer with 15 words on each side, all from one block).
  4. A judge call with its own prompt sees the question, the answer, the quotes and the
     block around each quote: answerable from the evidence, one correct answer, needs the
     document. A "no" to the first two rejects, and so does a reply that is not the
     expected JSON. The third answer is only stored (`answerable_without_document`, v2).
  5. Lexical overlap: rejected when 80% or more of the question's content words appear in
     its quotes. The script prints the distribution.
  6. Near-duplicates within a document are removed (content-word Jaccard of 0.7 or more).
  Format rules run before the checks: invalid JSON, missing fields, a wrong type, a quote
  outside 8-25 words, and a question that names a page or section number are rejected.
  Cheap checks run first, so the judge is only paid for drafts that passed them (order:
  1, 2, 5, 6, 4, 3).
- **Stored per question:** id, question, answer, type, evidence (page + text),
  `verified: true`, `verification: "machine"`, the checks passed, the generator model, the
  prompt version and `answerable_without_document`. `load_questions()` reads these files and `paper_01.json` alike.
- **Calibration.** The script runs checks 1, 5 and 4 on the 10 owner-verified paper_01
  questions with the same functions and judge prompt, and prints how many it accepts.
- **Targets** (printed as met or not met): at least 70 accepted questions, at least 15
  table questions, at least 10 multi-section questions. If one is missed, the script
  offers exactly one more round (`--extra-round`, new passages, same checks). The checks
  are never loosened; a target still missed is reported as missed.
- **Cost control.** Every call is logged to `eval/llm_usage.jsonl`; the script refuses to
  start if its estimate would cross the stage budget (note 0023).

## The credibility cost, stated plainly
- **No human checked these questions.** "Verified" here means "passed the six checks",
  not "a person read it against the PDF". A wrong answer that the judge accepts stays in.
- **The questions are worded from the parser's text.** The generator reads exactly the
  text our parser produced, so its questions share vocabulary and parsing quirks with the
  chunks being searched. That favours retrieval in general (all strategies alike) and
  makes the absolute numbers optimistic. Check 5 only limits the most direct copying.
- **The judge is the same model as the generator** (gpt-4o-mini). It tends to accept what
  it wrote itself; a different, stronger judge would be a better check.
- **Check 3 can add a wrong location** (the judge says yes to a passage that only
  mentions the same number), which makes a miss look like a hit. It can also miss a real
  second location when the answer is worded differently there.
- **The comparison between strategies is more trustworthy than the absolute level:** all
  strategies answer the same questions under the same rule, so the biases above mostly
  cancel in a difference.
- The owner-verified paper_01 set is kept as a separate row in every results table for
  this reason.

## Results (owner's run, v2)
- Answerable without the document (judge label, of the kept questions): TO FILL from
  owner's run: the "answerable without the document" line.
- Drafted / passed / kept: TO FILL from owner's run: totals printed by
  `generate_questions.py`.
- Rejections per check: TO FILL from owner's run: the "rejections per check" block.
- Lexical overlap distribution: TO FILL from owner's run: the five buckets.
- Extra evidence locations (confirmed / candidates): TO FILL from owner's run.
- Targets (70 / 15 / 10): TO FILL from owner's run: met or not met, and whether the extra
  round was used.
- **Verifier's acceptance on the owner set:** TO FILL from owner's run: "accepted N of 10"
  and the rejection reasons.
- Tokens used: TO FILL from owner's run: the stage total printed at the end.

## Known limits
- The owner's paper_01 evidence quotes were written as location markers; several stop
  before the answer itself (q01's quote ends at "a stack of"). In v1 this made the judge
  reject 7 of 10; v2 shows it the block around each quote. The calibration line shows the
  reasons; read them before reading the count.
- The context cut keeps tokens separated by single spaces, so a table's line breaks are
  gone in what the judge reads (its `|` symbols still mark the cells).
- A table quote must reproduce the row exactly, including every `|`. Drafts that fail
  this are rejected by check 1, which may leave few table questions.
- Pages that fell back to plain text (note 0018) have no table blocks, so they give no
  table questions.
- "Needs the document" is judged by a model that may know these famous papers. In v1 this
  rejected questions about well-known facts; v2 keeps them and labels them
  (`answerable_without_document`), so retrieval is scored on them and answer evaluation
  can exclude them.
- Types are named `factual`, `table` and `multi_section`; the owner set uses
  `two_section` for the same idea.
