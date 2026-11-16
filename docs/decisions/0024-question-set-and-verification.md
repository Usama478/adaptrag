# 0024: Question set and machine verification

Status: Accepted (code written; numbers are `TO FILL` until the owner's run)
Date: 2026-11-16

## Context
Stage 4 needs enough questions to compare strategies: the 10 owner-verified paper_01
questions (note 0016) cover one document and were written from our own chunk text. The
owner decided on 2026-11-16 that the larger set is built by an LLM and verified by code
plus an LLM judge, with no manual checking. This note records how, and what that choice
costs in credibility.

## Decision
- **One pipeline, frozen after this stage.** `scripts/generate_questions.py` with the rules
  in `src/adaptrag/evaluation/passages.py` and `generation.py` (prompt version
  `stage4-v1`). Stage 6 builds the test questions with exactly this code and these
  prompts, untuned. Any change bumps the prompt version and gets its own note.
- **Passages come from the parser, not from chunks.** A prose passage is a window of
  consecutive paragraph blocks on one page inside one section (about 150-300 words, never
  a heading; reference and acknowledgement sections are skipped). A table passage is one
  table block cut to whole rows, with its "Table ..." caption when the next block is one.
  A multi-section question gets two prose passages at least 2 pages (or a third of the
  document) apart. Passages are spread over the document (one from each equal stretch)
  with a fixed seed (`stage4:<corpus id>:<round>`), so no strategy's boundaries are favoured.
- **About 14 drafts per document, up to 10 kept:** 8 factual, 4 table (fewer if the
  document has fewer usable tables; the missing ones become factual), 2 multi-section. If
  more than 10 pass, multi-section and table questions are kept first, because they are
  the rarer types and have their own targets.
- **The generator** (through the `LLM` seam, temperature 0) returns JSON: question, short
  answer, type, and one evidence quote per passage. The quote must be 8 to 25 words copied
  exactly. The upper limit is from the stage file; the lower limit is ours, because the
  relevance rule accepts half a quote (note 0025) and half of a very short quote would
  match almost anything.
- **Verification** (each step is a pure, unit-tested function; the script prints the
  rejection count per step):
  1. The quote is found in the parsed document text after `normalise()`.
  2. Evidence pages are computed by our code (every page that holds the quote), never
     taken from the LLM.
  3. More evidence locations: other pages whose text holds the answer string as whole
     words (at most one place per page, at most 5 per question). A separate judge call
     asks "does this passage alone answer the question?"; each yes becomes one more
     evidence entry (the answer with 15 words on each side, all from one block).
  4. A judge call with its own prompt sees the question, the answer and the quotes only:
     answerable from the evidence, one correct answer, needs the document. Any "no"
     rejects; so does a reply that is not the expected JSON.
  5. Lexical overlap: rejected when 80% or more of the question's content words appear in
     its quotes. The script prints the distribution.
  6. Near-duplicates within a document are removed (content-word Jaccard of 0.7 or more).
  Format rules run before the checks: invalid JSON, missing fields, a wrong type, a quote
  outside 8-25 words, and a question that names a page or section number are rejected.
  Cheap checks run first, so the judge is only paid for drafts that passed them (order:
  1, 2, 5, 6, 4, 3).
- **Stored per question:** id, question, answer, type, evidence (page + text),
  `verified: true`, `verification: "machine"`, the checks passed, the generator model and
  the prompt version. `load_questions()` reads these files and `paper_01.json` alike.
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

## Results (owner's run)
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
  before the answer itself (q01's quote ends at "a stack of"). Check 4 asks whether the
  evidence states the answer, so it may reject correct owner questions for that reason.
  The calibration line shows the reasons; read them before reading the count.
- A table quote must reproduce the row exactly, including every `|`. Drafts that fail
  this are rejected by check 1, which may leave few table questions.
- Pages that fell back to plain text (note 0018) have no table blocks, so they give no
  table questions.
- "Needs the document" is judged by a model that may know these famous papers; questions
  about well-known facts may be rejected, which shifts the set towards details.
- Types are named `factual`, `table` and `multi_section`; the owner set uses
  `two_section` for the same idea.
