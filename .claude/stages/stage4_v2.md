# Stage 4 fix: question pipeline v2 (code-only)

Branch: `stage4-mini-evaluation`. Rules: `.claude/AUTONOMOUS.md` (run nothing).
Owner's decision (2026-11-18): fix the question pipeline BEFORE any strategy is scored.
No strategy has been scored yet, so these changes cannot favour one. The prompt version
becomes `stage4-v2`; v2 is the version that gets frozen and reused untuned in Stage 6.

## The v1 run (real numbers from the owner's run, record them in note 0024 as history)
- Drafted 126, passed 25, kept 25 (paper_01 1, paper_02 1, paper_03 2, paper_04 3,
  paper_05 5, paper_06 3, paper_07 7, doc_08 2, paper_09 1).
- Rejections: quote_length 44, judge_no_document_needed 33, quote_not_found 10,
  lexical_overlap 9, missing_fields 3, invalid_json 1, judge_not_single_answer 1, others 0.
- Lexical overlap of the 68 drafts that reached check 5: 0.0-0.2: 7, 0.2-0.4: 14,
  0.4-0.6: 22, 0.6-0.8: 16, 0.8-1.0: 9 (rejected).
- Extra evidence locations: 10 confirmed of 17 candidates.
- Calibration on the 10 owner-verified paper_01 questions: accepted 1 of 10
  (judge_not_answerable 7, quote_not_found 1, lexical_overlap 1).
- Targets: 25/70, table 4/15, multi-section 6/10, all not met.
- Usage: 210 calls, 105,995 input and 9,693 output tokens.

## Why v1 failed, and the changes
1. **"Needs the document" is not a retrieval criterion.** Stage 4 measures whether the
   right chunk is retrieved, not whether an LLM knows the answer from training. Our dev
   papers are famous, so the judge rejected good questions (33). Change: the judge still
   answers `needs_document`, but it no longer rejects. Store it on each question as
   `answerable_without_document` (true when needs_document is false) and print the count.
   Stage 6 answer evaluation will use this label.
2. **The judge saw only the quote.** The owner's quotes are location markers that stop
   before the answer (q01 ends at "a stack of"), so 7 of 10 were "not answerable".
   Change: for each evidence entry the judge also sees the parsed block that contains the
   quote (if the quote spans blocks, the blocks joined), at most 300 words, cut around the
   quote. Same for generated questions and for calibration.
3. **Quote length counted Markdown symbols as words.** `word_count` counts `|` and divider
   cells, so table rows look far too long (paper_09: 9 of 14 rejected). Change: a new pure
   function `quote_words()` ignores tokens made only of `|`, `-`, `:` and similar Markdown
   table symbols. The limits stay 8 to 25 words.
4. **Short quotes are extended by code, not rejected.** A quote under 8 words is extended,
   deterministically, to the sentence that contains it (prose: split at `. `, `? `, `! `
   or a line break) or to the table row that contains it (tables). If the result is over
   25 words, trim it to 25 words centred on the original quote. Work on the normalised
   block text, so the stored evidence is always found by check 1. A quote still under 8
   words after extension is rejected as before.
5. **More table questions.** A document with at least 1 usable table gets
   min(4, 2 x usable tables) table drafts (v1 needed 4 usable tables). The near-duplicate
   check stays.
6. **20 drafts per document** (v1: 14): about 11 factual, up to 6 table (from change 5,
   factual fills the gap), and 3 multi-section. Keep at most 10 per document, as before.

Everything else stays as in v1: checks 1, 2, 3, 5 and 6, the lexical overlap limit (0.8),
the judge's other two answers (answerable, single answer) still reject, temperature 0,
fixed seed, one extra round allowed, and the targets (70 / 15 / 10). Do not change any of
them.

## Script behaviour
- Prompt version `stage4-v2` is written into every question file.
- If `eval/questions/generated/` holds a file with another prompt version, the script stops
  with a message naming the file and saying to delete it (it never mixes versions).
- The usage log and stage budget continue (the v1 tokens count towards the stage total).

## Tests
Update the v1 tests these changes affect and add tests for: `quote_words()` (pipes and
divider cells ignored), quote extension (prose sentence, table row, trim to 25 centred,
still too short is rejected, the result is found by check 1), the judge no longer
rejecting on needs_document but storing the label, the judge prompt containing the block
text, the table-slot rule, 20 drafts per document, and the version-mix guard. In the final
report state the exact number of new tests and the expected total (currently 517 passed).

## Docs
- Note 0024: a section "v1 run (history)" with the numbers above and the reasons for each
  change; the method section updated to v2; results stay `TO FILL from owner's run` (v2).
  Add the lexical-overlap observation: most questions share 40-80% of their content words
  with the evidence, so retrieval is easier than with real user questions.
- CLAUDE.md and PROGRESS.md: prompt version `stage4-v2`, the log entry for this fix, and
  the explain-aloud question: "Why is 'needs the document' a check for answer evaluation
  but not for retrieval evaluation?"
