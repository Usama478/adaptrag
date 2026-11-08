# 0022: Stage 3 gate

Status: Accepted
Date: 2026-11-09

## Context
Stage 3 replaced the parser (note 0018), added the strategy interface and registry (note
0019), the recursive and section-aware strategies (note 0020), and the profiler and router
(note 0021). Before Stage 4 measures anything, the routed pipeline must be shown to work end
to end on real data. This is a pass/fail check, not the benchmark (that is Stage 4).

## Decision
- **The gate is the checklist below,** each item backed by a test, a script or an
  end-to-end run.
- **The Stage 2 gate is run again on the routed paper_01,** with the rules of note 0016
  (at least 10 scored questions and at least 80% cited_ok). A pipeline change must not
  break what already passed.

## Result (owner's run)

| Item | Result |
| --- | --- |
| One test per router rule | `test_router.py` (17 tests) |
| Profile signals checked on dev | `scripts/profile_report.py` on 9 dev documents |
| Extensibility (one file + one registry line) | `test_strategy_extensibility.py` |
| Reason strings are sentences | e.g. "Chose section-aware chunking because 26 headings were found and 85% are numbered like a paper's outline." |
| Dev/test split in the repo | `eval/corpus.json` (note 0017) |
| Headings and tables detected | 185 headings, 49 tables on dev (note 0018) |
| Strategies differ | chunk report: 482 / 531 / 614 chunks (fixed_size / recursive / section_aware, note 0020) |
| Docker end to end | upload of paper_07 -> job succeeded -> `GET /documents/6` shows profile, section_aware, numbered_outline and reason -> `/query` top 3 from page 3 (scores 0.853, 0.849, 0.835) including the parameter table |
| All dev documents routed | 9/9 succeeded; section_aware 8, recursive 1 (doc_08); 31.2 s total including model load (`scripts/ingest_folder.py`) |
| Stage 2 gate on routed paper_01 | cited the right page 9/10, evidence in top-5 8/10, PASS |

**Latency and tokens (Stage 2 gate on routed paper_01).**
- Input tokens per answer: 983-1548 (Stage 2: 1567-1999).
- Answer time: mean 2.63 s, max 5.39 s (Stage 2: mean 1.79 s).

**Docker image.** 2.42 GB on disk, 539 MB content (Stage 2: 2.33 GB / 523 MB).

**Tests.** 379 passed, 7 deselected.

## Findings and known limits
- **q02.** Correct answer (512) citing page 5; the gate's evidence lists only page 3, and
  the fact appears on both pages. Stage 4 questions must allow several evidence locations.
- **q08.** retrieved_ok is false because the chunk text has a glued word ("theWMT") while
  the evidence string came from parser 0.1.0 text.
- **Glued words still occur with parser 0.2.0.** The Step 4 check only looked for the 5
  Stage 2 tokens (corrected in notes 0018 and 0020). Stage 4 needs a general detector.
- **Chunks over 512 tokens after routing:** paper_01 1/56, paper_02 2/94, paper_05 1/30,
  MTEB 36/160 (22%). Table rows turn into far more tokens than words, so the end of large
  table chunks is not embedded. Candidate Stage 4 fix: measure table chunks in tokens.
- **9/10 vs Stage 2's 10/10 says nothing about strategies.** It is one paper, 10 questions,
  one run: no conclusion about which strategy is better. That is Stage 4.

## What Stage 4 changes
- Questions may list several evidence locations.
- A general glued-word detector replaces the check for the 5 Stage 2 tokens.
- Candidate fix for the unembedded table text: measure table chunks in tokens.
- Strategies are compared on the benchmark, not on this gate.
