# 0017: Evaluation corpus and frozen dev/test split

Status: Accepted
Date: 2026-11-08

## Context
The headline claim of this project is that routing beats any single fixed strategy. That
claim only means something if it is measured on documents the router's rules were not
written against. Stage 3 is about to write those rules, so the documents they will be
judged on have to be fixed first. Until now the repo had 5 dev samples and no test set.

## Decision
- **9 dev and 12 test documents**, listed in `eval/corpus.json` (id, split, category,
  title, path, source URL, SHA-256).
- **Split by document structure.** Four categories: `sectioned`, `short`, `table_heavy`,
  `poorly_structured`. Each category appears in both splits; the test set has exactly 3
  documents per category.
- **Frozen before any router rule exists.**
- **Dev files in `data/samples/`, test files in `data/test_set/`.** `data/` is gitignored.
- **PDFs are never committed (copyright).** The manifest stores the source URL and the
  SHA-256 of the file, so a run can show it used the same bytes.
- **Test documents are never opened, inspected or used for tuning** until the final
  evaluation. `scripts/fetch_corpus.py` parses them only to print page and word counts;
  it prints no text from them.
- **Validation is pure code.** `src/adaptrag/evaluation/corpus.py` (`parse_corpus`,
  `load_corpus`) checks the manifest without network or PDF access, tested in
  `tests/test_corpus.py`. Among other rules, the path of a test entry must be under
  `data/test_set/`, so a file cannot silently sit in the wrong split.
- **`scripts/fetch_corpus.py`** downloads missing files with `urllib.request` (no new
  dependency), keeps arXiv requests at least 3 seconds apart, writes through a `.part`
  file so no half files remain, records a missing SHA-256, and never overwrites one that
  is already recorded. It has no retries: re-running it is the retry, because it skips
  files that exist.

## Purpose
The test set measures whether the system generalizes to documents it has not seen. Its
existence also keeps the router honest: the router stays a few generic rules, not tuning
per document.

## Rejected
- **Collecting test documents later:** documents picked after seeing how the router
  behaves invite selection bias, even unintentionally.
- **Diversity by subject:** papers on different topics share the same layout, so the
  router would never face a different decision.
- **A random split:** dev must contain the 5 existing samples and every category, which
  a random draw does not guarantee.
- **Committing the PDFs:** copyright, and this is a public repo.

## Known limits
- 9 dev documents are few. Router rules must stay few as well.
- Categories were assigned from document type and title, not from detailed inspection.
- arXiv URLs are unversioned. A new version changes the file: the script reports a hash
  mismatch instead of silently using it.
- paper_04's source is confirmed (identical hash to a fresh arXiv download); paper_05's
  source comes from its arXiv stamp and is version-pinned (v1).
- The assistant that chose the documents already knows some of these papers from
  training. The choice was made by category only, with no tuning.

## Lesson
The owner's paper_04 sample turned out to be BERT. A fresh arXiv download added as
paper_06 had the identical SHA-256, so the hash revealed a duplicate that the file names
hid (and confirmed paper_04's source). paper_06 was replaced with Batch Normalization
(1502.03167), and validation now rejects duplicate hashes.

BERT was then relabelled sectioned (mainly sections with some tables; the table_heavy
label came from its file name), so the dev set had no truly table-heavy document while
the test set had three. MTEB (2210.07316), a paper made largely of results tables, was
added as dev table_heavy. The test set did not change; freezing protects the test set,
and no router rule existed yet.

## Pages and word counts
Printed by scripts/fetch_corpus.py on every run; not copied here, so this note cannot go
stale.
