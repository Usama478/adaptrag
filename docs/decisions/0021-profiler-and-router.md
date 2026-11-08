# 0021: Document profiler and router

Status: Accepted
Date: 2026-11-09

## Context
The router picks a chunking strategy per document from measured signals, so that a
document's own structure decides how it is cut. The rules come from dev documents only
(note 0017); test documents are never used to choose rules or thresholds.

## Decision
- **Profile signals** (`routing/profile.py`, measured from the parsed document):
  `page_count`, `words`, `headings`, `numbered_heading_share`, `headings_per_1k_words`,
  `tables`, `table_word_share`, `fallback_pages`, `fallback_page_share`, `words_per_page`.
- **Dev profiles** (`scripts/profile_report.py`, the 9 dev documents):

  | Document | headings | numbered_heading_share | headings_per_1k | table_word_share | fallback_page_share |
  | --- | --- | --- | --- | --- | --- |
  | doc_08 (Berkshire) | 9 | 0.111 | 2.506 | 0.064 | 0.0 |
  | paper_01 | 26 | 0.846 | 4.341 | 0.053 | 0.0 |
  | paper_02 | 23 | 0.609 | 2.419 | 0.09 | 0.0 |
  | paper_03 | 15 | 0.6 | 3.332 | 0.022 | 0.0 |
  | paper_04 | 35 | 0.514 | 3.601 | 0.044 | 0.0 |
  | paper_05 | 16 | 0.875 | 5.036 | 0.0 | 0.0 |
  | paper_06 | 18 | 0.722 | 2.542 | 0.032 | 0.0 |
  | paper_07 | 12 | 0.667 | 4.391 | 0.033 | 0.0 |
  | paper_09 | 31 | 0.419 | 1.801 | 0.271 | 0.125 |

- **Rule v1** (`routing/router.py`, ordered, first match wins):
  1. `headings < 5` -> `recursive` (`few_headings`)
  2. `numbered_heading_share < 0.3` -> `recursive` (`unnumbered_headings`)
  3. otherwise -> `section_aware` (`numbered_outline`)
- **Why the numbered share.** Every dev document clears the floor of 5 headings. Heading
  count does not separate them in a useful way: doc_08 has the fewest (9), only 3 fewer than
  paper_07 (12). Headings per 1k does not separate them at all: paper_02 and paper_09 have
  lower values than doc_08. The numbered share does: papers 0.419-0.875, Berkshire 0.111.
  Any threshold from about 0.15 to 0.40 gives the same split, so 0.3 is not finely tuned.
- **`fixed_size` is the control** in the benchmark, so the router never chooses it.
- **The decision is stored on the document** (`profile`, `strategy`, `route_rule`,
  `route_reason`; migration 8f3a6c1e9d47), in the same transaction as the chunks and vectors.
- **`DEFAULT_STRATEGY` removed.** Routes and scripts use the document's own strategy.
- **Documents without a usable strategy** (none stored, or a name no longer registered)
  answer 409 needs re-ingestion on `/query` and `/answer`. An unregistered name also logs a
  WARNING with the document id and the name.

## Router diagnostics on dev
`section_aware` 8 of 9 (89%), `recursive` 1 of 9 (11%). The one `recursive` document is
doc_08, by the numbered-share rule; no dev document has fewer than 5 headings.

## Rejected
- **Heading count alone:** all 9 dev documents would get `section_aware`.
- **A table rule:** only one dev document (MTEB, paper_09) is table-heavy, so a rule for it
  would be overfitting.

## Known limits
- Rule v1 is a hypothesis to test in Stage 4. It was built from 9 dev documents, of which 8
  are papers.
- Numbered-heading false positives: "2024 Annual Report" and "E. coli" count as numbered.
  IEEE "III-A" is not matched.
- A share just below 0.3 can display as "30%" in the reason. Routing compares the profile's
  value (3 decimals), not the displayed percentage.
- The "very short" rule is deferred to Stage 4.
- Table and fallback signals are recorded but unused.
- Documents ingested before routing need re-ingestion.
