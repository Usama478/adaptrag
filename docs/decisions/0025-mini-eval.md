# 0025: Mini evaluation design and results

Status: Accepted (owner's run done 2026-11-18; results and interpretation below)
Date: 2026-11-16, results filled 2026-11-18

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

## Results (owner's run, `eval/results/stage4/summary.md`)
Embedding model BAAI/bge-small-en-v1.5, parser 0.2.0, top 10, exact search, 9 dev
documents, 90 machine-verified questions. Intervals: cluster bootstrap over documents,
10,000 resamples, seed 20261116.

| row | R@1 | R@5 | R@10 | MRR | 95% R@5 | 95% MRR | N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_size | 0.511 | 0.867 | 0.944 | 0.659 | [0.778, 0.956] | [0.547, 0.760] | 90 |
| recursive | 0.567 | 0.844 | 0.900 | 0.691 | [0.756, 0.933] | [0.592, 0.790] | 90 |
| section_aware | 0.500 | 0.844 | 0.911 | 0.653 | [0.744, 0.933] | [0.590, 0.719] | 90 |
| router (rule v1) | 0.489 | 0.844 | 0.911 | 0.647 | [0.744, 0.933] | [0.588, 0.708] | 90 |
| best single (fixed_size) | 0.511 | 0.867 | 0.944 | 0.659 | [0.778, 0.956] | [0.547, 0.760] | 90 |
| oracle (upper bound) | 0.600 | 0.900 | 0.933 | 0.721 | [0.822, 0.978] | [0.625, 0.814] | 90 |

Best single strategy and oracle are chosen on these same dev numbers, so both are
optimistic.

- **Table token budget variants:** recursive+tb512 R@1 0.556, R@5 0.844, R@10 0.889, MRR
  0.682; section_aware+tb512 R@1 0.500, R@5 0.844, R@10 0.922, MRR 0.652.
- **Owner-verified paper_01 set (10 questions, one document, no interval):** fixed_size R@5
  0.900, MRR 0.814; recursive R@5 1.000, MRR 0.833; section_aware (routed) R@5 0.900, MRR
  0.912.
- **Best single strategy:** fixed_size (Recall@5 0.867). By MRR the best is recursive
  (0.691): the two measures disagree.
- **Ties:** recursive and section_aware are both 0.844 on Recall@5. Recursive alone finds
  4 questions (paper_01_r0_d12, paper_03_r0_d02, paper_03_r1_d18, paper_09_r0_d10);
  section_aware alone finds 4 other questions (paper_02_r0_d03, paper_04_r0_d10,
  paper_06_r1_d19, paper_07_r0_d14).
- **Paired outcomes (top 5, only first / only second):** fixed_size vs recursive 7 / 5;
  fixed_size vs section_aware 6 / 4; recursive vs section_aware 4 / 4.
- **Per question type, hits@5 (fixed_size / recursive / section_aware / router):** factual
  (60) 51 / 51 / 50 / 50; table (6) 4 / 2 / 3 / 3; multi_section (24) 23 / 23 / 23 / 23.
- **Per document:** the routed strategy is section_aware for 8 documents and recursive for
  doc_08. The best strategy by hits@5 differs per document (recursive for paper_01,
  paper_03, paper_05 and paper_09; fixed_size for paper_04, paper_06 and paper_07;
  section_aware for paper_02 and doc_08); the full table is in `summary.md`.
- **Evidence split / missing:** 0 / 0 for every variant (fixed_size, recursive,
  section_aware and both budget variants).
- **Glued-word candidates:** paper_01 331, paper_02 499, paper_03 199, paper_04 542,
  paper_05 94, paper_06 160, paper_07 186, doc_08 71, paper_09 1632. Questions whose
  evidence contains a candidate: 51 of 90 machine-verified, 3 of 10 owner-verified.
- **Table token budget:** table chunks over 512 tokens, recursive 25/125 without the
  budget and 0/151 with it; section_aware 30/126 and 0/156. Table hits@5: recursive 2/6
  and 2/6; section_aware 3/6 and 3/6. Outcome: **ADOPT**.
- **Very short:** no dev document is under 2,000 words (smallest: paper_07, 2733 words,
  5 pages). Outcome: **NO RULE**.
- **Router rule v1:** the routed strategy loses on 4 documents (paper_01, paper_03,
  paper_07, paper_09). Outcome: **KEEP v1**.
- **Top-1 scores and threshold:** answerable questions (n 90) min 0.548, median 0.728, max
  0.901; off-topic questions (n 180) min 0.320, median 0.443, max 0.607. At 0.55, 1
  answerable question falls below and 4 off-topic questions pass. The script's proposal,
  0.567, puts 268 of 270 scores on the right side.

## Interpretation
- **No strategy separates on dev.** Recall@5 is 0.867 (fixed_size), 0.844 (recursive),
  0.844 (section_aware) and 0.844 (router); the intervals overlap almost entirely.
  Recall@5 and MRR disagree on the best strategy (fixed_size vs recursive). The paired wins
  cancel (4 to 7 questions each way). A difference of one or two questions out of 90 is
  noise.
- **Routing has little to gain on dev.** The oracle (best strategy per document) adds only
  3 questions over the best single strategy (0.900 vs 0.867), and it is chosen on the same
  data, so it is an upper bound. Router rule v1 is KEEP v1 under the rule written before
  the run, but dev data does not support it: the router is not better than any fixed
  strategy here. Revisit it after Stage 5.
- **The questions are easy.** Recall@5 is 84-87% for every strategy and most questions
  share much of their wording with their evidence (note 0024). That compresses the
  differences between strategies.
- **Finding: the document title in the section-aware prefix.** Section-aware puts the full
  heading path on every chunk, and the path starts with the document title. "What does MTEB
  stand for?" (paper_09) fails under the router because every chunk starts with "MTEB:
  Massive Text Embedding Benchmark", so the chunk that answers looks like all the others;
  fixed_size and recursive find it. This is the hypothesis for the Stage 5 prefix switch
  (full prefix / no prefix / prefix without the title).
- **Failure patterns (14 questions fail under the router):**
  - Invalid questions (at least 2, note 0024): "highest score recorded in the first
    passage" and "number of layers in ResNet-34".
  - Exact numbers and table cells (for example the baseline+++ mAP, the coCondenser
    average, BERT-base on MNLI): dense search is weak at them; BM25 in Stage 5 is the
    candidate fix.
  - A reference-list question (paper_01, the Penn Treebank authors): the References
    heading was not detected, so the references sit under "7 Conclusion".
  - The title prefix (MTEB, above).
- **Split evidence is 0 for every variant.** Chunks overlap by 40 words and quotes are at
  most 25 words, so every quote fits whole in some chunk. The "half of a quote" part of the
  relevance rule was never needed in this run.
- **Glued-word detector: inconclusive.** Its most frequent candidates are real words that
  the tokenizer splits into pieces (Transformer, decoder, dataset, ImageNet), so it measures
  the tokenizer's vocabulary, not the parser's glued words. A better method is a Stage 5
  item (BM25 is sensitive to glued words).
- **Table token budget: ADOPT.** Table chunks over 512 tokens fall from 25 to 0
  (recursive) and from 30 to 0 (section_aware), and table hits stay at 2/6 and 3/6. The
  evidence is weak (6 table questions), but text past 512 tokens is never embedded, so the
  budget is right in principle. The default became 512 in the Stage 4 close-out.
- **Very short rule: NO RULE.** No dev document is under 2,000 words.
- **Threshold: kept at 0.55 (owner's decision).** The proposal of 0.567 was fitted on the
  same data and fixes 3 cases. An off-topic question that passes the threshold reaches the
  LLM, which answers NOT_FOUND, while a higher threshold could refuse real questions: the
  lowest answerable score was 0.548.

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
