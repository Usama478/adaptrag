# 0026: Stage 4 gate

Status: Accepted
Date: 2026-11-18

## Context
Stage 4 added the machine-verified question set (note 0024) and the mini evaluation (note
0025), and measured three known issues from note 0022. Before Stage 5 builds on it, the
stage must be shown to work end to end. This is a pass/fail check of the evaluation
machinery, not a claim that routing works: the results say it does not separate from the
fixed strategies on dev.

## Decision
- **The gate is the checklist below,** each item backed by a test, a script or the
  owner's run. One command prints the results table: `python scripts/mini_eval.py` (no
  arguments, no LLM calls, writes `eval/results/stage4/`).
- **The gate does not require that the router wins.** It requires that the questions,
  metrics and rules ran, that every rule printed its outcome, and that the results are
  reported as they are.

## Result (owner's run)

| Item | Result |
| --- | --- |
| Question set built and machine-verified | 90 questions on 9 dev documents, 10 each (note 0024); prompt version `stage4-v2` |
| Question targets | accepted 90 (target 70: met), multi-section 24 (target 10: met), table 6 (target 15: **not met**) |
| Verifier calibrated on the owner set | accepted 6 of 10 (v1: 1 of 10); strict, and it lets some bad questions through (note 0024) |
| Metrics and bootstrap | Recall@1/5/10 and MRR with a cluster bootstrap over documents (`test_metrics.py`, `test_analysis.py`) |
| Rules fixed before the run, each printed its outcome | table budget **ADOPT**, very short **NO RULE**, router **KEEP v1** (note 0025) |
| Strategies differ | chunks per strategy: paper_01 43 / 45 / 56, doc_08 52 / 56 / 57 (fixed_size / recursive / section_aware) |
| Reproducible | one command, no clock in the output (`scripts/mini_eval.py`) |
| Cost recorded | Stage 4 total 442,030 input and 37,375 output tokens, about $0.09 (`eval/llm_usage.jsonl`) |
| Tests | 543 passed, 7 deselected (after the table budget became the default) |

**Gate: PASS.**

## Findings
- **Recursive and section_aware tie on Recall@5 (0.844 each).** The tie is not "the same
  strategy": each finds 4 questions the other misses (recursive: paper_01_r0_d12,
  paper_03_r0_d02, paper_03_r1_d18, paper_09_r0_d10; section_aware: paper_02_r0_d03,
  paper_04_r0_d10, paper_06_r1_d19, paper_07_r0_d14). Their paired outcome is 4 against 4.
- **One failure explained in full: "What does MTEB stand for?" (paper_09).** The router
  chose section_aware, which writes the heading path on every chunk, and the path starts
  with the document title "MTEB: Massive Text Embedding Benchmark". Every chunk of the
  document therefore begins with the words the question asks about, so the chunk that
  states the expansion does not stand out: the router's top 3 are chunks from pages 2, 3-4
  and 4 with scores 0.682, 0.671 and 0.668, and the first relevant chunk is not in the top
  10. fixed_size and recursive, without the prefix, find it in the top 5. The Stage 5
  prefix switch (full prefix / no prefix / prefix without the title) tests this.
- **The router is not better than the fixed strategies on dev.** Router 0.844 against
  best single 0.867 on Recall@5; the oracle is 0.900. The questions are easy and the
  intervals overlap (note 0025).
- **Two of the 14 router failures are invalid questions** (note 0024). They stay in.

## What Stage 5 changes
- Re-ingest the routed dev documents: the table token budget is now in the params of
  recursive and section_aware, so their cache key changed.
- BM25 with reciprocal rank fusion, a reranker, parent expansion scored on the child chunk.
- A prefix switch with three settings, to separate boundaries from the prefix.
- Revisit router rule v1 after these, and replace the glued-word detector.
