"""The relevance rule of the mini evaluation (Stage 4, note 0025). Pure functions.

A chunk is relevant to an evidence entry if, after `normalise()` on both:
- it contains the whole evidence text, or
- it contains a run of consecutive evidence words at least half the evidence long. This
  covers evidence that a chunk boundary cut in two: whichever chunk holds the bigger half
  (or either one, for an even split) still counts.

A question is answered at rank r if the chunk at rank r is relevant to any of its
evidence entries.

Callers pass chunk text WITHOUT the section-aware heading prefix
(`strategies.section_aware.strip_heading_prefix`), so the prefix neither helps nor hurts.
"""

from collections.abc import Sequence

from adaptrag.evaluation.gate import normalise

# Results of `match_kind`.
WHOLE = "whole"
PARTIAL = "partial"

# Results of `evidence_status`: how a whole chunk set holds a question's evidence.
SPLIT = "split"
MISSING = "missing"


def min_run_words(evidence_word_count: int) -> int:
    """How many consecutive evidence words a chunk needs: half, rounded up."""
    return (evidence_word_count + 1) // 2


def match_kind(chunk_text: str, evidence_text: str) -> str | None:
    """WHOLE, PARTIAL, or None if the chunk is not relevant to this evidence entry."""
    chunk = normalise(chunk_text)
    evidence = normalise(evidence_text)
    if not evidence:
        # Blank evidence must never match everything.
        return None
    if evidence in chunk:
        return WHOLE

    words = evidence.split()
    run = min_run_words(len(words))
    # Spaces on both sides make the comparison word by word: "art" is not found in "part".
    padded_chunk = f" {chunk} "
    for start in range(len(words) - run + 1):
        if f" {' '.join(words[start : start + run])} " in padded_chunk:
            return PARTIAL
    return None


def is_relevant(chunk_text: str, evidence_texts: Sequence[str]) -> bool:
    """True if the chunk is relevant to at least one evidence entry."""
    return any(match_kind(chunk_text, evidence) is not None for evidence in evidence_texts)


def first_relevant_rank(chunk_texts: Sequence[str], evidence_texts: Sequence[str]) -> int | None:
    """The 1-based rank of the first relevant chunk in a ranked list, or None."""
    for rank, chunk_text in enumerate(chunk_texts, start=1):
        if is_relevant(chunk_text, evidence_texts):
            return rank
    return None


def evidence_status(chunk_texts: Sequence[str], evidence_texts: Sequence[str]) -> str:
    """How ALL chunks of a chunk set hold a question's evidence.

    WHOLE: some chunk contains some evidence entry whole.
    SPLIT: no chunk does, but some chunk holds at least half of one (cut by a boundary).
    MISSING: no chunk is relevant at all (the question cannot be answered by this set).
    """
    found_partial = False
    for chunk_text in chunk_texts:
        for evidence in evidence_texts:
            kind = match_kind(chunk_text, evidence)
            if kind == WHOLE:
                return WHOLE
            if kind == PARTIAL:
                found_partial = True
    return SPLIT if found_partial else MISSING
