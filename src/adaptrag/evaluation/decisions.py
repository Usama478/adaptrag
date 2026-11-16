"""Decision rules of Stage 4, fixed BEFORE any run (stage file section 4, note 0025).

`scripts/mini_eval.py` only prints what these pure functions return, so nothing can be
tuned after seeing the results.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

ADOPT = "ADOPT"
KEEP = "KEEP"

# Table token budget: ADOPT only if table chunks over the limit drop to 0 AND Recall@5 on
# table questions drops by no more than this many questions, for every strategy tried.
MAX_TABLE_HITS_LOST = 1

# "Very short" rule: none unless at least this many dev documents are under this size.
VERY_SHORT_WORDS = 2000
VERY_SHORT_MIN_DOCUMENTS = 2
NO_SHORT_RULE = "NO RULE"
CONSIDER_SHORT_RULE = "CONSIDER RULE"

# Router: propose a rule v2 only if the routed strategy loses on at least this many
# documents AND one profile threshold separates those documents from the others.
ROUTER_MIN_LOSSES = 3
KEEP_V1 = "KEEP v1"
BELOW = "below"
ABOVE = "above"

# Two strategies within this much on Recall@5 are reported as a TIE.
TIE_TOLERANCE = 0.01
_FLOAT_SLACK = 1e-9


@dataclass(frozen=True, slots=True)
class BudgetComparison:
    """One strategy without and with the table token budget.

    over_limit_*: table chunks longer than the model's token limit.
    table_hits_*: table questions with a relevant chunk in the top 5.
    """

    strategy: str
    over_limit_without: int
    over_limit_with: int
    table_hits_without: int
    table_hits_with: int


def table_budget_decision(comparisons: Sequence[BudgetComparison]) -> str:
    """ADOPT or KEEP for the table token budget; KEEP when there is nothing to compare."""
    if not comparisons:
        return KEEP
    for comparison in comparisons:
        if comparison.over_limit_with != 0:
            return KEEP
        if comparison.table_hits_without - comparison.table_hits_with > MAX_TABLE_HITS_LOST:
            return KEEP
    return ADOPT


def very_short_documents(words_by_document: Mapping[str, int]) -> list[str]:
    """Ids of the documents under VERY_SHORT_WORDS words, sorted."""
    return sorted(name for name, words in words_by_document.items() if words < VERY_SHORT_WORDS)


def very_short_decision(words_by_document: Mapping[str, int]) -> str:
    """CONSIDER RULE only if enough documents are very short; otherwise NO RULE."""
    if len(very_short_documents(words_by_document)) >= VERY_SHORT_MIN_DOCUMENTS:
        return CONSIDER_SHORT_RULE
    return NO_SHORT_RULE


@dataclass(frozen=True, slots=True)
class DocumentOutcome:
    """One document: the strategy the router chose and each strategy's Recall@5 hits."""

    document_id: str
    routed: str
    hits_by_strategy: Mapping[str, int]


def routed_loses(outcome: DocumentOutcome) -> bool:
    """True if some strategy answered more questions in the top 5 than the routed one."""
    return max(outcome.hits_by_strategy.values()) > outcome.hits_by_strategy[outcome.routed]


@dataclass(frozen=True, slots=True)
class Threshold:
    """A profile field and a value with every losing document on one side of it."""

    field: str
    value: float
    losing_side: str

    def describe(self) -> str:
        sign = "<" if self.losing_side == BELOW else ">"
        return f"{self.field} {sign} {self.value:g}"


def separating_threshold(
    profiles: Mapping[str, Mapping[str, float]], losing: Collection[str]
) -> Threshold | None:
    """The first profile field (alphabetical) that puts all losing documents on one side.

    The value is halfway between the two groups. None if no single field separates them,
    or if one of the groups is empty.
    """
    losers = [name for name in sorted(profiles) if name in losing]
    others = [name for name in sorted(profiles) if name not in losing]
    if not losers or not others:
        return None
    fields = sorted(set.intersection(*(set(profile) for profile in profiles.values())))
    for field in fields:
        losing_values = [profiles[name][field] for name in losers]
        other_values = [profiles[name][field] for name in others]
        if max(losing_values) < min(other_values):
            value = (max(losing_values) + min(other_values)) / 2
            return Threshold(field=field, value=value, losing_side=BELOW)
        if min(losing_values) > max(other_values):
            value = (min(losing_values) + max(other_values)) / 2
            return Threshold(field=field, value=value, losing_side=ABOVE)
    return None


@dataclass(frozen=True, slots=True)
class RouterDecision:
    """KEEP v1, or a candidate threshold for a rule v2 (never applied in Stage 4)."""

    verdict: str
    losing_documents: tuple[str, ...]
    threshold: Threshold | None


def router_decision(
    outcomes: Sequence[DocumentOutcome], profiles: Mapping[str, Mapping[str, float]]
) -> RouterDecision:
    """Apply the router rule: enough losses AND one separating profile threshold."""
    losing = tuple(sorted(outcome.document_id for outcome in outcomes if routed_loses(outcome)))
    if len(losing) < ROUTER_MIN_LOSSES:
        return RouterDecision(verdict=KEEP_V1, losing_documents=losing, threshold=None)
    threshold = separating_threshold(profiles, losing)
    if threshold is None:
        return RouterDecision(verdict=KEEP_V1, losing_documents=losing, threshold=None)
    return RouterDecision(
        verdict=f"CANDIDATE v2: {threshold.describe()}",
        losing_documents=losing,
        threshold=threshold,
    )


def tied_pairs(
    recall_by_strategy: Mapping[str, float], tolerance: float = TIE_TOLERANCE
) -> list[tuple[str, str]]:
    """Every pair of strategies whose Recall@5 differ by at most the tolerance."""
    names = sorted(recall_by_strategy)
    return [
        (first, second)
        for position, first in enumerate(names)
        for second in names[position + 1 :]
        if abs(recall_by_strategy[first] - recall_by_strategy[second]) <= tolerance + _FLOAT_SLACK
    ]


@dataclass(frozen=True, slots=True)
class ThresholdProposal:
    """A not-found threshold and how many of the scores it puts on the right side."""

    threshold: float
    correct: int
    total: int


def propose_threshold(
    answerable: Sequence[float], off_topic: Sequence[float]
) -> ThresholdProposal | None:
    """The top-1 score threshold that best separates answerable from off-topic questions.

    A score at or above the threshold passes. "Correct" = answerable scores that pass plus
    off-topic scores that do not. Candidates are the midpoints between neighbouring
    scores; of equally good ones the lowest wins (it refuses fewer real questions). None
    if a group is empty or all scores are the same. A proposal only: the owner decides.
    """
    if not answerable or not off_topic:
        return None
    scores = sorted({*answerable, *off_topic})
    best: ThresholdProposal | None = None
    for lower, upper in zip(scores, scores[1:], strict=False):
        threshold = (lower + upper) / 2
        correct = sum(1 for score in answerable if score >= threshold)
        correct += sum(1 for score in off_topic if score < threshold)
        if best is None or correct > best.correct:
            best = ThresholdProposal(
                threshold=threshold, correct=correct, total=len(answerable) + len(off_topic)
            )
    return best
