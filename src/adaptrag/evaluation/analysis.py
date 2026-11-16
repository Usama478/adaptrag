"""Turn per-question ranks into the rows of the mini-evaluation table (Stage 4). Pure.

Shape used everywhere here: ranks[strategy][document id] is the list of ranks of that
document's questions, in question order (see evaluation/metrics.py for what a rank is).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial

from adaptrag.evaluation.metrics import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    Rank,
    bootstrap_interval,
    hits_at_k,
    mrr,
    recall_at_k,
)

HEADLINE_K = 5

RanksByDocument = Mapping[str, Sequence[Rank]]


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One row of the results table."""

    name: str
    questions: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    recall_at_5_interval: tuple[float, float]
    mrr_interval: tuple[float, float]


def pooled(ranks_by_document: RanksByDocument) -> list[Rank]:
    """All ranks, documents in id order."""
    return [rank for name in sorted(ranks_by_document) for rank in ranks_by_document[name]]


def metric_row(
    name: str,
    ranks_by_document: RanksByDocument,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> MetricRow:
    """The metrics of one row, with cluster-bootstrap intervals for Recall@5 and MRR."""
    ranks = pooled(ranks_by_document)
    recall_at_5 = partial(recall_at_k, k=HEADLINE_K)
    return MetricRow(
        name=name,
        questions=len(ranks),
        recall_at_1=recall_at_k(ranks, 1),
        recall_at_5=recall_at_k(ranks, HEADLINE_K),
        recall_at_10=recall_at_k(ranks, 10),
        mrr=mrr(ranks),
        recall_at_5_interval=bootstrap_interval(ranks_by_document, recall_at_5, seed, resamples),
        mrr_interval=bootstrap_interval(ranks_by_document, mrr, seed, resamples),
    )


def _score(ranks: Sequence[Rank]) -> tuple[int, float]:
    """How strategies are compared on the same questions: Recall@5 hits, then MRR."""
    return hits_at_k(ranks, HEADLINE_K), mrr(ranks)


def best_single_strategy(ranks: Mapping[str, RanksByDocument], strategies: Sequence[str]) -> str:
    """The strategy with the best pooled score; the earliest in `strategies` wins a tie."""
    if not strategies:
        raise ValueError("no strategies to choose from")
    best = strategies[0]
    for name in strategies[1:]:
        if _score(pooled(ranks[name])) > _score(pooled(ranks[best])):
            best = name
    return best


def best_strategy_per_document(
    ranks: Mapping[str, RanksByDocument], strategies: Sequence[str]
) -> dict[str, str]:
    """For each document, its best strategy; the earliest in `strategies` wins a tie."""
    if not strategies:
        raise ValueError("no strategies to choose from")
    best: dict[str, str] = {}
    for document in sorted(ranks[strategies[0]]):
        chosen = strategies[0]
        for name in strategies[1:]:
            if _score(ranks[name][document]) > _score(ranks[chosen][document]):
                chosen = name
        best[document] = chosen
    return best


def compose(
    ranks: Mapping[str, RanksByDocument], choice: Mapping[str, str]
) -> dict[str, Sequence[Rank]]:
    """Each document's ranks under the strategy `choice` names for it (router, oracle)."""
    return {document: ranks[strategy][document] for document, strategy in choice.items()}


@dataclass(frozen=True, slots=True)
class PairedOutcome:
    """Per-question comparison of two strategies at k: who found what the other missed."""

    first_only: int
    second_only: int
    both: int
    neither: int


def paired_outcome(
    first: Sequence[Rank], second: Sequence[Rank], k: int = HEADLINE_K
) -> PairedOutcome:
    """Compare two rank lists of the same questions, in the same order."""
    if len(first) != len(second):
        raise ValueError(f"rank lists differ in length: {len(first)} and {len(second)}")
    first_only = second_only = both = neither = 0
    for first_rank, second_rank in zip(first, second, strict=True):
        first_hit = first_rank is not None and first_rank <= k
        second_hit = second_rank is not None and second_rank <= k
        if first_hit and second_hit:
            both += 1
        elif first_hit:
            first_only += 1
        elif second_hit:
            second_only += 1
        else:
            neither += 1
    return PairedOutcome(first_only=first_only, second_only=second_only, both=both, neither=neither)
