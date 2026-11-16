"""Retrieval metrics of the mini evaluation (Stage 4, note 0025). Pure functions.

Every question has one rank: the 1-based position of the first relevant chunk in its
retrieved list, or None if no retrieved chunk is relevant.

Recall@k here is the share of questions with a relevant chunk in the top k. (Each
question counts once, however many evidence entries it has.)

Confidence intervals use a cluster bootstrap: whole DOCUMENTS are drawn with replacement,
not single questions. Questions about one document rise and fall together (one bad parse
hurts all of them), so drawing questions on their own would make the interval too narrow.
"""

import math
import random
from collections.abc import Callable, Mapping, Sequence

Rank = int | None

MRR_CUTOFF = 10
BOOTSTRAP_SEED = 20261116
BOOTSTRAP_RESAMPLES = 10_000
INTERVAL_LOW = 0.025
INTERVAL_HIGH = 0.975


def hits_at_k(ranks: Sequence[Rank], k: int) -> int:
    """How many questions have their first relevant chunk at rank k or better."""
    if k < 1:
        raise ValueError(f"k must be 1 or more, got {k}")
    return sum(1 for rank in ranks if rank is not None and rank <= k)


def recall_at_k(ranks: Sequence[Rank], k: int) -> float:
    """hits_at_k / number of questions; 0.0 when there are no questions."""
    if not ranks:
        return 0.0
    return hits_at_k(ranks, k) / len(ranks)


def mrr(ranks: Sequence[Rank], cutoff: int = MRR_CUTOFF) -> float:
    """Mean of 1/rank; a question with no relevant chunk within the cutoff adds 0."""
    if cutoff < 1:
        raise ValueError(f"cutoff must be 1 or more, got {cutoff}")
    if not ranks:
        return 0.0
    total = sum(1.0 / rank for rank in ranks if rank is not None and rank <= cutoff)
    return total / len(ranks)


def bootstrap_statistics(
    ranks_by_document: Mapping[str, Sequence[Rank]],
    metric: Callable[[Sequence[Rank]], float],
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float]:
    """The metric on each resample, in the order drawn.

    One resample draws as many documents as there are, with replacement, and pools all
    questions of the drawn documents (a document drawn twice counts twice). Documents are
    sorted by id first, so the same data and seed always give the same list. Documents
    without questions are left out.
    """
    if resamples < 1:
        raise ValueError(f"resamples must be 1 or more, got {resamples}")
    documents = sorted(name for name, ranks in ranks_by_document.items() if ranks)
    if not documents:
        return []
    rng = random.Random(seed)
    statistics: list[float] = []
    for _ in range(resamples):
        drawn = rng.choices(documents, k=len(documents))
        pooled = [rank for name in drawn for rank in ranks_by_document[name]]
        statistics.append(metric(pooled))
    return statistics


def percentile_interval(
    values: Sequence[float], low: float = INTERVAL_LOW, high: float = INTERVAL_HIGH
) -> tuple[float, float]:
    """The values at the low and high positions of the sorted list (percentile method).

    Positions are rounded outwards (down for low, up for high), so the interval is never
    narrower than the data supports. No values: (0.0, 0.0).
    """
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    last = len(ordered) - 1
    return ordered[math.floor(low * last)], ordered[math.ceil(high * last)]


def bootstrap_interval(
    ranks_by_document: Mapping[str, Sequence[Rank]],
    metric: Callable[[Sequence[Rank]], float],
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """The 95% cluster-bootstrap interval of a metric (documents resampled)."""
    return percentile_interval(bootstrap_statistics(ranks_by_document, metric, seed, resamples))
