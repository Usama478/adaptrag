"""Retrieval metrics and the cluster bootstrap (pure unit tests, hand-computed)."""

import random

import pytest

from adaptrag.evaluation.metrics import (
    Rank,
    bootstrap_interval,
    bootstrap_statistics,
    mrr,
    percentile_interval,
    recall_at_k,
)

# Five documents of different sizes and quality, so resamples give many different values.
MIXED: dict[str, list[Rank]] = {
    "a": [1, 1, 2],
    "b": [None, None, None, None, 7],
    "c": [3, None],
    "d": [1, 9, None, 4, 2, None, 1],
    "e": [None],
}

# One document with a single hit, one with nine misses.
UNLIKE: dict[str, list[Rank]] = {"hit": [1], "miss": [None] * 9}


def recall_at_5(ranks: list[Rank]) -> float:
    return recall_at_k(ranks, 5)


def test_recall_at_k_counts_ranks_up_to_and_including_k() -> None:
    ranks: list[Rank] = [1, 3, 5, 6, None]

    assert recall_at_k(ranks, 1) == pytest.approx(1 / 5)
    assert recall_at_k(ranks, 5) == pytest.approx(3 / 5)  # rank 5 counts, rank 6 does not
    assert recall_at_k(ranks, 10) == pytest.approx(4 / 5)  # None never counts


def test_recall_at_k_of_no_questions_is_zero() -> None:
    assert recall_at_k([], 5) == 0.0


def test_recall_at_k_rejects_k_below_one() -> None:
    with pytest.raises(ValueError):
        recall_at_k([1], 0)


def test_mrr_is_the_mean_of_one_over_rank() -> None:
    # (1/1 + 1/2 + 1/4 + 0) / 4
    assert mrr([1, 2, 4, None]) == pytest.approx(0.4375)


def test_mrr_counts_the_cutoff_rank_and_nothing_after_it() -> None:
    assert mrr([10]) == pytest.approx(0.1)
    assert mrr([11, 1]) == pytest.approx(0.5)  # rank 11 adds 0, but still counts as a question
    assert mrr([3], cutoff=2) == 0.0


def test_bootstrap_with_the_same_seed_gives_the_same_result() -> None:
    first = bootstrap_statistics(MIXED, recall_at_5, seed=1, resamples=50)
    again = bootstrap_statistics(MIXED, recall_at_5, seed=1, resamples=50)
    other = bootstrap_statistics(MIXED, recall_at_5, seed=2, resamples=50)

    assert first == again
    assert first != other  # the seed is really used
    assert bootstrap_interval(MIXED, recall_at_5, seed=1, resamples=50) == bootstrap_interval(
        MIXED, recall_at_5, seed=1, resamples=50
    )


def test_bootstrap_draws_whole_documents_with_the_standard_library_generator() -> None:
    # The same draws, written out by hand: documents sorted by id, drawn with replacement.
    rng = random.Random(7)
    names = sorted(MIXED)
    expected = []
    for _ in range(20):
        drawn = rng.choices(names, k=len(names))
        expected.append(mrr([rank for name in drawn for rank in MIXED[name]]))

    assert bootstrap_statistics(MIXED, mrr, seed=7, resamples=20) == expected


def test_bootstrap_resamples_documents_not_questions() -> None:
    statistics = bootstrap_statistics(UNLIKE, recall_at_5, seed=3, resamples=200)

    # Two documents drawn: hit+hit = 1/1, hit+miss = 1/10, miss+miss = 0/18. Drawing
    # single questions instead would give values such as 0.2 or 0.3.
    assert {round(value, 6) for value in statistics} == {0.0, 0.1, 1.0}


def test_bootstrap_interval_reaches_both_extremes_for_two_unlike_documents() -> None:
    # Each extreme (both draws the same document) has probability 1/4, far more than 2.5%.
    assert bootstrap_interval(UNLIKE, recall_at_5, seed=3, resamples=2000) == (0.0, 1.0)


def test_bootstrap_leaves_out_documents_without_questions() -> None:
    data: dict[str, list[Rank]] = {"full": [1, 2], "empty": []}

    # If "empty" could be drawn, a resample of two empties would score 0.0.
    assert set(bootstrap_statistics(data, recall_at_5, seed=5, resamples=100)) == {1.0}


def test_percentile_interval_rounds_positions_outwards() -> None:
    values = [float(number) for number in range(101)]  # positions 0..100

    # 2.5% of 100 = 2.5 -> down to 2; 97.5% of 100 = 97.5 -> up to 98.
    assert percentile_interval(values) == (2.0, 98.0)
    assert percentile_interval([]) == (0.0, 0.0)
