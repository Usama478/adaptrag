"""Rows of the mini-evaluation table: metrics, best strategy, oracle, pairs (unit tests)."""

import pytest

from adaptrag.evaluation.analysis import (
    best_single_strategy,
    best_strategy_per_document,
    compose,
    metric_row,
    paired_outcome,
)
from adaptrag.evaluation.metrics import Rank

RANKS: dict[str, dict[str, list[Rank]]] = {
    # doc_a: recursive answers more in the top 5. doc_b: both answer 1, fixed_size earlier.
    "fixed_size": {"doc_a": [1, None, None], "doc_b": [1, None]},
    "recursive": {"doc_a": [2, 3, None], "doc_b": [4, None]},
}
ORDER = ("fixed_size", "recursive")


def test_metric_row_matches_a_hand_computed_example() -> None:
    row = metric_row("example", {"a": [1, None], "b": [3, 6]}, resamples=200)

    assert (row.name, row.questions) == ("example", 4)
    assert row.recall_at_1 == pytest.approx(1 / 4)
    assert row.recall_at_5 == pytest.approx(2 / 4)
    assert row.recall_at_10 == pytest.approx(3 / 4)
    assert row.mrr == pytest.approx((1 + 1 / 3 + 1 / 6) / 4)
    # Each document has Recall@5 = 1/2, so every resample of documents gives 1/2 as well.
    assert row.recall_at_5_interval == (0.5, 0.5)
    # MRR differs per document (0.5 and 0.25), so its interval has a width.
    assert row.mrr_interval == (pytest.approx(0.25), pytest.approx(0.5))


def test_best_single_strategy_compares_hits_then_mrr_then_order() -> None:
    more_hits: dict[str, dict[str, list[Rank]]] = {"x": {"d": [5, 5]}, "y": {"d": [1, None]}}
    same_hits: dict[str, dict[str, list[Rank]]] = {"x": {"d": [5, None]}, "y": {"d": [1, None]}}
    all_equal: dict[str, dict[str, list[Rank]]] = {"x": {"d": [2]}, "y": {"d": [2]}}

    assert best_single_strategy(more_hits, ("x", "y")) == "x"  # 2 hits beat a better MRR
    assert best_single_strategy(same_hits, ("x", "y")) == "y"  # equal hits: MRR decides
    assert best_single_strategy(all_equal, ("x", "y")) == "x"  # all equal: the first listed
    assert best_single_strategy(all_equal, ("y", "x")) == "y"


def test_best_strategy_is_chosen_for_each_document_on_its_own() -> None:
    assert best_strategy_per_document(RANKS, ORDER) == {"doc_a": "recursive", "doc_b": "fixed_size"}


def test_compose_takes_each_document_from_its_chosen_strategy() -> None:
    choice = {"doc_a": "recursive", "doc_b": "fixed_size"}

    assert compose(RANKS, choice) == {"doc_a": [2, 3, None], "doc_b": [1, None]}


def test_paired_outcome_counts_each_kind_of_question_once() -> None:
    first: list[Rank] = [1, 6, None, 2]
    second: list[Rank] = [2, 1, None, 9]

    outcome = paired_outcome(first, second)

    assert (outcome.both, outcome.first_only, outcome.second_only, outcome.neither) == (1, 1, 1, 1)


def test_paired_outcome_refuses_lists_of_different_lengths() -> None:
    with pytest.raises(ValueError):
        paired_outcome([1, 2], [1])
