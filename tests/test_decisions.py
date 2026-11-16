"""The Stage 4 decision rules: table budget, very short, router, ties, threshold (unit)."""

import pytest

from adaptrag.evaluation.decisions import (
    ABOVE,
    ADOPT,
    BELOW,
    CONSIDER_SHORT_RULE,
    KEEP,
    KEEP_V1,
    NO_SHORT_RULE,
    BudgetComparison,
    DocumentOutcome,
    propose_threshold,
    routed_loses,
    router_decision,
    separating_threshold,
    table_budget_decision,
    tied_pairs,
    very_short_decision,
    very_short_documents,
)


def comparison(
    over_with: int, hits_without: int, hits_with: int, strategy: str = "recursive"
) -> BudgetComparison:
    return BudgetComparison(
        strategy=strategy,
        over_limit_without=5,
        over_limit_with=over_with,
        table_hits_without=hits_without,
        table_hits_with=hits_with,
    )


def outcome(document_id: str, routed_hits: int, other_hits: int) -> DocumentOutcome:
    """A document routed to section_aware, with recursive as the only rival."""
    return DocumentOutcome(
        document_id=document_id,
        routed="section_aware",
        hits_by_strategy={"section_aware": routed_hits, "recursive": other_hits},
    )


# Table token budget


def test_budget_is_adopted_when_nothing_is_over_and_at_most_one_hit_is_lost() -> None:
    both = [comparison(0, 10, 9), comparison(0, 8, 8, strategy="section_aware")]

    assert table_budget_decision(both) == ADOPT


def test_budget_is_kept_out_when_a_table_chunk_is_still_over_the_limit() -> None:
    both = [comparison(0, 10, 10), comparison(1, 8, 8, strategy="section_aware")]

    assert table_budget_decision(both) == KEEP


def test_budget_is_kept_out_when_one_strategy_loses_two_table_hits() -> None:
    both = [comparison(0, 10, 10), comparison(0, 8, 6, strategy="section_aware")]

    assert table_budget_decision(both) == KEEP


def test_budget_is_kept_out_when_there_is_nothing_to_compare() -> None:
    assert table_budget_decision([]) == KEEP


def test_gaining_table_hits_does_not_block_adoption() -> None:
    # A rule written with abs() would wrongly treat +2 hits as a loss of 2.
    assert table_budget_decision([comparison(0, 5, 7)]) == ADOPT


# Very short documents


def test_very_short_rule_needs_two_documents_under_2000_words() -> None:
    one_short = {"a": 1999, "b": 5000}
    two_short = {"b": 1500, "a": 1999, "c": 9000}

    assert very_short_decision(one_short) == NO_SHORT_RULE
    assert very_short_decision(two_short) == CONSIDER_SHORT_RULE
    assert very_short_documents(two_short) == ["a", "b"]


def test_a_document_of_exactly_2000_words_is_not_very_short() -> None:
    assert very_short_decision({"a": 2000, "b": 2000}) == NO_SHORT_RULE


# Router rule


def test_routed_strategy_loses_only_when_another_one_has_more_hits() -> None:
    assert routed_loses(outcome("d", routed_hits=7, other_hits=8))
    assert not routed_loses(outcome("d", routed_hits=8, other_hits=8))  # a tie is not a loss
    assert not routed_loses(outcome("d", routed_hits=9, other_hits=8))


def test_router_keeps_v1_with_only_two_losses_even_if_a_threshold_separates_them() -> None:
    outcomes = [outcome("d1", 5, 8), outcome("d2", 5, 8), outcome("d3", 8, 5), outcome("d4", 8, 5)]
    headings = {"d1": 2, "d2": 3, "d3": 9, "d4": 8}
    profiles = {name: {"headings": value} for name, value in headings.items()}

    decision = router_decision(outcomes, profiles)

    assert decision.verdict == KEEP_V1
    assert decision.losing_documents == ("d1", "d2")
    assert decision.threshold is None


def test_router_keeps_v1_when_no_single_threshold_separates_the_losers() -> None:
    outcomes = [
        outcome("d1", 5, 8),
        outcome("d2", 5, 8),
        outcome("d3", 5, 8),
        outcome("d4", 8, 5),
        outcome("d5", 8, 5),
    ]
    # Losers hold the lowest AND the highest value: no cut puts them on one side.
    headings = {"d1": 2, "d2": 20, "d3": 5, "d4": 10, "d5": 3}
    profiles = {name: {"headings": value} for name, value in headings.items()}

    decision = router_decision(outcomes, profiles)

    assert decision.verdict == KEEP_V1
    assert decision.losing_documents == ("d1", "d2", "d3")


def test_router_proposes_the_threshold_that_separates_three_losers() -> None:
    outcomes = [
        outcome("d1", 5, 8),
        outcome("d2", 5, 8),
        outcome("d3", 5, 8),
        outcome("d4", 8, 5),
        outcome("d5", 8, 5),
    ]
    # "a_words" comes first alphabetically but does not separate; "headings" does.
    profiles = {
        "d1": {"a_words": 100, "headings": 2},
        "d2": {"a_words": 900, "headings": 3},
        "d3": {"a_words": 500, "headings": 4},
        "d4": {"a_words": 300, "headings": 10},
        "d5": {"a_words": 700, "headings": 12},
    }

    decision = router_decision(outcomes, profiles)

    assert decision.threshold is not None
    assert decision.threshold.field == "headings"
    assert decision.threshold.value == 7.0  # halfway between 4 and 10
    assert decision.threshold.losing_side == BELOW
    assert decision.verdict == "CANDIDATE v2: headings < 7"


def test_threshold_can_have_the_losers_above_it() -> None:
    profiles = {"d1": {"tables": 30}, "d2": {"tables": 40}, "d3": {"tables": 4}}

    threshold = separating_threshold(profiles, {"d1", "d2"})

    assert threshold is not None
    assert (threshold.value, threshold.losing_side) == (17.0, ABOVE)
    assert threshold.describe() == "tables > 17"


def test_no_threshold_when_every_document_loses() -> None:
    profiles = {"d1": {"tables": 1}, "d2": {"tables": 2}}

    assert separating_threshold(profiles, {"d1", "d2"}) is None


# Ties and the not-found threshold


def test_strategies_within_one_point_of_recall_are_tied() -> None:
    recall = {"fixed_size": 0.70, "recursive": 0.71, "section_aware": 0.75}

    # 0.71 - 0.70 is a hair over 0.01 in floating point; it must still count as "within 0.01".
    assert tied_pairs(recall) == [("fixed_size", "recursive")]
    assert tied_pairs({"a": 0.70, "b": 0.72}) == []


def test_threshold_between_two_clear_groups_gets_everything_right() -> None:
    proposal = propose_threshold([0.70, 0.80, 0.65], [0.40, 0.45])

    assert proposal is not None
    assert proposal.threshold == pytest.approx(0.55)  # halfway between 0.45 and 0.65
    assert (proposal.correct, proposal.total) == (5, 5)


def test_threshold_with_overlapping_groups_takes_the_lowest_of_the_best() -> None:
    # 0.45 and 0.65 both get 4 of 5 right; 0.55 and 0.75 get 3.
    proposal = propose_threshold([0.5, 0.7, 0.8], [0.4, 0.6])

    assert proposal is not None
    assert proposal.threshold == pytest.approx(0.45)
    assert (proposal.correct, proposal.total) == (4, 5)


def test_no_threshold_without_two_groups_of_different_scores() -> None:
    assert propose_threshold([0.7, 0.8], []) is None
    assert propose_threshold([], [0.4]) is None
    assert propose_threshold([0.5, 0.5], [0.5]) is None
