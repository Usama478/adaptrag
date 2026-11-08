"""Router rules: each rule, its boundaries, rule order, and the reason sentences."""

import pytest

from adaptrag.routing.profile import DocumentProfile
from adaptrag.routing.router import route
from adaptrag.strategies.registry import strategy_names


def profile(headings: int, numbered_share: float) -> DocumentProfile:
    """A profile where only the two signals the router reads matter."""
    return DocumentProfile(
        page_count=10,
        words=5000,
        headings=headings,
        numbered_heading_share=numbered_share,
        headings_per_1k_words=2.0,
        tables=0,
        table_word_share=0.0,
        fallback_pages=0,
        fallback_page_share=0.0,
        words_per_page=500.0,
    )


def grid() -> list[DocumentProfile]:
    """Profiles on both sides of each threshold, and at the extremes."""
    return [
        profile(headings, share)
        for headings in (0, 1, 4, 5, 6, 9, 26, 500)
        for share in (0.0, 0.111, 0.299, 0.3, 0.301, 0.419, 0.875, 1.0)
    ]


def test_few_headings_rule() -> None:
    # The share alone would pass, so only the heading count can cause this result.
    decision = route(profile(headings=3, numbered_share=0.85))

    assert decision.strategy == "recursive"
    assert decision.rule == "few_headings"


def test_unnumbered_headings_rule() -> None:
    # The heading count alone would pass, so only the share can cause this result.
    decision = route(profile(headings=9, numbered_share=0.111))

    assert decision.strategy == "recursive"
    assert decision.rule == "unnumbered_headings"


def test_numbered_outline_rule() -> None:
    decision = route(profile(headings=26, numbered_share=0.85))

    assert decision.strategy == "section_aware"
    assert decision.rule == "numbered_outline"


def test_exactly_five_headings_is_not_few() -> None:
    assert route(profile(headings=5, numbered_share=0.85)).rule == "numbered_outline"


def test_four_headings_is_few() -> None:
    assert route(profile(headings=4, numbered_share=0.85)).rule == "few_headings"


def test_exactly_the_minimum_share_is_enough_for_section_aware() -> None:
    assert route(profile(headings=9, numbered_share=0.3)).strategy == "section_aware"


def test_just_below_the_minimum_share_is_not_enough() -> None:
    decision = route(profile(headings=9, numbered_share=0.299))

    assert decision.strategy == "recursive"
    assert decision.rule == "unnumbered_headings"


def test_few_headings_is_checked_before_the_numbered_share() -> None:
    # Both recursive rules match here; the first one must win.
    assert route(profile(headings=3, numbered_share=0.0)).rule == "few_headings"


def test_every_returned_strategy_is_registered() -> None:
    assert {route(item).strategy for item in grid()} <= set(strategy_names())


def test_the_grid_reaches_both_strategies_and_never_the_control() -> None:
    # Exact equality also fails if a bug made the router always return one strategy.
    assert {route(item).strategy for item in grid()} == {"recursive", "section_aware"}


@pytest.mark.parametrize(
    ("headings", "numbered_share", "expected"),
    [
        (
            26,
            0.85,
            "Chose section-aware chunking because 26 headings were found and 85% are "
            "numbered like a paper's outline.",
        ),
        (
            12,
            0.419,
            "Chose section-aware chunking because 12 headings were found and 42% are "
            "numbered like a paper's outline.",
        ),
        (
            9,
            0.111,
            "Chose recursive chunking because only 11% of the 9 headings are numbered, "
            "so they may not form a reliable outline.",
        ),
        (
            7,
            0.0,
            "Chose recursive chunking because only 0% of the 7 headings are numbered, "
            "so they may not form a reliable outline.",
        ),
        (3, 0.85, "Chose recursive chunking because only 3 headings were found."),
        (0, 0.0, "Chose recursive chunking because only 0 headings were found."),
    ],
)
def test_reason_states_the_real_numbers(
    headings: int, numbered_share: float, expected: str
) -> None:
    assert route(profile(headings, numbered_share)).reason == expected


def test_reason_uses_the_singular_for_one_heading() -> None:
    reason = route(profile(headings=1, numbered_share=1.0)).reason

    assert reason == "Chose recursive chunking because only 1 heading was found."
