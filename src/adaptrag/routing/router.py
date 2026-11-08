"""The router: explicit rules that map a document profile to a chunking strategy.

The rules are ordered and the first match wins. fixed_size is the control in the
benchmark, so the router never chooses it.
"""

from dataclasses import dataclass

from adaptrag.routing.profile import DocumentProfile

# Both thresholds come from the dev profiles (note 0021).
MIN_HEADINGS = 5
MIN_NUMBERED_SHARE = 0.3


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """The chosen strategy, the rule that chose it, and a sentence explaining why."""

    strategy: str
    rule: str
    reason: str


def route(profile: DocumentProfile) -> RouteDecision:
    """Return the decision of the first rule that matches this profile."""
    headings = profile.headings
    numbered_percent = f"{profile.numbered_heading_share:.0%}"

    if headings < MIN_HEADINGS:
        found = "1 heading was" if headings == 1 else f"{headings} headings were"
        return RouteDecision(
            strategy="recursive",
            rule="few_headings",
            reason=f"Chose recursive chunking because only {found} found.",
        )

    if profile.numbered_heading_share < MIN_NUMBERED_SHARE:
        return RouteDecision(
            strategy="recursive",
            rule="unnumbered_headings",
            reason=(
                f"Chose recursive chunking because only {numbered_percent} of the "
                f"{headings} headings are numbered, so they may not form a reliable outline."
            ),
        )

    return RouteDecision(
        strategy="section_aware",
        rule="numbered_outline",
        reason=(
            f"Chose section-aware chunking because {headings} headings were found and "
            f"{numbered_percent} are numbered like a paper's outline."
        ),
    )
