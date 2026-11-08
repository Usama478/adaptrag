"""The registry: strategy names mapped to their classes.

An explicit dict on purpose: adding a strategy is one new file plus one line here, and
the full list of strategies can be read in one place.
"""

from adaptrag.embedding.base import Embedder
from adaptrag.strategies.base import Strategy
from adaptrag.strategies.fixed_size import FixedSizeStrategy

STRATEGIES: dict[str, type[Strategy]] = {
    "fixed_size": FixedSizeStrategy,
}

# Every document uses this strategy until Step 5 replaces it with the router's choice.
DEFAULT_STRATEGY = "fixed_size"


class UnknownStrategyError(LookupError):
    """No strategy is registered under this name."""

    def __init__(self, name: str) -> None:
        known = ", ".join(strategy_names()) or "(none)"
        super().__init__(f"Unknown strategy {name!r}. Known strategies: {known}")


def strategy_names() -> list[str]:
    """Every registered strategy name, sorted."""
    return sorted(STRATEGIES)


def strategy_class(name: str) -> type[Strategy]:
    """The class registered under this name, or UnknownStrategyError."""
    try:
        return STRATEGIES[name]
    except KeyError:
        raise UnknownStrategyError(name) from None


def build_strategy(name: str, embedder: Embedder) -> Strategy:
    """A new instance of the named strategy, holding the given embedder."""
    return strategy_class(name)(embedder)
