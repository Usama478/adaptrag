"""The Strategy base class and the registry (pure: no database, no real model)."""

import json
from typing import cast

import pytest
from sqlalchemy.orm import Session

from adaptrag.chunking.chunks import TextChunk
from adaptrag.embedding.base import Embedder
from adaptrag.embedding.fake import FakeEmbedder
from adaptrag.parsing.blocks import ParsedDocument
from adaptrag.retrieval.results import ScoredChunk
from adaptrag.strategies import base
from adaptrag.strategies.base import Strategy
from adaptrag.strategies.registry import STRATEGIES, UnknownStrategyError, build_strategy


class DummyStrategy(Strategy):
    """The smallest complete strategy. It exists only in this test file."""

    name = "dummy"

    @property
    def params(self) -> dict[str, int | float | str | bool]:
        return {"size": 10, "ratio": 0.5, "mode": "plain", "strict": True}

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        return []


def test_subclass_without_chunk_cannot_be_instantiated() -> None:
    class NoChunkMethod(Strategy):
        name = "no_chunk_method"

        @property
        def params(self) -> dict[str, int | float | str | bool]:
            return {}

    with pytest.raises(TypeError, match="abstract method.*chunk"):
        NoChunkMethod(FakeEmbedder())


def test_params_survive_a_json_round_trip() -> None:
    params = DummyStrategy(FakeEmbedder()).params

    assert json.loads(json.dumps(params)) == params


def test_build_strategy_returns_registered_class_with_given_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(STRATEGIES, DummyStrategy.name, DummyStrategy)
    embedder = FakeEmbedder()

    strategy = build_strategy(DummyStrategy.name, embedder)

    assert type(strategy) is DummyStrategy
    assert strategy.embedder is embedder


def test_unknown_name_error_lists_known_names_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "adaptrag.strategies.registry.STRATEGIES",
        {"zeta": DummyStrategy, "alpha": DummyStrategy},
    )

    with pytest.raises(UnknownStrategyError) as error:
        build_strategy("missing", FakeEmbedder())

    assert "'missing'" in str(error.value)
    assert "alpha, zeta" in str(error.value)


def test_every_registry_key_equals_its_class_name() -> None:
    for key, cls in STRATEGIES.items():
        assert key == cls.name


def test_default_retrieve_delegates_to_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[tuple[Session, int, Embedder, str, int]] = []
    expected = [
        ScoredChunk(chunk_id=1, chunk_index=0, text="text", page_start=1, page_end=1, score=0.9)
    ]

    def fake_retrieve_dense(
        session: Session, chunk_set_id: int, embedder: Embedder, question: str, k: int
    ) -> list[ScoredChunk]:
        received.append((session, chunk_set_id, embedder, question, k))
        return expected

    monkeypatch.setattr(base, "retrieve_dense", fake_retrieve_dense)
    session = cast(Session, object())
    embedder = FakeEmbedder()

    result = DummyStrategy(embedder).retrieve(session, chunk_set_id=7, query="a question", k=3)

    assert received == [(session, 7, embedder, "a question", 3)]
    assert result is expected
