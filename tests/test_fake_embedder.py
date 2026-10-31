"""Tests for FakeEmbedder (pure logic, no model, no database)."""

import math

import pytest

from adaptrag.embedding.base import Embedder
from adaptrag.embedding.fake import FakeEmbedder


def length(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def test_satisfies_embedder_protocol() -> None:
    embedder: Embedder = FakeEmbedder()  # mypy checks the protocol here
    assert embedder.model_name == "fake"


def test_same_text_gives_same_vector() -> None:
    assert FakeEmbedder().embed_passages(["hello"]) == FakeEmbedder().embed_passages(["hello"])


def test_different_texts_give_different_vectors() -> None:
    first, second = FakeEmbedder().embed_passages(["hello", "goodbye"])
    assert first != second


@pytest.mark.parametrize("dimension", [1, 8, 32, 33, 384])
def test_vectors_have_declared_dimension_and_unit_length(dimension: int) -> None:
    # 33 and 384 need more than one SHA-256 digest (32 bytes each).
    embedder = FakeEmbedder(dimension=dimension)
    vectors = embedder.embed_passages(["a", "b c", "d e f"]) + [embedder.embed_query("q")]
    for vector in vectors:
        assert len(vector) == embedder.dimension == dimension
        assert length(vector) == pytest.approx(1.0)


def test_large_dimension_is_deterministic_and_text_dependent() -> None:
    first, second = FakeEmbedder(dimension=384).embed_passages(["hello", "goodbye"])
    assert first == FakeEmbedder(dimension=384).embed_passages(["hello"])[0]
    assert first != second
    # The extra digests are not copies of the first one.
    assert first[32:64] != first[0:32]


def test_embed_passages_keeps_input_order() -> None:
    embedder = FakeEmbedder()
    texts = ["one", "two", "three"]
    batch = embedder.embed_passages(texts)
    one_by_one = [embedder.embed_passages([text])[0] for text in texts]
    assert batch == one_by_one
    assert embedder.embed_passages(list(reversed(texts))) == list(reversed(batch))


def test_query_differs_from_passage_for_same_text() -> None:
    embedder = FakeEmbedder()
    assert embedder.embed_query("hello") != embedder.embed_passages(["hello"])[0]


def test_empty_list_gives_no_vectors() -> None:
    assert FakeEmbedder().embed_passages([]) == []


def test_count_tokens_is_words_plus_two() -> None:
    assert FakeEmbedder().count_tokens("three little words") == 5


@pytest.mark.parametrize("dimension", [0, -1])
def test_dimension_below_one_is_rejected(dimension: int) -> None:
    with pytest.raises(ValueError):
        FakeEmbedder(dimension=dimension)
