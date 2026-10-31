"""Tests for the real sentence-transformers embedder.

Marked slow: they load BAAI/bge-small-en-v1.5 (downloaded from Hugging Face on first run).
Excluded by default; run them with `python -m pytest -m slow`. CI does not run them.
"""

import math

import pytest

from adaptrag.config import get_settings
from adaptrag.embedding.base import Embedder

pytestmark = pytest.mark.slow


def load_embedder() -> Embedder:
    # Imported here, not at the top: importing torch takes seconds, and the normal
    # (not slow) test run still collects this file.
    from adaptrag.embedding.sentence_transformer import get_embedder

    settings = get_settings()
    return get_embedder(
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_query_instruction,
    )


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return load_embedder()


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_model_reports_dimension_and_max_tokens(embedder: Embedder) -> None:
    assert embedder.model_name == "BAAI/bge-small-en-v1.5"
    assert embedder.dimension == 384
    assert embedder.max_tokens == 512


def test_vectors_are_unit_length(embedder: Embedder) -> None:
    vectors = embedder.embed_passages(["The cat sat on the mat."])
    vectors.append(embedder.embed_query("Where did the cat sit?"))
    for vector in vectors:
        assert len(vector) == 384
        assert math.sqrt(dot(vector, vector)) == pytest.approx(1.0, abs=1e-5)


def test_query_scores_higher_with_related_passage(embedder: Embedder) -> None:
    cat, stocks = embedder.embed_passages(
        ["The cat sat on the mat.", "Stock markets fell sharply today."]
    )
    query = embedder.embed_query("Where did the cat sit?")
    assert dot(query, cat) > dot(query, stocks)


def test_count_tokens_includes_special_tokens(embedder: Embedder) -> None:
    # "hello" is one token; [CLS] and [SEP] make three.
    assert embedder.count_tokens("hello") == 3


def test_model_is_loaded_once(embedder: Embedder) -> None:
    assert load_embedder() is embedder
