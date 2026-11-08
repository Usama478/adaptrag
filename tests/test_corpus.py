"""Tests for the evaluation corpus manifest and its validation (pure unit tests)."""

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from adaptrag.evaluation.corpus import (
    CATEGORIES,
    CorpusEntry,
    CorpusError,
    load_corpus,
    parse_corpus,
)

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "eval" / "corpus.json"

Mutation = Callable[[dict[str, Any]], None]


@pytest.fixture
def real_corpus() -> list[CorpusEntry]:
    return load_corpus(MANIFEST_PATH)


def valid_manifest() -> dict[str, Any]:
    """A small manifest that passes: one dev entry and one test entry."""
    return {
        "version": 1,
        "documents": [
            {
                "id": "dev_a",
                "split": "dev",
                "category": "sectioned",
                "title": None,
                "path": "data/samples/a.pdf",
                "source_url": None,
                "sha256": "a" * 64,
            },
            {
                "id": "test_b",
                "split": "test",
                "category": "short",
                "title": "A title",
                "path": "data/test_set/b.pdf",
                "source_url": "https://example.com/b.pdf",
                "sha256": None,
            },
        ],
    }


def set_field(index: int, key: str, value: object) -> Mutation:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["documents"][index][key] = value

    return mutate


def set_top_level(key: str, value: object) -> Mutation:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest[key] = value

    return mutate


def remove_key(index: int, key: str) -> Mutation:
    def mutate(manifest: dict[str, Any]) -> None:
        del manifest["documents"][index][key]

    return mutate


def add_dev_entry_with_same_path(manifest: dict[str, Any]) -> None:
    # Same split as the original, so only the path repetition can be the error.
    manifest["documents"].append({**manifest["documents"][0], "id": "dev_c"})


# The real manifest


def test_real_manifest_parses_with_21_entries(real_corpus: list[CorpusEntry]) -> None:
    assert len(real_corpus) == 21


def test_real_manifest_has_9_dev_and_12_test(real_corpus: list[CorpusEntry]) -> None:
    splits = Counter(entry.split for entry in real_corpus)
    assert splits == {"dev": 9, "test": 12}


def test_every_category_appears_in_both_splits(real_corpus: list[CorpusEntry]) -> None:
    for split in ("dev", "test"):
        found = {entry.category for entry in real_corpus if entry.split == split}
        assert found == set(CATEGORIES), split


def test_exactly_3_test_entries_per_category(real_corpus: list[CorpusEntry]) -> None:
    counts = Counter(entry.category for entry in real_corpus if entry.split == "test")
    assert counts == dict.fromkeys(CATEGORIES, 3)


def test_every_test_entry_has_a_source_url(real_corpus: list[CorpusEntry]) -> None:
    assert [e.id for e in real_corpus if e.split == "test" and e.source_url is None] == []


def test_every_entry_has_a_recorded_hash(real_corpus: list[CorpusEntry]) -> None:
    # Fails until the owner has run scripts/fetch_corpus.py, which records the hashes.
    assert [entry.id for entry in real_corpus if entry.sha256 is None] == []


# Validation


def test_valid_manifest_parses_in_file_order() -> None:
    # The failure cases below start from this manifest, so they fail for their own reason.
    assert parse_corpus(valid_manifest()) == [
        CorpusEntry(
            id="dev_a",
            split="dev",
            category="sectioned",
            title=None,
            path="data/samples/a.pdf",
            source_url=None,
            sha256="a" * 64,
        ),
        CorpusEntry(
            id="test_b",
            split="test",
            category="short",
            title="A title",
            path="data/test_set/b.pdf",
            source_url="https://example.com/b.pdf",
            sha256=None,
        ),
    ]


def test_null_hashes_may_repeat() -> None:
    manifest = valid_manifest()
    manifest["documents"][0]["sha256"] = None
    assert [entry.sha256 for entry in parse_corpus(manifest)] == [None, None]


def test_manifest_must_be_an_object() -> None:
    with pytest.raises(CorpusError, match="must be an object"):
        parse_corpus([])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(set_field(1, "id", "dev_a"), "dev_a: duplicate id", id="duplicate_id"),
        pytest.param(add_dev_entry_with_same_path, "duplicate path", id="duplicate_path"),
        pytest.param(set_field(0, "split", "train"), "split must", id="unknown_split"),
        pytest.param(set_field(0, "category", "other"), "category must", id="unknown_category"),
        pytest.param(
            set_field(1, "path", "data/samples/b.pdf"),
            "test_b: path must start with data/test_set/",
            id="test_entry_under_samples",
        ),
        pytest.param(
            set_field(0, "path", "data/test_set/a.pdf"),
            "dev_a: path must start with data/samples/",
            id="dev_entry_under_test_set",
        ),
        pytest.param(set_field(0, "path", "data/samples/a.txt"), "ending in", id="not_pdf"),
        pytest.param(set_field(0, "sha256", "A" * 64), "sha256 must", id="sha256_uppercase"),
        pytest.param(set_field(0, "sha256", "a" * 63), "sha256 must", id="sha256_63_chars"),
        pytest.param(
            set_field(1, "sha256", "a" * 64),
            r"test_b: duplicate sha256 \(same file as dev_a\)",
            id="duplicate_sha256",
        ),
        pytest.param(
            set_field(0, "source_url", "http://example.com/a.pdf"),
            "source_url must",
            id="source_url_http",
        ),
        pytest.param(set_field(0, "extra", 1), "keys must", id="unknown_extra_key"),
        pytest.param(remove_key(0, "sha256"), "keys must", id="missing_key"),
        pytest.param(set_top_level("version", 2), "version must", id="version_2"),
        pytest.param(set_top_level("version", True), "version must", id="version_true"),
        pytest.param(set_top_level("documents", []), "non-empty list", id="empty_documents"),
        pytest.param(set_field(1, "title", "   "), "test_b: title must", id="blank_title"),
    ],
)
def test_invalid_manifest_is_rejected(mutate: Mutation, message: str) -> None:
    manifest = valid_manifest()
    mutate(manifest)
    with pytest.raises(CorpusError, match=message):
        parse_corpus(manifest)
