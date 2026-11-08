"""Evaluation corpus: the manifest of benchmark documents and their dev/test split.

Pure code only (no network, no PDF parsing), so every rule is unit-tested.
`scripts/fetch_corpus.py` uses it to download the documents and record their hashes.

The manifest (`eval/corpus.json`) is frozen before any router rule exists (note 0017).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("dev", "test")
CATEGORIES = ("sectioned", "short", "table_heavy", "poorly_structured")
DEV_DIR = "data/samples"
TEST_DIR = "data/test_set"
MANIFEST_KEYS = ("id", "split", "category", "title", "path", "source_url", "sha256")

MANIFEST_VERSION = 1

_SPLIT_DIRS = {"dev": DEV_DIR, "test": TEST_DIR}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class CorpusError(ValueError):
    """The corpus manifest is not valid."""


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """One benchmark document. `path` is relative to the repo root."""

    id: str
    split: str
    category: str
    title: str | None
    path: str
    source_url: str | None
    sha256: str | None


def _parse_document(document: object, position: int) -> CorpusEntry:
    """Check one manifest entry on its own. Uniqueness is checked by `parse_corpus`."""
    if not isinstance(document, dict):
        raise CorpusError(f"document {position} is not an object")

    # Name the entry by its id when it has a usable one, otherwise by its position.
    entry_id = document.get("id")
    name = entry_id if isinstance(entry_id, str) and entry_id else f"document {position}"

    if set(document) != set(MANIFEST_KEYS):
        raise CorpusError(f"{name}: keys must be exactly {MANIFEST_KEYS}")
    if not isinstance(entry_id, str) or not entry_id:
        raise CorpusError(f"{name}: id must be a non-empty string")

    split = document["split"]
    if split not in SPLITS:
        raise CorpusError(f"{name}: split must be one of {SPLITS}")

    category = document["category"]
    if category not in CATEGORIES:
        raise CorpusError(f"{name}: category must be one of {CATEGORIES}")

    title = document["title"]
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise CorpusError(f"{name}: title must be null or a non-blank string")

    path = document["path"]
    if not isinstance(path, str) or not path.endswith(".pdf"):
        raise CorpusError(f"{name}: path must be a string ending in .pdf")
    required_dir = _SPLIT_DIRS[split]
    if not path.startswith(f"{required_dir}/"):
        raise CorpusError(f"{name}: path must start with {required_dir}/ for split {split}")

    source_url = document["source_url"]
    if source_url is not None and (
        not isinstance(source_url, str) or not source_url.startswith("https://")
    ):
        raise CorpusError(f"{name}: source_url must be null or start with https://")

    sha256 = document["sha256"]
    if sha256 is not None and (
        not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None
    ):
        raise CorpusError(f"{name}: sha256 must be null or 64 lowercase hex characters")

    return CorpusEntry(
        id=entry_id,
        split=split,
        category=category,
        title=title,
        path=path,
        source_url=source_url,
        sha256=sha256,
    )


def parse_corpus(data: object) -> list[CorpusEntry]:
    """Validate a loaded manifest and return its entries in file order.

    Raises CorpusError, naming the entry when its id is known.
    """
    if not isinstance(data, dict):
        raise CorpusError("the manifest must be an object")

    version = data.get("version")
    # bool is a subclass of int in Python, so `true` must be ruled out explicitly.
    if not isinstance(version, int) or isinstance(version, bool) or version != MANIFEST_VERSION:
        raise CorpusError(f"version must be {MANIFEST_VERSION}")

    documents = data.get("documents")
    if not isinstance(documents, list) or not documents:
        raise CorpusError('"documents" must be a non-empty list')

    entries: list[CorpusEntry] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    hash_owners: dict[str, str] = {}  # sha256 -> id of the entry that has it
    for position, document in enumerate(documents, start=1):
        entry = _parse_document(document, position)
        if entry.id in seen_ids:
            raise CorpusError(f"{entry.id}: duplicate id")
        if entry.path in seen_paths:
            raise CorpusError(f"{entry.id}: duplicate path {entry.path}")
        # Null means "not recorded yet", so only real hashes can clash.
        if entry.sha256 is not None:
            if entry.sha256 in hash_owners:
                other_id = hash_owners[entry.sha256]
                raise CorpusError(f"{entry.id}: duplicate sha256 (same file as {other_id})")
            hash_owners[entry.sha256] = entry.id
        seen_ids.add(entry.id)
        seen_paths.add(entry.path)
        entries.append(entry)
    return entries


def load_corpus(path: Path) -> list[CorpusEntry]:
    """Read a manifest file (UTF-8 JSON) and validate it with `parse_corpus`."""
    return parse_corpus(json.loads(path.read_text(encoding="utf-8")))
