"""Stage 2 gate: load hand-verified questions and score answers against their evidence.

Pure functions only (no database, no model, no LLM), so every rule is unit-tested.
`scripts/gate_check.py` runs the real pipeline and uses these rules to score it.

Question file format: `eval/question_format_example.json`.

Rules:
- Only entries with `"verified": true` are scored. A verified entry whose evidence still
  has a null page or a "TO FILL" text is refused: it cannot be scored honestly.
- cited_ok: the answer was found AND at least one citation's page_start..page_end
  contains at least one evidence page (the gate rule in .claude/PROGRESS.md).
- retrieved_ok: at least one top-k source contains at least one evidence text, after
  both are normalised (casefold, every whitespace run collapsed to one space).
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptrag.answering.answer import Citation

PLACEHOLDER_MARKER = "TO FILL"

# Stage 2 gate: at least 8 of 10 questions cite the right page.
GATE_MIN_QUESTIONS = 10
GATE_MIN_CITED = 8

NOT_VERIFIED = "not verified"
EVIDENCE_NOT_FILLED = "evidence not filled (null page or TO FILL text)"


class QuestionFileError(ValueError):
    """The question file is not in the expected format."""


@dataclass(frozen=True, slots=True)
class Evidence:
    page: int
    text: str


@dataclass(frozen=True, slots=True)
class GateQuestion:
    """A verified question whose evidence is filled in: ready to score."""

    id: str
    question: str
    answer: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class SkippedQuestion:
    """An entry the loader will not score, with the reason."""

    id: str
    reason: str


def normalise(text: str) -> str:
    """Casefold and collapse every whitespace run (spaces, tabs, newlines) to one space."""
    return " ".join(text.casefold().split())


def cited_ok(found: bool, citations: Sequence[Citation], evidence_pages: Sequence[int]) -> bool:
    """True if the answer was found and a citation's page range holds an evidence page."""
    if not found:
        return False
    return any(
        citation.page_start <= page <= citation.page_end
        for citation in citations
        for page in evidence_pages
    )


def retrieved_ok(source_texts: Sequence[str], evidence_texts: Sequence[str]) -> bool:
    """True if a source text contains an evidence text (both normalised)."""
    sources = [normalise(text) for text in source_texts]
    # The loader rejects blank evidence; skipping it here too keeps "" from matching all.
    evidence = [normalise(text) for text in evidence_texts if text.strip()]
    return any(needle in source for source in sources for needle in evidence)


def gate_passed(cited_count: int, scored_count: int) -> bool:
    """At least 10 scored questions, and at least 8 of every 10 cite the right page."""
    if scored_count < GATE_MIN_QUESTIONS:
        return False
    return cited_count * GATE_MIN_QUESTIONS >= scored_count * GATE_MIN_CITED


def _evidence_is_filled(entry: dict[str, Any]) -> bool:
    """Every evidence item has an integer page >= 1 and a real (non-placeholder) text."""
    for item in entry["evidence"]:
        page = item.get("page")
        text = item.get("text")
        # bool is a subclass of int in Python, so `true` must be ruled out explicitly.
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            return False
        if not isinstance(text, str) or not text.strip() or PLACEHOLDER_MARKER in text:
            return False
    return True


def parse_questions(data: Any) -> tuple[list[GateQuestion], list[SkippedQuestion]]:
    """Split the entries of a loaded question file into scorable and skipped, in file order.

    Raises QuestionFileError if the structure is wrong (missing keys, wrong types,
    duplicate ids).
    """
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise QuestionFileError('expected an object with a "questions" list')

    scorable: list[GateQuestion] = []
    skipped: list[SkippedQuestion] = []
    seen_ids: set[str] = set()
    for position, entry in enumerate(data["questions"], start=1):
        if not isinstance(entry, dict):
            raise QuestionFileError(f"entry {position} is not an object")
        for key in ("id", "question", "answer", "evidence", "verified"):
            if key not in entry:
                raise QuestionFileError(f'entry {position} has no "{key}"')
        question_id = entry["id"]
        if not isinstance(question_id, str) or question_id in seen_ids:
            raise QuestionFileError(f"entry {position}: id missing, not a string, or repeated")
        seen_ids.add(question_id)
        evidence = entry["evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise QuestionFileError(f"{question_id}: evidence must be a non-empty list")
        if not all(isinstance(item, dict) for item in evidence):
            raise QuestionFileError(f"{question_id}: every evidence item must be an object")

        if entry["verified"] is not True:
            skipped.append(SkippedQuestion(id=question_id, reason=NOT_VERIFIED))
            continue
        if not _evidence_is_filled(entry):
            skipped.append(SkippedQuestion(id=question_id, reason=EVIDENCE_NOT_FILLED))
            continue

        scorable.append(
            GateQuestion(
                id=question_id,
                question=str(entry["question"]),
                answer=str(entry["answer"]),
                evidence=tuple(Evidence(page=item["page"], text=item["text"]) for item in evidence),
            )
        )
    return scorable, skipped


def load_questions(path: Path) -> tuple[list[GateQuestion], list[SkippedQuestion]]:
    """Read a question file (UTF-8 JSON) and split it with `parse_questions`."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QuestionFileError(f"{path} is not valid JSON: {exc}") from exc
    return parse_questions(data)
