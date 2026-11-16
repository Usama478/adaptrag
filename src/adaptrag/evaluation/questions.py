"""Evaluation questions as the mini evaluation reads them (Stage 4).

Works for both kinds of question file: the owner-verified `eval/questions/paper_01.json`
and the machine-verified files in `eval/questions/generated/`. Both go through the Stage 2
loader's rules (`gate.parse_questions`): only verified entries with filled evidence count.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from adaptrag.evaluation.gate import Evidence, QuestionFileError, parse_questions

DEFAULT_TYPE = "factual"


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """A scorable question, with the corpus id of its document and its type."""

    id: str
    document_id: str
    type: str
    question: str
    answer: str
    evidence: tuple[Evidence, ...]

    @property
    def evidence_texts(self) -> tuple[str, ...]:
        return tuple(item.text for item in self.evidence)


def load_eval_questions(path: Path, document_id: str) -> list[EvalQuestion]:
    """The scorable questions of one file, in file order. Raises QuestionFileError."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QuestionFileError(f"{path} is not valid JSON: {exc}") from exc
    scorable, _skipped = parse_questions(data)

    # parse_questions has checked that every entry is an object with a unique string id.
    types: dict[str, str] = {}
    for entry in data["questions"]:
        question_type = entry.get("type")
        types[entry["id"]] = question_type if isinstance(question_type, str) else DEFAULT_TYPE

    return [
        EvalQuestion(
            id=question.id,
            document_id=document_id,
            type=types[question.id],
            question=question.question,
            answer=question.answer,
            evidence=question.evidence,
        )
        for question in scorable
    ]
