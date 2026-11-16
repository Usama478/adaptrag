"""Loading evaluation questions with their type, for both kinds of question file (unit)."""

import json
from collections import Counter
from pathlib import Path

from adaptrag.evaluation.gate import load_questions
from adaptrag.evaluation.questions import load_eval_questions

# Tracked in git (unlike data/), so a test may read it.
OWNER_SET = Path(__file__).resolve().parents[1] / "eval" / "questions" / "paper_01.json"


def test_types_are_read_and_unverified_entries_are_left_out(tmp_path: Path) -> None:
    def entry(question_id: str, verified: bool = True, **extra: str) -> dict[str, object]:
        return {
            "id": question_id,
            "question": f"Question {question_id}?",
            "answer": "An answer.",
            "evidence": [{"page": 2, "text": "some evidence text"}],
            "verified": verified,
            **extra,
        }

    data = {
        "questions": [
            entry("q1", type="table"),
            entry("q2"),  # no type given
            entry("q3", verified=False, type="factual"),
        ]
    }
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    questions = load_eval_questions(path, "doc_x")

    assert [(q.id, q.type, q.document_id) for q in questions] == [
        ("q1", "table", "doc_x"),
        ("q2", "factual", "doc_x"),
    ]
    assert questions[0].evidence_texts == ("some evidence text",)


def test_the_owner_verified_set_still_loads_with_both_loaders() -> None:
    scorable, skipped = load_questions(OWNER_SET)
    questions = load_eval_questions(OWNER_SET, "paper_01")

    assert (len(scorable), skipped) == (10, [])
    assert [question.id for question in questions] == [item.id for item in scorable]
    assert Counter(question.type for question in questions) == {"factual": 9, "two_section": 1}
    assert len(questions[-1].evidence) == 2  # q10 has two evidence entries
