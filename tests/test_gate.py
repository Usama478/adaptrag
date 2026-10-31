"""Tests for the Stage 2 gate scoring rules and question loader (pure unit tests)."""

import json
from pathlib import Path
from typing import Any

import pytest

from adaptrag.answering.answer import Citation
from adaptrag.evaluation.gate import (
    EVIDENCE_NOT_FILLED,
    NOT_VERIFIED,
    QuestionFileError,
    cited_ok,
    gate_passed,
    load_questions,
    normalise,
    parse_questions,
    retrieved_ok,
)


def citation(page_start: int, page_end: int) -> Citation:
    return Citation(source=1, chunk_id=1, page_start=page_start, page_end=page_end)


def entry(
    question_id: str,
    evidence: list[dict[str, Any]],
    verified: bool = True,
) -> dict[str, Any]:
    return {
        "id": question_id,
        "type": "factual",
        "question": "What is it?",
        "answer": "It.",
        "evidence": evidence,
        "verified": verified,
    }


# cited_ok


def test_cited_ok_when_a_citation_range_contains_the_evidence_page() -> None:
    # Page 7 is inside 6-8 (not at an edge), and also at the edge of 7-7.
    assert cited_ok(True, [citation(6, 8)], [7])
    assert cited_ok(True, [citation(7, 7)], [7])


def test_cited_ok_includes_both_ends_of_the_range() -> None:
    assert cited_ok(True, [citation(5, 7)], [5])
    assert cited_ok(True, [citation(5, 7)], [7])


def test_cited_ok_false_when_no_citation_contains_an_evidence_page() -> None:
    # 4 is just before 5-7 and 8 just after: an off-by-one range check would pass one.
    assert not cited_ok(True, [citation(5, 7), citation(2, 3)], [4])
    assert not cited_ok(True, [citation(5, 7)], [8])


def test_cited_ok_false_without_citations() -> None:
    assert not cited_ok(True, [], [7])


def test_cited_ok_false_when_not_found_even_if_pages_match() -> None:
    assert not cited_ok(False, [citation(7, 7)], [7])


def test_cited_ok_true_when_one_of_two_evidence_pages_is_cited() -> None:
    assert cited_ok(True, [citation(9, 9)], [6, 9])
    assert cited_ok(True, [citation(6, 6)], [6, 9])


# retrieved_ok and normalisation


def test_normalise_casefolds_and_collapses_every_whitespace_run() -> None:
    assert normalise("  We  Use\tthe\n\nAdam   optimizer ") == "we use the adam optimizer"


def test_retrieved_ok_matches_across_case_and_whitespace_differences() -> None:
    sources = ["Unrelated text.", "...training. We used the Adam\noptimizer  [20] with beta1..."]
    assert retrieved_ok(sources, ["we used the ADAM optimizer"])


def test_retrieved_ok_false_when_the_evidence_text_is_absent() -> None:
    sources = ["We used the SGD optimizer.", "Another chunk about attention."]
    assert not retrieved_ok(sources, ["We used the Adam optimizer"])


def test_retrieved_ok_true_when_one_of_two_evidence_texts_is_present() -> None:
    sources = ["The second phrase is here."]
    assert retrieved_ok(sources, ["first phrase", "second phrase"])


def test_retrieved_ok_blank_evidence_matches_nothing() -> None:
    # "" is a substring of every string; it must not count as found.
    assert not retrieved_ok(["any text"], ["   "])


def test_retrieved_ok_false_without_sources() -> None:
    assert not retrieved_ok([], ["some phrase"])


# gate verdict


def test_gate_passes_at_eight_of_ten_and_fails_at_seven() -> None:
    assert gate_passed(8, 10)
    assert gate_passed(10, 10)
    assert not gate_passed(7, 10)


def test_gate_fails_with_fewer_than_ten_scored_questions() -> None:
    # 5/5 is 100%, but the gate needs ten questions.
    assert not gate_passed(5, 5)


# loader


def test_loader_keeps_a_verified_entry_with_filled_evidence() -> None:
    data = {"questions": [entry("q01", [{"page": 7, "text": "We used Adam."}])]}

    scorable, skipped = parse_questions(data)

    assert skipped == []
    assert [question.id for question in scorable] == ["q01"]
    assert scorable[0].evidence[0].page == 7
    assert scorable[0].evidence[0].text == "We used Adam."


def test_loader_skips_unverified_entries() -> None:
    data = {"questions": [entry("q01", [{"page": 7, "text": "We used Adam."}], verified=False)]}

    scorable, skipped = parse_questions(data)

    assert scorable == []
    assert [(item.id, item.reason) for item in skipped] == [("q01", NOT_VERIFIED)]


@pytest.mark.parametrize(
    "evidence",
    [
        [{"page": None, "text": "We used Adam."}],
        [{"page": 7, "text": "TO FILL: exact phrase from the PDF"}],
        [{"page": None, "text": "TO FILL: exact phrase from the PDF"}],
        # One filled item is not enough: every evidence item must be filled.
        [{"page": 7, "text": "We used Adam."}, {"page": None, "text": "TO FILL: phrase"}],
        [{"page": 0, "text": "We used Adam."}],
        [{"page": 7, "text": "   "}],
    ],
)
def test_loader_refuses_verified_entries_with_unfilled_evidence(
    evidence: list[dict[str, Any]],
) -> None:
    scorable, skipped = parse_questions({"questions": [entry("q01", evidence)]})

    assert scorable == []
    assert [(item.id, item.reason) for item in skipped] == [("q01", EVIDENCE_NOT_FILLED)]


def test_loader_keeps_file_order_and_splits_a_mixed_file() -> None:
    data = {
        "questions": [
            entry("q01", [{"page": 3, "text": "a"}]),
            entry("q02", [{"page": 4, "text": "b"}], verified=False),
            entry("q03", [{"page": 5, "text": "c"}]),
        ]
    }

    scorable, skipped = parse_questions(data)

    assert [question.id for question in scorable] == ["q01", "q03"]
    assert [item.id for item in skipped] == ["q02"]


def test_loader_rejects_a_missing_key() -> None:
    broken = entry("q01", [{"page": 7, "text": "a"}])
    del broken["verified"]

    with pytest.raises(QuestionFileError, match="verified"):
        parse_questions({"questions": [broken]})


def test_loader_rejects_repeated_ids() -> None:
    item = entry("q01", [{"page": 7, "text": "a"}])

    with pytest.raises(QuestionFileError, match="repeated"):
        parse_questions({"questions": [item, item]})


def test_load_questions_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "paper.json"
    data = {
        "paper": "paper.pdf",
        "questions": [entry("q01", [{"page": 2, "text": "x"}], verified=False)],
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    scorable, skipped = load_questions(path)

    assert scorable == []
    assert [item.id for item in skipped] == ["q01"]


def test_load_questions_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(QuestionFileError, match="not valid JSON"):
        load_questions(path)
