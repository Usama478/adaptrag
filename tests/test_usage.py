"""The LLM usage log, the stage budget guard and RecordingLLM (unit tests, FakeLLM)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from adaptrag.evaluation.usage import (
    MAX_STAGE_INPUT_TOKENS,
    MAX_STAGE_OUTPUT_TOKENS,
    RecordingLLM,
    UsageBudgetError,
    UsageLogError,
    UsageRecord,
    append_usage,
    check_budget,
    read_usage,
    stage_totals,
)
from adaptrag.llm.base import LLMError
from adaptrag.llm.fake import FakeLLM

FIXED_TIME = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


def record(stage: str = "stage4", input_tokens: int = 100, output_tokens: int = 20) -> UsageRecord:
    return UsageRecord(
        time="2026-11-16T12:00:00+00:00",
        stage=stage,
        script="generate_questions",
        model="fake-llm",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def recording(llm: FakeLLM, path: Path) -> RecordingLLM:
    return RecordingLLM(llm, path, "stage4", "test_script", now=lambda: FIXED_TIME)


def test_appended_records_are_read_back_in_order(tmp_path: Path) -> None:
    path = tmp_path / "new_folder" / "usage.jsonl"  # the folder does not exist yet
    first, second = record(input_tokens=1), record(input_tokens=2)

    append_usage(path, first)
    append_usage(path, second)

    assert read_usage(path) == [first, second]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2  # one line per call


def test_a_missing_log_means_nothing_was_used(tmp_path: Path) -> None:
    assert read_usage(tmp_path / "absent.jsonl") == []


def test_a_broken_line_is_refused_instead_of_skipped(tmp_path: Path) -> None:
    path = tmp_path / "usage.jsonl"
    append_usage(path, record())
    with path.open("a", encoding="utf-8") as log:
        log.write("this is not json\n")

    with pytest.raises(UsageLogError):
        read_usage(path)


def test_a_line_with_wrong_keys_or_a_negative_count_is_refused(tmp_path: Path) -> None:
    good = {
        "time": "t",
        "stage": "stage4",
        "script": "s",
        "model": "m",
        "input_tokens": 1,
        "output_tokens": 1,
    }
    missing_key = {key: value for key, value in good.items() if key != "model"}
    negative = {**good, "output_tokens": -1}

    for position, bad in enumerate((missing_key, negative)):
        path = tmp_path / f"usage_{position}.jsonl"
        path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        with pytest.raises(UsageLogError):
            read_usage(path)


def test_stage_totals_add_only_the_named_stage() -> None:
    records = [
        record("stage4", 100, 20),
        record("stage6", 5000, 900),
        record("stage4", 50, 5),
    ]

    assert stage_totals(records, "stage4") == (150, 25)
    assert stage_totals(records, "stage9") == (0, 0)


def test_budget_guard_allows_reaching_a_limit_exactly() -> None:
    check_budget(MAX_STAGE_INPUT_TOKENS - 10, 0, 10, 0)
    check_budget(0, MAX_STAGE_OUTPUT_TOKENS - 10, 0, 10)


def test_budget_guard_refuses_to_cross_the_input_limit() -> None:
    with pytest.raises(UsageBudgetError):
        check_budget(MAX_STAGE_INPUT_TOKENS - 10, 0, 11, 0)


def test_budget_guard_refuses_to_cross_the_output_limit() -> None:
    with pytest.raises(UsageBudgetError):
        check_budget(0, MAX_STAGE_OUTPUT_TOKENS - 10, 0, 11)


def test_recording_llm_logs_one_line_per_call(tmp_path: Path) -> None:
    path = tmp_path / "usage.jsonl"
    llm = recording(FakeLLM(reply="one two three"), path)

    reply = llm.complete("sys tem", "us er here")
    llm.complete("sys tem", "again")

    assert reply == "one two three"
    records = read_usage(path)
    assert len(records) == 2
    # FakeLLM counts words as tokens: 2 + 3 in, 3 out.
    assert records[0] == UsageRecord(
        time=FIXED_TIME.isoformat(),
        stage="stage4",
        script="test_script",
        model="fake-llm",
        input_tokens=5,
        output_tokens=3,
    )
    assert (llm.calls, llm.input_tokens, llm.output_tokens) == (2, 5 + 3, 3 + 3)
    assert llm.calls_without_usage == 0


def test_recording_llm_logs_a_failed_call_and_passes_the_error_on(tmp_path: Path) -> None:
    path = tmp_path / "usage.jsonl"
    llm = recording(FakeLLM(raise_error=True), path)

    with pytest.raises(LLMError):
        llm.complete("system", "user")

    records = read_usage(path)
    assert [(item.input_tokens, item.output_tokens) for item in records] == [(0, 0)]
    assert (llm.calls, llm.calls_without_usage) == (1, 1)


def test_fake_llm_gives_scripted_replies_in_order_then_the_default() -> None:
    llm = FakeLLM(reply="default", replies=["first", "second"])

    replies = [llm.complete("system", f"user {number}") for number in range(3)]

    assert replies == ["first", "second", "default"]
    assert llm.users == ["user 0", "user 1", "user 2"]
    assert llm.calls == 3
