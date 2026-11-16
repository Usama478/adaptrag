"""The LLM usage log and the per-stage token budget (.claude/AUTONOMOUS.md, note 0023).

Every script that calls the LLM wraps it in `RecordingLLM`, which appends one JSON line
per call to `eval/llm_usage.jsonl`. Before its first call, a script adds up what the stage
has used so far and refuses to start if its own estimate would cross the stage budget.
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from adaptrag.llm.base import LLM, TokenUsage

# Per stage, set by the owner in AUTONOMOUS.md.
MAX_STAGE_INPUT_TOKENS = 10_000_000
MAX_STAGE_OUTPUT_TOKENS = 1_000_000

_RECORD_KEYS = ("time", "stage", "script", "model", "input_tokens", "output_tokens")


class UsageBudgetError(RuntimeError):
    """Running would cross the stage's token budget, so the script must not start."""


class UsageLogError(ValueError):
    """The usage log has a line that cannot be read, so the totals cannot be trusted."""


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """One LLM call. `time` is an ISO 8601 UTC timestamp."""

    time: str
    stage: str
    script: str
    model: str
    input_tokens: int
    output_tokens: int


def append_usage(path: Path, record: UsageRecord) -> None:
    """Add one line to the log, creating the file and its folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(asdict(record)) + "\n")


def read_usage(path: Path) -> list[UsageRecord]:
    """Every record in the log, oldest first. A missing file means nothing was used yet.

    A broken line raises UsageLogError: skipping it would under-count the usage.
    """
    if not path.exists():
        return []
    records: list[UsageRecord] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UsageLogError(f"{path} line {number} is not valid JSON") from exc
        if not isinstance(data, dict) or set(data) != set(_RECORD_KEYS):
            raise UsageLogError(f"{path} line {number} must have exactly {_RECORD_KEYS}")
        for key in ("input_tokens", "output_tokens"):
            value = data[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise UsageLogError(f"{path} line {number}: {key} must be a whole number >= 0")
        records.append(UsageRecord(**data))
    return records


def stage_totals(records: Iterable[UsageRecord], stage: str) -> tuple[int, int]:
    """(input tokens, output tokens) used so far by every script of this stage."""
    used_input = 0
    used_output = 0
    for record in records:
        if record.stage == stage:
            used_input += record.input_tokens
            used_output += record.output_tokens
    return used_input, used_output


def check_budget(
    used_input: int, used_output: int, estimated_input: int, estimated_output: int
) -> None:
    """Raise UsageBudgetError if the used tokens plus the estimate cross either limit."""
    if used_input + estimated_input > MAX_STAGE_INPUT_TOKENS:
        raise UsageBudgetError(
            f"input tokens: {used_input} used + {estimated_input} estimated "
            f"> {MAX_STAGE_INPUT_TOKENS} allowed for the stage"
        )
    if used_output + estimated_output > MAX_STAGE_OUTPUT_TOKENS:
        raise UsageBudgetError(
            f"output tokens: {used_output} used + {estimated_output} estimated "
            f"> {MAX_STAGE_OUTPUT_TOKENS} allowed for the stage"
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RecordingLLM:
    """Wraps an `LLM` and logs one line per call, including calls that fail.

    It implements the `LLM` protocol itself, so evaluation code cannot tell the
    difference. A call whose usage is unknown is logged with 0 tokens and counted in
    `calls_without_usage`.
    """

    def __init__(
        self,
        inner: LLM,
        path: Path,
        stage: str,
        script: str,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.model_name = inner.model_name
        self.last_usage: TokenUsage | None = None
        self.calls = 0
        self.calls_without_usage = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._inner = inner
        self._path = path
        self._stage = stage
        self._script = script
        self._now = now

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        self.last_usage = None
        try:
            return self._inner.complete(system, user)
        finally:
            # The inner LLM resets its usage at the start of a call, so after a failure
            # this is either the failed call's own usage or None.
            self._record(self._inner.last_usage)

    def _record(self, usage: TokenUsage | None) -> None:
        self.last_usage = usage
        if usage is None:
            self.calls_without_usage += 1
            usage = TokenUsage(input_tokens=0, output_tokens=0)
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        append_usage(
            self._path,
            UsageRecord(
                time=self._now().isoformat(),
                stage=self._stage,
                script=self._script,
                model=self.model_name,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ),
        )
