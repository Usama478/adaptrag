"""Build the machine-verified question set for the dev documents (Stage 4, note 0024).

For each of the 9 dev documents in eval/corpus.json: choose passages from the parser
output with a fixed seed, let the LLM draft 20 questions, verify every draft by code
and with a judge prompt, and keep up to 10. Writes eval/questions/generated/<corpus id>.json.
The rules live in src/adaptrag/evaluation/passages.py and generation.py (unit-tested).
Prompt version stage4-v2 (note 0024); a file with another version stops the script.

- A document whose file already exists is skipped, so a run that stopped half-way can
  simply be started again.
- --extra-round: one more round for every document, only when a target was missed. The
  checks are never loosened.
- Calibration: checks 1, 5 and 4 also run on the 10 owner-verified paper_01 questions.
- Every LLM call is logged to eval/llm_usage.jsonl; the run refuses to start if its
  estimate would cross the stage's token budget.

Costs money (real LLM calls, temperature 0). Dev documents only.

Usage: python scripts/generate_questions.py [--extra-round]
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from adaptrag.config import get_settings
from adaptrag.evaluation.corpus import TEST_DIR, CorpusEntry, load_corpus
from adaptrag.evaluation.gate import load_questions
from adaptrag.evaluation.generation import (
    PROMPT_VERSION,
    REJECTION_REASONS,
    GenerationStats,
    calibrate,
    estimate_usage,
    files_with_other_version,
    generate_for_document,
    select_kept,
    targets,
)
from adaptrag.evaluation.passages import DraftSlot, plan_drafts
from adaptrag.evaluation.usage import (
    RecordingLLM,
    UsageBudgetError,
    check_budget,
    read_usage,
    stage_totals,
)
from adaptrag.llm.openai_client import get_openai_llm
from adaptrag.llm.provider import NOT_CONFIGURED
from adaptrag.parsing.blocks import ParsedDocument
from adaptrag.parsing.pdf import parse_pdf

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "eval" / "corpus.json"
OUTPUT_DIR = REPO_ROOT / "eval" / "questions" / "generated"
OWNER_SET = REPO_ROOT / "eval" / "questions" / "paper_01.json"
OWNER_DOCUMENT = "paper_01"
USAGE_LOG = REPO_ROOT / "eval" / "llm_usage.jsonl"

STAGE = "stage4"
SCRIPT = "generate_questions"
SEED = "stage4"
# The first round plus the one extra round the stage file allows.
MAX_ROUNDS = 2
GENERATOR_TEMPERATURE = 0.0
OVERLAP_BUCKETS = 5


def dev_entries() -> list[CorpusEntry]:
    """The dev documents of the manifest. A test-set path can never get through."""
    entries = [entry for entry in load_corpus(CORPUS) if entry.split == "dev"]
    for entry in entries:
        if entry.path.startswith(f"{TEST_DIR}/"):
            raise SystemExit(f"{entry.id}: test documents are not allowed here.")
        if not (REPO_ROOT / entry.path).exists():
            hint = "run python scripts/fetch_corpus.py"
            raise SystemExit(f"{entry.id}: {entry.path} is missing ({hint}).")
    return entries


def output_path(entry: CorpusEntry) -> Path:
    return OUTPUT_DIR / f"{entry.id}.json"


def read_file(entry: CorpusEntry) -> dict[str, Any] | None:
    path = output_path(entry)
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def write_file(entry: CorpusEntry, data: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    output_path(entry).write_text(text, encoding="utf-8")


def type_counts(files: list[dict[str, Any]]) -> Counter[str]:
    return Counter(question["type"] for data in files for question in data["questions"])


def stats_entry(round_number: int, stats: GenerationStats, kept: int) -> dict[str, Any]:
    """The counts of one round, stored in the question file so later runs can print them."""
    return {
        "round": round_number,
        "drafted": stats.drafted,
        "accepted": stats.accepted,
        "kept_after_round": kept,
        "rejected": {reason: stats.rejected[reason] for reason in REJECTION_REASONS},
        "overlaps": [round(value, 3) for value in stats.overlaps],
        "extra_candidates": stats.extra_candidates,
        "extra_evidence": stats.extra_evidence,
    }


def print_rejections(rejected: Counter[str]) -> None:
    for reason in REJECTION_REASONS:
        print(f"  {reason}: {rejected[reason]}")


def print_overlaps(overlaps: list[float]) -> None:
    """Check 5 input: how much of each question's wording comes from its evidence."""
    print(f"lexical overlap of {len(overlaps)} drafts that reached check 5:")
    for bucket in range(OVERLAP_BUCKETS):
        low = bucket / OVERLAP_BUCKETS
        high = (bucket + 1) / OVERLAP_BUCKETS
        last = bucket == OVERLAP_BUCKETS - 1
        count = sum(1 for value in overlaps if low <= value < high or (last and value == high))
        marker = "  (rejected: at or above 0.8)" if last else ""
        print(f"  {low:.1f}-{high:.1f}: {count}{marker}")


def count_answerable_without_document(questions: list[dict[str, Any]]) -> int:
    """Kept questions the judge labelled as answerable without the document."""
    return sum(1 for question in questions if question.get("answerable_without_document") is True)


def print_totals(files: list[dict[str, Any]]) -> None:
    """Totals over every round stored in the files (this run and earlier ones)."""
    rejected: Counter[str] = Counter()
    overlaps: list[float] = []
    drafted = accepted = candidates = extra = 0
    for data in files:
        for entry in data["generation"]:
            drafted += entry["drafted"]
            accepted += entry["accepted"]
            candidates += entry["extra_candidates"]
            extra += entry["extra_evidence"]
            rejected.update(entry["rejected"])
            overlaps.extend(entry["overlaps"])
    kept = sum(len(data["questions"]) for data in files)
    print()
    print(f"ALL DOCUMENTS: drafted {drafted}, passed verification {accepted}, kept {kept}")
    print("rejections per check:")
    print_rejections(rejected)
    print_overlaps(overlaps)
    print(f"extra evidence locations: {extra} confirmed of {candidates} candidates")
    labelled = count_answerable_without_document(
        [question for data in files for question in data["questions"]]
    )
    print(f"answerable without the document (judge label, not a rejection): {labelled} of {kept}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate machine-verified dev questions.")
    parser.add_argument(
        "--extra-round",
        action="store_true",
        help="one more generation round for every document (only if a target was missed)",
    )
    args = parser.parse_args()

    # WARNING: the INFO lines (one per LLM call) would bury the report; usage goes to the log.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    key = settings.openai_api_key
    if key is None or not key.get_secret_value().strip():
        raise SystemExit(NOT_CONFIGURED)

    entries = dev_entries()
    existing = {entry.id: read_file(entry) for entry in entries}

    # Never mix prompt versions: stop before anything is parsed or paid for.
    other = files_with_other_version(
        {
            output_path(entry).name: data.get("prompt_version")
            for entry in entries
            if (data := existing[entry.id]) is not None
        }
    )
    if other:
        raise SystemExit(
            f"These files in {OUTPUT_DIR.relative_to(REPO_ROOT)}/ were not written with prompt "
            f"version {PROMPT_VERSION}: {', '.join(other)}. Delete them (the script never "
            "mixes versions) and run it again."
        )

    # Decide the round of each document before anything is parsed or paid for.
    rounds: dict[str, int] = {}
    if args.extra_round:
        missing = [entry.id for entry in entries if existing[entry.id] is None]
        if missing:
            raise SystemExit(f"No question file yet for {missing}; run without --extra-round.")
        files = [data for data in existing.values() if data is not None]
        if all(target.met for target in targets(type_counts(files))):
            print("Every target is already met; --extra-round has nothing to do.")
        elif any(data["rounds"] >= MAX_ROUNDS for data in files):
            print("The one extra round was already used; the checks are not loosened.")
        else:
            rounds = {entry.id: files[position]["rounds"] for position, entry in enumerate(entries)}
    else:
        rounds = {entry.id: 0 for entry in entries if existing[entry.id] is None}
        for entry in entries:
            if entry.id not in rounds:
                print(f"{entry.id}: {output_path(entry).name} exists, skipped")

    # Parse and plan first, so the budget estimate uses the real number of drafts.
    parsed: dict[str, ParsedDocument] = {}
    plans: dict[str, list[DraftSlot]] = {}
    for entry in entries:
        if entry.id in rounds or entry.id == OWNER_DOCUMENT:
            parsed[entry.id] = parse_pdf(REPO_ROOT / entry.path)
        if entry.id in rounds:
            seed = f"{SEED}:{entry.id}:{rounds[entry.id]}"
            plans[entry.id] = plan_drafts(parsed[entry.id], seed)

    owner_questions, _skipped = load_questions(OWNER_SET)
    if OWNER_DOCUMENT not in parsed:
        raise SystemExit(f"{OWNER_DOCUMENT} is not a dev document in {CORPUS.name}.")

    draft_count = sum(len(slots) for slots in plans.values())
    estimated_input, estimated_output = estimate_usage(
        draft_count, len(owner_questions), settings.llm_max_output_tokens
    )
    used_input, used_output = stage_totals(read_usage(USAGE_LOG), STAGE)
    print(f"{STAGE} usage so far: {used_input} input, {used_output} output tokens")
    print(f"this run, upper estimate: {estimated_input} input, {estimated_output} output tokens")
    try:
        check_budget(used_input, used_output, estimated_input, estimated_output)
    except UsageBudgetError as exc:
        raise SystemExit(f"Refusing to run: {exc}") from exc

    llm = RecordingLLM(
        get_openai_llm(
            key.get_secret_value(),
            settings.llm_model,
            settings.llm_timeout_seconds,
            settings.llm_max_output_tokens,
            GENERATOR_TEMPERATURE,
        ),
        USAGE_LOG,
        STAGE,
        SCRIPT,
    )
    print(f"generator and judge: {llm.model_name}, prompts {PROMPT_VERSION}, seed {SEED!r}")

    for entry in entries:
        if entry.id not in rounds:
            continue
        round_number = rounds[entry.id]
        before = existing[entry.id]
        earlier = before["questions"] if before is not None else []
        records, stats = generate_for_document(
            llm,
            parsed[entry.id],
            plans[entry.id],
            id_prefix=f"{entry.id}_r{round_number}",
            round_number=round_number,
            earlier_questions=[question["question"] for question in earlier],
        )
        kept = select_kept([*earlier, *records])
        generation = before["generation"] if before is not None else []
        data = {
            "paper": Path(entry.path).name,
            "corpus_id": entry.id,
            "prompt_version": PROMPT_VERSION,
            "generator_model": llm.model_name,
            "seed": SEED,
            "rounds": round_number + 1,
            "questions": kept,
            "generation": [*generation, stats_entry(round_number, stats, len(kept))],
        }
        write_file(entry, data)
        existing[entry.id] = data
        print(
            f"{entry.id}: round {round_number}, drafted {stats.drafted}, "
            f"passed {stats.accepted}, kept {len(kept)} in total "
            f"({count_answerable_without_document(records)} of the new ones answerable "
            "without the document)"
        )
        print_rejections(stats.rejected)

    files = [data for data in existing.values() if data is not None]
    print_totals(files)

    print()
    print(f"calibration on the {len(owner_questions)} owner-verified {OWNER_DOCUMENT} questions")
    print("(checks 1, 5 and 4; the same functions and judge prompt as above):")
    calibration = calibrate(llm, parsed[OWNER_DOCUMENT], owner_questions)
    print(f"  accepted {calibration.accepted} of {calibration.total}")
    print_rejections(calibration.rejected)

    print()
    print("targets:")
    all_met = True
    for target in targets(type_counts(files)):
        verdict = "met" if target.met else "NOT MET"
        all_met = all_met and target.met
        print(f"  {target.name}: {target.have} (need at least {target.need}): {verdict}")
    if len(files) < len(entries):
        print("  (some documents have no question file yet)")
    elif not all_met and all(data["rounds"] < MAX_ROUNDS for data in files):
        print("A target was missed. One more round is allowed; the checks stay as they are:")
        print("  python scripts/generate_questions.py --extra-round")
    elif not all_met:
        print("A target was missed after the extra round. Report it as it is (note 0024).")

    print()
    print(f"LLM calls this run: {llm.calls} ({llm.calls_without_usage} without usage data)")
    print(f"tokens this run: {llm.input_tokens} input, {llm.output_tokens} output")
    total_input, total_output = stage_totals(read_usage(USAGE_LOG), STAGE)
    print(f"{STAGE} total: {total_input} input, {total_output} output tokens ({USAGE_LOG.name})")


if __name__ == "__main__":
    main()
