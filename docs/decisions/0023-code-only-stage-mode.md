# 0023: Code-only stage mode

Status: Accepted
Date: 2026-11-16

## Context
Stages 0-3 were built one step per task: the assistant wrote a step, the owner ran the
checks, and only then did the next step start. That kept every change small, but each stage
needed many chats. From Stage 4 on, the owner wants a whole stage written in one go and
checked once at the end. The rule that the assistant never runs commands (working mode since
Stage 2 Step 3) must stay, and so must the evaluation rules (test documents are never read).

## Decision
- **A code-only stage is started by the owner with `/goal`.** The assistant writes the whole
  stage (code, tests, docs) from a stage file, `.claude/stages/stageN.md`, and runs nothing.
- **`.claude/AUTONOMOUS.md` holds the rules of the mode.** It overrides exactly one rule of
  `.claude/CLAUDE.md`, "one step per task", and only during a code-only stage. Everything
  else in `CLAUDE.md` still applies.
- **The owner runs and checks each stage before the next one starts.** The stage ends with a
  run checklist (exact commands, in order, the ones that cost money marked, the expected
  test count). Fixes follow in the same chat.
- **`.claude/settings.json` enforces what a permission file can enforce:** it denies Bash,
  web access, reading or editing `.env`, and reading or editing `data/test_set/`. The other
  rules in `AUTONOMOUS.md` (no sub-agents, no weakened tests, no edits to existing
  migrations) are binding without enforcement.
- **Nothing runs during the stage, so the mode replaces running with three habits:**
  - A library API is used only after reading its installed source in `.venv/`. A missing
    package means stop and name the `pip install` line.
  - A number that needs a run is written as `TO FILL from owner's run: <what>`.
  - A decision that depends on results is written as a rule before any run, and the script
    prints the rule's outcome (for example ADOPT or KEEP), so nothing is tuned after seeing
    results.
- **Decisions the stage file does not cover:** cheap to reverse, the assistant picks the
  simplest option and lists it in the final report; expensive to reverse, the assistant
  stops and writes a BLOCKED report.
- **Spending limit: at most $5 of OpenAI usage per stage (owner's decision).** In code this
  is a token guard: scripts that call the LLM log every call to `eval/llm_usage.jsonl` and
  refuse to start if the stage would cross 10,000,000 input or 1,000,000 output tokens
  (`AUTONOMOUS.md`). Whether those token caps stay under $5 at the current gpt-4o-mini
  price: TO FILL from owner's check of OpenAI's pricing page.
- **The final report has its own headings** (listed in `AUTONOMOUS.md`): it adds the run
  checklist, the decisions taken without the owner, the library facts confirmed from
  `.venv/`, and the stage's explain-aloud questions.

## Rejected
- **Step by step for every remaining stage:** safest, but too slow for the remaining stages
  (owner's decision).
- **Letting the assistant run commands** (tests, scripts, docker): it would catch its own
  errors, but it breaks the working agreement that the owner runs everything, and scripts
  that call the LLM cost money.
- **Relying on `settings.json` alone:** a permission file cannot express rules such as "do
  not weaken a test" or "write the decision rule before the run".

## Known limits
- **The first run of a stage will find errors.** Code written without running it can fail on
  imports, types or wrong assumptions; the fixes happen after the owner's run.
- **The owner sees a whole stage at once,** which is harder to review and to explain than
  one step. The explain-aloud questions in the final report are the counterweight.
- **`AUTONOMOUS.md` rules that `settings.json` does not enforce depend on the assistant
  following them.** The owner's review of the diff is the check.
- **`settings.json` denies `Read(./.env.*)`, which also matches `.env.example`.** A stage
  that adds a setting cannot read that file to update it; the owner may have to add the
  line, or narrow the rule.
- **The token guard counts tokens, not dollars.** The $5 limit holds only as long as the
  token caps and the model's price agree; the real cost is on the owner's OpenAI usage page.
- **Branch name.** A code-only stage uses one branch per stage (`stage4-mini-evaluation`),
  not `stageN-stepM-short-name`.
