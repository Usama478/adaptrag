# Code-only stage mode

Applies when the owner starts a stage with `/goal`. You write the WHOLE stage in one go:
code, tests and docs. You run nothing. When you finish, the owner runs the stage's run
checklist, and fixes follow in the same chat. Everything in `.claude/CLAUDE.md` still
applies (conventions, seams, evaluation rules, "coding agents never run commands"). This
file only overrides "one step per task": during a code-only stage, do every sub-step of
the stage file.

`.claude/settings.json` denies Bash, web access, `.env` and the test set. Treat the rules
below as binding even where settings.json cannot enforce them.

## You may
- Create and edit files in the repo.
- Read any file in the repo, and the installed package source in `.venv/` (to confirm a
  library's real API before using it: "Step 0").

## You must not
- Run any command, use the web, or start sub-agents.
- Read `.env`, the API key, or anything in `data/test_set/`. Never use test documents to
  choose rules or thresholds.
- Use a library API you have not confirmed by reading its installed source in `.venv/`. If
  a needed package is not installed, STOP and name the exact `pip install` line.
- Write a number that needs a run. Write `TO FILL from owner's run: <what>` instead.
- Weaken a test, skip or xfail one, or write a test that cannot fail.
- Edit a migration that already exists on `main`. Write new migrations by hand (no
  autogenerate is possible), following the existing ones in `migrations/versions/`.
- Change the dev/test split, the embedding model, the LLM model, or frozen question files
  of an earlier stage.

## Writing so the first run works
- Every new script runs with NO required arguments: the defaults are the dev corpus and the
  standard paths. Then the run checklist has exact commands with no placeholders.
- Scripts that call the LLM append one line per call (time, script, model, input tokens,
  output tokens) to `eval/llm_usage.jsonl`. Before starting, they add up this stage's
  usage and refuse to run if their own estimate would cross 10,000,000 input or 1,000,000
  output tokens for the stage. They print the totals at the end.
- Decisions that depend on results are written as rules in code or in the stage file
  BEFORE any run. Scripts print the rule's outcome (for example "ADOPT" or "KEEP"), so
  nothing gets tuned after seeing results.
- Re-read every file you changed at the end. Check imports, types (mypy strict), the import
  order rules in CLAUDE.md, and that each new test can fail for its own reason.

## Decisions
- Covered by the stage file: follow it.
- Not covered and cheap to reverse: pick the simplest library-first option and list it in
  the final report under "Decisions taken without the owner".
- Not covered and expensive to reverse: STOP and write a BLOCKED report (what is done,
  what is blocked, the question, the options with a recommendation).

## Final report (last message of the stage, exact headings)
1. Files changed (path + one line each)
2. Tests added: the exact number of new tests, and the promise each one protects
3. Run checklist: exact commands in order, marking the ones that cost money, with the
   expected test count (previous count + new tests)
4. Placeholders left (`TO FILL from owner's run`)
5. Decisions taken without the owner
6. Library facts confirmed by reading `.venv/` source (package, file, what was confirmed)
7. Anything you were unsure about
8. Explain-aloud questions for this stage
