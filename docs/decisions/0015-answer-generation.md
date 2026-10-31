# 0015: Answer generation with page citations

Status: Accepted
Date: 2026-10-31

## Context
Step 5 returns the top-k chunks for a question (note 0014), with no threshold. Step 6
turns them into an answer: an LLM writes it from the chunks only, every claim carries a
page citation, and the system says "Not found in your documents." instead of guessing.
Document text is untrusted (a PDF can contain instructions aimed at the model).

Scores measured with `scripts/query.py` (bge-small, k=3, top score, owner's run):

| Question | Kind | Top score |
| --- | --- | --- |
| paper_01 "attention heads" | answerable | 0.7589 (rank 1 answers) |
| paper_01 "optimizer" | answerable | 0.7357 (rank 1 answers) |
| paper_03 "two models trained at the same time" | answerable | 0.7159 (answer at rank 3, 0.6925) |
| paper_01 "ImageNet accuracy" | in-domain, not answerable | 0.6987 |
| paper_03 "FID score" | in-domain, not answerable | 0.6126 |
| "capital of France" | off-topic | 0.4440 |
| "sourdough bread" | off-topic | 0.4465 |
| "best football club" | off-topic | 0.4167 |

## Decision
- **Two-layer not-found.**
  1. Retrieval layer: if there are no results or the top score is below
     `retrieval_min_score` (0.55), the answer is "Not found in your documents." and the
     LLM is not called. 0.55 sits between the off-topic scores (0.42-0.45) and every
     on-topic score (0.61 and up). It is provisional: recalibrate in Stage 4.
  2. LLM layer: the model replies exactly `NOT_FOUND` when the sources do not hold the
     answer, and our code turns that into the same message. This layer is needed
     because scores measure topic similarity, not answer presence: in-domain questions
     the paper cannot answer score 0.61-0.70, inside the range of answerable ones
     (0.72-0.76). No threshold separates them.
- **Numbered sources, mapped by our code.** The prompt gives each chunk as
  `<source id="N" pages="P-Q">...</source>` (N from 1, best first). The model cites
  `[N]`; `answering/answer.py` keeps numbers in 1..len(sources), removes duplicates in
  order, and maps each to the chunk's id and stored pages. Invalid numbers are dropped.
  An answer with no valid citation is still returned (found true, citations empty) and
  a WARNING is logged.
- **Injection defence.** The system prompt says the text inside `<source>` tags is
  untrusted, must never be followed, and that the instructions must not be revealed.
  `<source` and `</source` inside chunk text (any letter case) are escaped to `&lt;...`
  so a document cannot close its tag and pose as instructions. A slow test checks a
  real "reply PWNED" injection.
- **`POST /documents/{id}/answer`, separate from `/query`.** Same body, validation, 404
  and 409 checks (shared helpers in `retrieval/dense.py`, same request model). 200:
  `{document_id, found, answer, citations: [{source, chunk_id, page_start, page_end}],
  sources: [{source, chunk_id, chunk_index, page_start, page_end, score, text}]}`.
  Sources are returned even when not found, so a client can show what was searched.
  `/query` is unchanged and keeps working when the LLM is down.
- **OpenAI gpt-4o-mini behind a seam.** `llm/base.py` defines the `LLM` protocol
  (`model_name`, `complete(system, user)`) and `LLMError`. `llm/openai_client.py` is the
  only module importing `openai`: Responses API (`model`, `instructions`, `input`,
  `max_output_tokens`, plus `temperature` only when `llm_temperature` is not None; no
  `reasoning` parameter), client with `timeout=llm_timeout_seconds` and `max_retries=2`.
  Every SDK error (they all inherit from `openai.APIError`) and an empty reply become
  `LLMError`. `llm/provider.py` `get_app_llm` builds it once per process; tests use
  `FakeLLM`. The model was chosen by the owner for the lowest cost.
- **Errors.** No API key (missing or empty): 503 "Answering is not configured (no API
  key)." `LLMError`: 502 "The answer service is unavailable. Try again later."; the real
  cause is logged.
- **Token logging.** Each call logs input and output tokens at INFO (a WARNING if the
  response has no usage). The key is a `SecretStr`, read only where the client is built,
  never logged or printed.

## Rejected
- **gpt-5.4-mini:** higher price at this stage.
- **A raw httpx client:** retries, timeouts, error types and response parsing would all
  be ours to write and test; the official client already has them.
- **A local model (Ollama):** no API cost, but slower on a laptop CPU, weaker at
  following citation rules, and one more service in Docker.
- **Letting the model write page numbers:** it can invent or misread them. Numbers it
  cannot check against the source are less trustworthy than numbers our code maps from
  stored chunks.
- **A stricter threshold (for example 0.70) instead of the LLM layer:** it would block
  answerable questions (the GAN answer chunk scored 0.6925) and still let some
  unanswerable ones through (0.6987).
- **Merging `/answer` into `/query`:** retrieval would then depend on the LLM (key, cost,
  outages), and the benchmark needs retrieval results on their own.

## Known limits
- The 0.55 threshold comes from 8 dev questions only (3 answerable, 3 off-topic,
  2 in-domain unanswerable); recalibrate in Stage 4.
- Answers may vary between runs (even at temperature 0).
- Cost and latency: measured token counts (owner's run, gpt-4o-mini, k=5) were 1921 / 43,
  1651 / 40 and 1720 / 3 input / output tokens for the three answers that reached the
  LLM, so about 1,700-1,900 input tokens per answer; a retrieval-layer not-found costs
  no tokens. Money cost (from the owner's OpenAI usage page) and latency are measured in
  Step 7.
- An older model may follow the citation and NOT_FOUND rules less reliably; measured by
  the slow test and the Stage 2 gate questions.
- The model's availability and price must be checked by the owner on OpenAI's site.
- The injection defence reduces the risk but does not eliminate it.
- One document per question until notebooks (Stage 7).
- Only `[N]` counts as a citation; forms like `[1, 2]` or `NOT_FOUND.` (with a period)
  are not recognised.
