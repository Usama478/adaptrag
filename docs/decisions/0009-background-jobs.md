# 0009: Background ingestion jobs

Date: 2026-10-30
Status: Accepted

## Decision
- FastAPI BackgroundTasks runs ingestion after the upload response is sent.
- All job logic lives in one plain function, `run_ingest_job(engine, upload_dir,
  job_id)` in `ingest.py`, so a queue can call it later unchanged.
- A separate `jobs` table (queued, running, succeeded, failed, error,
  finished_at): one row per processing attempt. Deleting a document deletes
  its jobs (ON DELETE CASCADE).
- `POST /documents` returns `job_id`; `GET /jobs/{id}` reports status and error.

## Rejected
- Parsing inside the request: slow uploads and timeout risk.
- A Redis-backed queue (RQ, Celery, arq) now: an extra server and worker for
  traffic the MVP doesn't have.
- Job status on `Document`: a document can be processed more than once.

## Known limits
- A server restart mid-job leaves it stuck at "running" (fix in Stage 9:
  mark stale running jobs failed on startup).
- No retries. Jobs run inside the API process.
