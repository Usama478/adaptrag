# AdaptRAG

A document-aware RAG notebook. It profiles each uploaded document and picks a chunking and retrieval strategy for it. (Work in progress: Stage 0 complete.)

## Quick start

```bash
git clone https://github.com/Usama478/adaptrag
cd adaptrag
cp .env.example .env
docker compose up --build
```

Then open http://localhost:8000/health

## Development (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy
python -m uvicorn adaptrag.main:app --reload
```