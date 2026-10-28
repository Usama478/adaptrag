# 0001: Framework, vector store, metadata store

Date: 2026-10-17
Status: SQLite choice superseded by 0004 (PostgreSQL). FastAPI and Chroma still stand.

## 1. Web framework

**Chose:** FastAPI
**Rejected:** Flask, Django
**Why:**
- It validates request data automatically using Python type hints
- It supports async, which helps with background ingestion and streaming answers
- It generates an interactive `/docs` page for free (I used it in Step 4)
- Flask has no built-in validation. Django is heavier than needed for an API-only backend.

## 2. Vector store (where embeddings are stored)

**Chose:** Chroma
**Rejected:** FAISS, pgvector, Qdrant
**Why:**
- It stores metadata (document id, page, section, strategy) next to each vector
- It can delete by metadata filter, which I need for "delete means delete"
- It runs locally with no separate server
- FAISS only stores vectors, so I'd have to build my own metadata layer.

## 3. Metadata store (documents, notebooks, query logs)

**Chose:** SQLite
**Rejected:** Postgres (for now)
**Why:**
- Zero setup, a single file, and enough for one instance of the app
- Python includes it
- **Revisit** if I deploy with multiple workers or need many writers at once.