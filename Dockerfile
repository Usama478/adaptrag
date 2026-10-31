FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# CPU-only PyTorch first, so installing sentence-transformers does not pull the
# default build with GPU/CUDA libraries we never use (note 0012).
# Its own layer, before the code is copied, so code changes do not reinstall it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
RUN pip install --no-cache-dir .

# Both folders are mount points for named volumes (docker-compose.yml): uploads and the
# Hugging Face model cache. They must exist and belong to appuser BEFORE a volume is
# mounted, because a new named volume copies the ownership of the image's folder.
# Otherwise the volume is owned by root and appuser cannot write to it (note 0013).
RUN useradd --create-home appuser \
    && mkdir -p /app/data /app/.cache/huggingface \
    && chown -R appuser:appuser /app/data /app/.cache
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "adaptrag.main:app", "--host", "0.0.0.0", "--port", "8000"]