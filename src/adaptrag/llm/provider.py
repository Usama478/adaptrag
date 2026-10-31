"""How the app gets its LLM: a FastAPI dependency.

Routes ask for `Depends(get_app_llm)`. Tests replace it with a `FakeLLM` through
`app.dependency_overrides`, so normal tests never call a real model.
"""

from typing import Annotated

from fastapi import Depends, HTTPException

from adaptrag.config import Settings, get_settings
from adaptrag.llm.base import LLM

NOT_CONFIGURED = "Answering is not configured (no API key)."


def get_app_llm(settings: Annotated[Settings, Depends(get_settings)]) -> LLM:
    """Return the real LLM, or answer 503 if no API key is configured."""
    key = settings.openai_api_key
    # An empty line `ADAPTRAG_OPENAI_API_KEY=` in .env gives an empty SecretStr, not None.
    if key is None or not key.get_secret_value().strip():
        raise HTTPException(status_code=503, detail=NOT_CONFIGURED)

    # Imported here, like the embedder: only a request that needs the LLM imports openai.
    from adaptrag.llm.openai_client import get_openai_llm

    return get_openai_llm(
        key.get_secret_value(),
        settings.llm_model,
        settings.llm_timeout_seconds,
        settings.llm_max_output_tokens,
        settings.llm_temperature,
    )
