"""How the app gets its embedder: a FastAPI dependency.

Routes ask for `Depends(get_app_embedder)`. Tests replace it with a `FakeEmbedder`
through `app.dependency_overrides`, so normal tests never load the real model.
"""

from typing import Annotated

from fastapi import Depends

from adaptrag.config import Settings, get_settings
from adaptrag.embedding.base import Embedder


def get_app_embedder(settings: Annotated[Settings, Depends(get_settings)]) -> Embedder:
    """Return the real embedder (loaded once per process by `get_embedder`)."""
    # Imported here, not at the top: importing the app (or collecting tests) must not
    # import torch. The import only happens when a request actually needs the model.
    from adaptrag.embedding.sentence_transformer import get_embedder

    return get_embedder(
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_query_instruction,
    )
