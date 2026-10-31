"""The Embedder seam: what the rest of the app may ask of an embedding model.

Code outside `embedding/` depends on this protocol, never on sentence-transformers,
so the model library can change (or be faked in tests) without touching callers.
"""

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Turns text into normalised vectors (unit length, so cosine similarity = dot product).

    model_name: identifies the model; part of the embedding cache key (Step 4).
    dimension: length of every vector this embedder returns.
    max_tokens: the model's input limit; longer text is cut off by the model.
    """

    model_name: str
    dimension: int
    max_tokens: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document chunks as plain text, one vector per text, in input order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (with the model's query instruction, if it has one)."""
        ...

    def count_tokens(self, text: str) -> int:
        """Number of model tokens in `text`, including the model's special tokens."""
        ...
