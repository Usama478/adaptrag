"""A fake embedder for tests: deterministic, instant, no model and no network.

Vectors come from a SHA-256 hash of the text, so the same text always gives the same
vector and different texts almost always give different ones. The numbers carry no
meaning: use the real model (slow tests) to check that similar texts score higher.
"""

import hashlib
import math
from collections.abc import Sequence

FAKE_QUERY_PREFIX = "query: "


class FakeEmbedder:
    """Implements the `Embedder` protocol without loading a model."""

    def __init__(self, dimension: int = 8, max_tokens: int = 512) -> None:
        if not 1 <= dimension <= 32:
            # One SHA-256 digest has 32 bytes; one byte per dimension.
            raise ValueError(f"dimension must be between 1 and 32, got {dimension}")
        self.model_name = "fake"
        self.dimension = dimension
        self.max_tokens = max_tokens

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Bytes 0..255 shifted to -127.5..127.5, so a vector is never all zeros.
        raw = [byte - 127.5 for byte in digest[: self.dimension]]
        length = math.sqrt(sum(x * x for x in raw))
        return [x / length for x in raw]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        # Like the real embedder, a query gets a prefix, so it differs from the passage.
        return self._vector(FAKE_QUERY_PREFIX + text)

    def count_tokens(self, text: str) -> int:
        # Words plus 2, standing in for the real model's [CLS] and [SEP] tokens.
        return len(text.split()) + 2
