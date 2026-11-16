"""The real embedder: a sentence-transformers model running on the CPU.

This is the only module that imports sentence_transformers or torch. Loading a model
reads its weights from disk and builds it in memory, so `get_embedder` loads it once per
process, on first use, and returns the same object afterwards.
"""

from collections.abc import Sequence
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from adaptrag.embedding.base import Embedder


class SentenceTransformerEmbedder:
    """Implements the `Embedder` protocol with a sentence-transformers model."""

    def __init__(self, model_name: str, device: str, query_instruction: str) -> None:
        self.model_name = model_name
        self._query_instruction = query_instruction
        self._model = SentenceTransformer(model_name, device=device)

        dimension = self._model.get_embedding_dimension()
        max_tokens = self._model.max_seq_length
        if dimension is None or max_tokens is None:
            raise ValueError(f"model {model_name} does not report its dimension or max length")
        self.dimension: int = dimension
        self.max_tokens: int = max_tokens

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Plain lists keep numpy out of the Embedder interface.
        return [[float(x) for x in vector] for vector in vectors]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_instruction + text])[0]

    def count_tokens(self, text: str) -> int:
        # No truncation: the point is to see how far a text goes past max_tokens.
        token_ids = self._model.tokenizer(text, add_special_tokens=True, truncation=False)
        return len(token_ids["input_ids"])

    def vocabulary(self) -> frozenset[str]:
        """Every entry of the tokenizer's vocabulary (whole words and "##" word pieces).

        Not part of the `Embedder` protocol: only the Stage 4 glued-word detector needs it.
        """
        return frozenset(self._model.tokenizer.get_vocab())


@lru_cache
def get_embedder(model_name: str, device: str, query_instruction: str) -> Embedder:
    """Return the embedder for these settings, loading the model on the first call only."""
    return SentenceTransformerEmbedder(model_name, device, query_instruction)


def get_vocabulary(model_name: str, device: str, query_instruction: str) -> frozenset[str]:
    """The tokenizer vocabulary of the (cached) embedder for these settings."""
    embedder = get_embedder(model_name, device, query_instruction)
    if not isinstance(embedder, SentenceTransformerEmbedder):
        raise TypeError("the vocabulary is only available from the sentence-transformers embedder")
    return embedder.vocabulary()
