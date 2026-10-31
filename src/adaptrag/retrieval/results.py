"""The shape of one retrieval result, shared by the vector store, the API and scripts."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A stored chunk with its similarity to the question.

    score = 1 - cosine distance = cosine similarity: 1.0 means the same direction,
    0.0 unrelated, -1.0 opposite. Higher is better. No threshold is applied here
    (note 0014): Step 6 chooses one from measured scores.
    """

    chunk_id: int
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    score: float
