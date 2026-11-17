"""The recursive strategy: cut at paragraph and sentence boundaries, keep tables apart.

Uses the same size budget as fixed-size on purpose: the benchmark should measure where
the cuts go, not how big the chunks are.
"""

from adaptrag.chunking.chunks import TextChunk
from adaptrag.chunking.fixed import DEFAULT_OVERLAP_WORDS, DEFAULT_SIZE_WORDS
from adaptrag.chunking.segments import DEFAULT_TABLE_TOKEN_BUDGET, chunk_blocks
from adaptrag.embedding.base import Embedder
from adaptrag.parsing.blocks import ParsedDocument
from adaptrag.strategies.base import Strategy

# One dict feeds both the stored params and the splitter, so they can never disagree.
# Bump "version" whenever the separators or the table handling (chunking/segments.py)
# change: they are code, not params, but they change the chunks, so they must still change
# the cache key.
PARAMS: dict[str, int] = {
    "size_words": DEFAULT_SIZE_WORDS,
    "overlap_words": DEFAULT_OVERLAP_WORDS,
    "version": 1,
}

TABLE_TOKEN_BUDGET_PARAM = "table_token_budget"


class RecursiveStrategy(Strategy):
    """Recursive splitting of the prose between tables; each table is chunked on its own.

    table_token_budget: DEFAULT_TABLE_TOKEN_BUDGET (512, what the registry builds) keeps
    every table chunk within that many model tokens and repeats the header row (note 0025).
    None turns it off: tables are then limited by the word budget only.
    """

    name = "recursive"

    def __init__(
        self, embedder: Embedder, table_token_budget: int | None = DEFAULT_TABLE_TOKEN_BUDGET
    ) -> None:
        super().__init__(embedder)
        if table_token_budget is not None and table_token_budget < 1:
            raise ValueError(f"table_token_budget must be 1 or more, got {table_token_budget}")
        self._table_token_budget = table_token_budget

    @property
    def params(self) -> dict[str, int | float | str | bool]:
        # A copy, so a caller changing the result cannot change the cache key.
        params: dict[str, int | float | str | bool] = dict(PARAMS)
        # Left out when the budget is off, so chunk sets stored without it still match.
        if self._table_token_budget is not None:
            params[TABLE_TOKEN_BUDGET_PARAM] = self._table_token_budget
        return params

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        pieces = chunk_blocks(
            doc.blocks,
            PARAMS["size_words"],
            PARAMS["overlap_words"],
            self._table_token_budget,
            self.embedder.count_tokens,
        )
        return [
            TextChunk(index=index, text=text, page_start=page_start, page_end=page_end)
            for index, (text, page_start, page_end) in enumerate(pieces)
        ]
