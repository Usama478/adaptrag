"""FixedSizeStrategy must match today's hard-wired fixed-size code exactly (pure tests)."""

import json

from adaptrag.chunking.fixed import STRATEGY_NAME, chunk_fixed_size
from adaptrag.embedding.fake import FakeEmbedder
from adaptrag.ingest import FIXED_SIZE_PARAMS
from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument
from adaptrag.strategies.fixed_size import FixedSizeStrategy
from adaptrag.strategies.registry import build_strategy, strategy_names


def words(start: int, stop: int) -> str:
    """Numbered words like "w0 w1 w2", so every chunk's text is different."""
    return " ".join(f"w{i}" for i in range(start, stop))


def two_page_doc() -> ParsedDocument:
    """Four paragraphs of 150 words over 2 pages: more than one chunk's worth of words."""
    blocks = tuple(
        ParsedBlock(
            text=words(start, start + 150),
            page=page,
            block_type=BlockType.PARAGRAPH,
        )
        for page, start in [(1, 0), (1, 150), (2, 300), (2, 450)]
    )
    return ParsedDocument(page_count=2, parser_version="test", blocks=blocks)


def test_name_equals_the_strategy_string_ingest_stores() -> None:
    assert FixedSizeStrategy.name == STRATEGY_NAME


def test_params_equal_todays_settings_and_survive_json() -> None:
    params = FixedSizeStrategy(FakeEmbedder()).params

    assert params == FIXED_SIZE_PARAMS
    assert json.loads(json.dumps(params)) == params


def test_chunk_matches_todays_chunker_exactly() -> None:
    doc = two_page_doc()
    expected = chunk_fixed_size(doc, **FIXED_SIZE_PARAMS)

    # The document must really produce several chunks over both pages, or this proves little.
    assert len(expected) > 1
    assert (expected[0].page_start, expected[-1].page_end) == (1, 2)
    assert FixedSizeStrategy(FakeEmbedder()).chunk(doc) == expected


def test_registry_builds_fixed_size_strategy() -> None:
    strategy = build_strategy(STRATEGY_NAME, FakeEmbedder())

    assert type(strategy) is FixedSizeStrategy
    assert STRATEGY_NAME in strategy_names()
