"""Choose the passages that question generation starts from (Stage 4, note 0024).

Passages come from the PARSER's blocks, never from any strategy's chunks, so no strategy
is favoured. Pure code: the same document and seed always give the same passages.

- A prose passage is a window of consecutive paragraph blocks on one page, inside one
  section: about 150-300 words. Headings are never part of a passage.
- A table passage is one table block (cut to whole rows), with its caption if the block
  next to it starts with "Table".
- A multi-section question gets two prose passages from pages far apart.
- Passages are spread over the whole document: the candidates are cut into equal
  stretches and one is drawn from each.
"""

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TypeVar

from adaptrag.chunking.splitter import word_count
from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument

FACTUAL = "factual"
TABLE = "table"
MULTI_SECTION = "multi_section"
QUESTION_TYPES = (FACTUAL, TABLE, MULTI_SECTION)

# Drafts per document (stage4_v2.md, change 6): 20 in total. Multi-section drafts are fixed;
# table drafts follow the number of usable tables; factual drafts fill the rest.
DRAFTS_PER_DOCUMENT = 20
MULTI_SECTION_DRAFTS = 3
MAX_TABLE_DRAFTS = 4
# A usable table can be used for this many drafts (stage4_v2.md, change 5).
TABLE_DRAFTS_PER_TABLE = 2

TARGET_WORDS = 150
MAX_WORDS = 300
MIN_WORDS = 60
MAX_CAPTION_WORDS = 60
MIN_TABLE_ROWS = 3

# Two passages of a multi-section question are at least this many pages apart, or a
# third of the document if that is more.
MIN_PAGE_GAP = 2

# Sections that hold no facts worth asking about.
_SKIPPED_HEADING = re.compile(r"\b(references|bibliography|acknowledge?ments?)\b", re.IGNORECASE)

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class Passage:
    """Text shown to the question generator. `first_block` is its position in the document."""

    text: str
    page: int
    kind: str
    first_block: int


@dataclass(frozen=True, slots=True)
class DraftSlot:
    """One question to draft: its type and the passage (two for multi-section).

    variant: 0 for the first draft from a passage, 1 for the second draft from the same
    table. The generator is told about it, because with temperature 0 the same message
    would give the same question twice.
    """

    type: str
    passages: tuple[Passage, ...]
    variant: int = 0


def _in_skipped_section(block: ParsedBlock) -> bool:
    return any(_SKIPPED_HEADING.search(heading) for heading in block.heading_path)


def _first_words(text: str, limit: int) -> str:
    """The text unchanged if it is short enough, else its first `limit` words."""
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


def _windows_from_run(run: Sequence[tuple[int, ParsedBlock]]) -> list[Passage]:
    """Cut a run of consecutive paragraphs (one page, one section) into windows."""
    windows: list[Passage] = []
    current: list[tuple[int, ParsedBlock]] = []
    words = 0

    def close() -> None:
        nonlocal current, words
        if current:
            # Only a single block can be over the limit; cutting it keeps an exact prefix.
            text = _first_words("\n\n".join(block.text for _, block in current), MAX_WORDS)
            if word_count(text) >= MIN_WORDS:
                first_index, first_block = current[0]
                windows.append(
                    Passage(text=text, page=first_block.page, kind="prose", first_block=first_index)
                )
        current = []
        words = 0

    for index, block in run:
        block_words = word_count(block.text)
        if current and words + block_words > MAX_WORDS:
            close()
        current.append((index, block))
        words += block_words
        if words >= TARGET_WORDS:
            close()
    close()
    return windows


def prose_windows(doc: ParsedDocument) -> list[Passage]:
    """Every prose passage of the document, in document order."""
    windows: list[Passage] = []
    run: list[tuple[int, ParsedBlock]] = []
    for index, block in enumerate(doc.blocks):
        usable = block.block_type is BlockType.PARAGRAPH and not _in_skipped_section(block)
        # A heading, a table, a skipped block or a new page ends the run.
        if run and (not usable or block.page != run[-1][1].page):
            windows.extend(_windows_from_run(run))
            run = []
        if usable:
            run.append((index, block))
    windows.extend(_windows_from_run(run))
    return windows


def _table_text(text: str) -> str:
    """The table cut to whole rows within MAX_WORDS (always at least its first row)."""
    rows = text.split("\n")
    kept: list[str] = []
    words = 0
    for row in rows:
        row_words = word_count(row)
        if kept and words + row_words > MAX_WORDS:
            break
        kept.append(row)
        words += row_words
    return "\n".join(kept)


def _caption(doc: ParsedDocument, index: int) -> str | None:
    """The paragraph right before or after a table, on its page, that starts with "Table"."""
    table = doc.blocks[index]
    for neighbour_index in (index - 1, index + 1):
        if not 0 <= neighbour_index < len(doc.blocks):
            continue
        neighbour = doc.blocks[neighbour_index]
        if (
            neighbour.block_type is BlockType.PARAGRAPH
            and neighbour.page == table.page
            and neighbour.text.lower().startswith("table")
        ):
            return _first_words(neighbour.text, MAX_CAPTION_WORDS)
    return None


def table_passages(doc: ParsedDocument) -> list[Passage]:
    """One passage per usable table (enough rows, at least one digit), in document order."""
    passages: list[Passage] = []
    for index, block in enumerate(doc.blocks):
        if block.block_type is not BlockType.TABLE or _in_skipped_section(block):
            continue
        if len(block.text.split("\n")) < MIN_TABLE_ROWS:
            continue
        if not any(character.isdigit() for character in block.text):
            continue
        text = _table_text(block.text)
        caption = _caption(doc, index)
        if caption is not None:
            text = f"{caption}\n\n{text}"
        passages.append(Passage(text=text, page=block.page, kind="table", first_block=index))
    return passages


def spread_pick(items: Sequence[_T], count: int, rng: random.Random) -> list[_T]:
    """Up to `count` items: one drawn from each of `count` equal stretches, in order."""
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    picked: list[_T] = []
    for stretch in range(count):
        start = stretch * len(items) // count
        end = (stretch + 1) * len(items) // count
        picked.append(items[rng.randrange(start, end)])
    return picked


def far_apart_pairs(
    windows: Sequence[Passage], page_count: int, count: int, rng: random.Random
) -> list[tuple[Passage, Passage]]:
    """Up to `count` pairs of prose passages from pages far apart; no passage used twice."""
    min_gap = max(MIN_PAGE_GAP, page_count // 3)
    pairs = [
        (first, second)
        for first, second in combinations(windows, 2)
        if second.page - first.page >= min_gap
    ]
    rng.shuffle(pairs)
    chosen: list[tuple[Passage, Passage]] = []
    used: set[int] = set()
    for first, second in pairs:
        if len(chosen) == count:
            break
        if first.first_block in used or second.first_block in used:
            continue
        chosen.append((first, second))
        used.update((first.first_block, second.first_block))
    return chosen


def table_draft_count(usable_tables: int) -> int:
    """How many table drafts a document gets: min(4, 2 x usable tables)."""
    return min(MAX_TABLE_DRAFTS, TABLE_DRAFTS_PER_TABLE * max(0, usable_tables))


def table_slots(tables: Sequence[Passage], rng: random.Random) -> list[DraftSlot]:
    """Table drafts for the usable tables: each table once, then a second draft from the
    first tables of the pick until `table_draft_count` is reached."""
    count = table_draft_count(len(tables))
    chosen = spread_pick(tables, count, rng)
    slots = [DraftSlot(type=TABLE, passages=(passage,)) for passage in chosen]
    for passage in chosen[: count - len(chosen)]:
        slots.append(DraftSlot(type=TABLE, passages=(passage,), variant=1))
    return slots


def plan_drafts(
    doc: ParsedDocument,
    seed: str,
    drafts: int = DRAFTS_PER_DOCUMENT,
    multi_section: int = MULTI_SECTION_DRAFTS,
) -> list[DraftSlot]:
    """The questions to draft for one document: factual, then table, then multi-section.

    Table drafts: min(4, 2 x usable tables). Factual drafts fill the rest of `drafts`, so a
    document with few tables still gets 20 drafts (if it has enough prose and page gaps).
    """
    rng = random.Random(seed)
    windows = prose_windows(doc)
    tables = table_slots(table_passages(doc), rng)
    prose = spread_pick(windows, max(0, drafts - multi_section - len(tables)), rng)
    pairs = far_apart_pairs(windows, doc.page_count, multi_section, rng)

    slots = [DraftSlot(type=FACTUAL, passages=(passage,)) for passage in prose]
    slots.extend(tables)
    slots.extend(DraftSlot(type=MULTI_SECTION, passages=pair) for pair in pairs)
    return slots
