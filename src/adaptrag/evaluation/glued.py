"""A general glued-word detector (Stage 4, note 0025). It measures; it fixes nothing.

The parser sometimes joins two words at a line break ("shownin", "theWMT"). Instead of a
fixed list of known cases, a token (a run of letters) is a CANDIDATE if either:
- two words: it is not in the embedding model's tokenizer vocabulary, and it can be cut
  at one point into two vocabulary words of at least 2 letters each; or
- case join: a lowercase letter is directly followed by an uppercase one ("theWMT").

Candidates are not proof. Real compounds missing from the vocabulary and product names
written in camel case are flagged too, so the report shows examples for a human to judge.
"""

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass

MIN_PART_LETTERS = 2

TWO_WORDS = "two_words"
CASE_JOIN = "case_join"

_LETTER_RUN = re.compile(r"[A-Za-z]+")
_LOWER_THEN_UPPER = re.compile(r"[a-z][A-Z]")


@dataclass(frozen=True, slots=True)
class GluedToken:
    """One candidate, as written in the text, and the reason it was flagged."""

    token: str
    kind: str


def vocabulary_words(vocabulary: Iterable[str]) -> frozenset[str]:
    """The whole words of a tokenizer vocabulary, lowercased.

    Word pieces ("##ing"), special tokens ("[CLS]"), digits, punctuation and single
    letters are left out: none of them is evidence that a string is a real word.
    """
    return frozenset(
        entry.lower()
        for entry in vocabulary
        if len(entry) >= MIN_PART_LETTERS and entry.isascii() and entry.isalpha()
    )


def split_parts(token: str, words: Collection[str]) -> tuple[str, str] | None:
    """The two vocabulary words a token splits into, or None.

    None also when the token itself is a vocabulary word. The earliest cut wins.
    """
    lower = token.lower()
    if lower in words:
        return None
    for cut in range(MIN_PART_LETTERS, len(lower) - MIN_PART_LETTERS + 1):
        first, second = lower[:cut], lower[cut:]
        if first in words and second in words:
            return first, second
    return None


def glued_kind(token: str, words: Collection[str]) -> str | None:
    """TWO_WORDS, CASE_JOIN, or None for a token made of letters only."""
    if split_parts(token, words) is not None:
        return TWO_WORDS
    if _LOWER_THEN_UPPER.search(token):
        return CASE_JOIN
    return None


def find_glued(text: str, words: Collection[str]) -> list[GluedToken]:
    """Every candidate in the text, in order; a token that occurs twice is listed twice."""
    found: list[GluedToken] = []
    for token in _LETTER_RUN.findall(text):
        kind = glued_kind(token, words)
        if kind is not None:
            found.append(GluedToken(token=token, kind=kind))
    return found
