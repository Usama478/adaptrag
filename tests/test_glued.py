"""The glued-word detector, with a small fake vocabulary (pure unit tests)."""

from adaptrag.evaluation.glued import (
    CASE_JOIN,
    TWO_WORDS,
    GluedToken,
    find_glued,
    glued_kind,
    split_parts,
    vocabulary_words,
)

# Shaped like a real tokenizer vocabulary: words, word pieces, special tokens, a capital.
FAKE_VOCABULARY = [
    "The",
    "decoder",
    "shown",
    "in",
    "to",
    "into",
    "an",
    "attention",
    "wmt",
    "model",
    "a",
    "##ing",
    "[CLS]",
    "x1",
    "!",
]
WORDS = vocabulary_words(FAKE_VOCABULARY)


def test_vocabulary_words_keeps_only_whole_lowercased_words() -> None:
    expected = {"the", "decoder", "shown", "in", "to", "into", "an", "attention", "wmt", "model"}
    assert WORDS == expected


def test_two_vocabulary_words_glued_together_are_a_candidate() -> None:
    assert split_parts("Thedecoder", WORDS) == ("the", "decoder")
    assert glued_kind("Thedecoder", WORDS) == TWO_WORDS
    assert glued_kind("shownin", WORDS) == TWO_WORDS


def test_a_vocabulary_word_is_never_a_candidate() -> None:
    # "into" = "in" + "to", but it is a word itself.
    assert split_parts("into", WORDS) is None
    assert glued_kind("into", WORDS) is None
    assert glued_kind("Attention", WORDS) is None


def test_each_part_needs_at_least_two_letters() -> None:
    # "a" + "model" would split, but a single letter proves nothing.
    assert glued_kind("amodel", WORDS) is None


def test_an_unknown_word_that_does_not_split_is_not_a_candidate() -> None:
    assert glued_kind("transformer", WORDS) is None


def test_three_words_glued_are_not_a_two_word_split() -> None:
    assert split_parts("theanattention", WORDS) is None


def test_lowercase_followed_by_uppercase_is_a_case_join() -> None:
    assert glued_kind("zzQQ", WORDS) == CASE_JOIN
    assert glued_kind("QQzz", WORDS) is None  # uppercase then lowercase is a normal word shape
    assert glued_kind("WMT", WORDS) is None


def test_find_glued_lists_every_occurrence_in_text_order() -> None:
    text = "Thedecoder is shownin (Thedecoder), and zzQQ."

    assert find_glued(text, WORDS) == [
        GluedToken("Thedecoder", TWO_WORDS),
        GluedToken("shownin", TWO_WORDS),
        GluedToken("Thedecoder", TWO_WORDS),
        GluedToken("zzQQ", CASE_JOIN),
    ]
