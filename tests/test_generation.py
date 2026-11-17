"""Question generation: reply parsing, every verification check, and the flows (FakeLLM)."""

import json
from pathlib import Path
from typing import Any

import pytest

from adaptrag.evaluation.gate import Evidence, GateQuestion, load_questions
from adaptrag.evaluation.generation import (
    CHECKS_PASSED,
    GENERATOR_SYSTEM,
    INVALID_JSON,
    JUDGE_INVALID,
    JUDGE_NOT_ANSWERABLE,
    JUDGE_NOT_SINGLE_ANSWER,
    LEXICAL_OVERLAP,
    LLM_ERROR,
    MAX_QUOTE_WORDS,
    MENTIONS_LOCATION,
    MIN_QUOTE_WORDS,
    MISSING_FIELDS,
    NEAR_DUPLICATE,
    PROMPT_VERSION,
    QUOTE_LENGTH,
    QUOTE_NOT_FOUND,
    TYPE_MISMATCH,
    Draft,
    JudgeVerdict,
    ReplyError,
    answer_candidates,
    calibrate,
    estimate_usage,
    extend_quote,
    files_with_other_version,
    generate_for_document,
    generator_user,
    is_near_duplicate,
    judge_user,
    lexical_overlap,
    locate_quotes,
    mentions_location,
    page_texts,
    parse_candidate_reply,
    parse_generator_reply,
    parse_judge_reply,
    quote_context,
    quote_length_ok,
    quote_pages,
    quote_words,
    select_kept,
    targets,
)
from adaptrag.evaluation.passages import FACTUAL, MULTI_SECTION, TABLE, DraftSlot, Passage
from adaptrag.evaluation.questions import load_eval_questions
from adaptrag.llm.fake import FakeLLM
from adaptrag.parsing.blocks import BlockType, ParsedBlock, ParsedDocument

# 13 words; on page 1 of the test document and nowhere else.
QUOTE = "the base model was trained for one hundred thousand steps on eight gpus"
PAGE_1 = f"Training details. In our setup {QUOTE} which took twelve hours."
PAGE_2 = "Evaluation uses the standard development split and reports accuracy."
PAGE_3 = "As noted before, the schedule ran for one hundred thousand steps in total."
# Content words: many, optimisation, updates, base, network, receive. Only "base" is in QUOTE.
QUESTION = "How many optimisation updates did the base network receive?"
ANSWER = "one hundred thousand"


def judge_reply(
    answerable: bool = True, single_answer: bool = True, needs_document: bool = True
) -> str:
    return json.dumps(
        {
            "answerable": answerable,
            "single_answer": single_answer,
            "needs_document": needs_document,
        }
    )


JUDGE_OK = judge_reply()


def paragraph(text: str, page: int) -> ParsedBlock:
    return ParsedBlock(text=text, page=page, block_type=BlockType.PARAGRAPH)


def heading(text: str, page: int) -> ParsedBlock:
    return ParsedBlock(
        text=text, page=page, block_type=BlockType.HEADING, heading_path=(text,), level=1
    )


def make_doc(*blocks: ParsedBlock) -> ParsedDocument:
    page_count = max(block.page for block in blocks)
    return ParsedDocument(page_count=page_count, parser_version="test", blocks=blocks)


def two_page_doc() -> ParsedDocument:
    return make_doc(paragraph(PAGE_1, 1), paragraph(PAGE_2, 2))


def three_page_doc() -> ParsedDocument:
    """Like two_page_doc, plus a page that states the answer a second time."""
    return make_doc(paragraph(PAGE_1, 1), paragraph(PAGE_2, 2), paragraph(PAGE_3, 3))


def slot(question_type: str = FACTUAL) -> DraftSlot:
    return DraftSlot(question_type, (Passage(text=PAGE_1, page=1, kind="prose", first_block=0),))


def generator_reply(**changes: Any) -> str:
    data: dict[str, Any] = {
        "question": QUESTION,
        "answer": ANSWER,
        "type": FACTUAL,
        "evidence": [QUOTE],
    }
    data.update(changes)
    return json.dumps(data)


def reason_of(reply: str, expected_type: str = FACTUAL, quote_count: int = 1) -> str:
    with pytest.raises(ReplyError) as info:
        parse_generator_reply(reply, expected_type, quote_count)
    return info.value.reason


def record(question_id: str, question_type: str) -> dict[str, Any]:
    return {"id": question_id, "type": question_type}


# Reply parsing


def test_a_valid_generator_reply_becomes_a_draft() -> None:
    draft = parse_generator_reply(generator_reply(), FACTUAL, 1)

    assert draft == Draft(question=QUESTION, answer=ANSWER, type=FACTUAL, quotes=(QUOTE,))


def test_a_reply_inside_one_code_fence_is_accepted() -> None:
    fenced = "```json\n" + generator_reply() + "\n```"

    assert parse_generator_reply(fenced, FACTUAL, 1).question == QUESTION


def test_a_reply_that_is_not_a_json_object_is_invalid() -> None:
    assert reason_of("Sure! Here is a question.") == INVALID_JSON
    assert reason_of("[1, 2]") == INVALID_JSON
    assert reason_of('{"question": ') == INVALID_JSON


def test_missing_blank_or_wrongly_typed_fields_are_rejected() -> None:
    without_answer = json.loads(generator_reply())
    del without_answer["answer"]

    assert reason_of(json.dumps(without_answer)) == MISSING_FIELDS
    assert reason_of(generator_reply(question="   ")) == MISSING_FIELDS
    assert reason_of(generator_reply(answer=28.4)) == MISSING_FIELDS
    assert reason_of(generator_reply(evidence=QUOTE)) == MISSING_FIELDS  # a string, not a list
    assert reason_of(generator_reply(evidence=[])) == MISSING_FIELDS
    assert reason_of(generator_reply(evidence=["  "])) == MISSING_FIELDS


def test_a_type_other_than_the_one_asked_for_is_rejected() -> None:
    assert reason_of(generator_reply(type=TABLE)) == TYPE_MISMATCH
    assert reason_of(generator_reply(type="essay"), expected_type="essay") == TYPE_MISMATCH


def test_a_multi_section_reply_needs_one_quote_per_passage() -> None:
    two = generator_reply(type=MULTI_SECTION, evidence=[QUOTE, "a second quote"])
    one = generator_reply(type=MULTI_SECTION)

    assert parse_generator_reply(two, MULTI_SECTION, 2).quotes == (QUOTE, "a second quote")
    assert reason_of(one, MULTI_SECTION, 2) == MISSING_FIELDS


def test_a_judge_reply_needs_three_real_booleans() -> None:
    assert parse_judge_reply(judge_reply(single_answer=False)) == JudgeVerdict(True, False, True)

    words_not_booleans = judge_reply().replace("true", '"yes"')
    for bad, reason in (
        (words_not_booleans, MISSING_FIELDS),
        ('{"answerable": true, "single_answer": true}', MISSING_FIELDS),
        ("The question looks fine.", INVALID_JSON),
    ):
        with pytest.raises(ReplyError) as info:
            parse_judge_reply(bad)
        assert info.value.reason == reason


def test_a_location_check_reply_is_one_boolean() -> None:
    assert parse_candidate_reply('{"answers": true}') is True
    assert parse_candidate_reply('{"answers": false}') is False
    with pytest.raises(ReplyError):
        parse_candidate_reply('{"answers": "true"}')


# Checks


def test_check_1_and_2_find_every_page_that_holds_the_quote() -> None:
    sentence = "The Learning   rate\nwas warmed up linearly over four thousand steps."
    doc = make_doc(
        paragraph("Introduction text.", 1),
        paragraph(sentence, 2),
        paragraph("Something else entirely.", 3),
        paragraph(f"We repeat: {sentence}", 4),
    )

    assert quote_pages(page_texts(doc), "learning rate was warmed up linearly") == [2, 4]


def test_a_quote_that_is_not_in_the_document_has_no_pages() -> None:
    pages = page_texts(two_page_doc())

    assert quote_pages(pages, "the base model was trained for two hundred steps") == []
    assert quote_pages(pages, "   ") == []


def test_locate_quotes_gives_an_entry_per_occurrence_or_nothing() -> None:
    pages = page_texts(three_page_doc())
    shared = "one hundred thousand steps"  # on pages 1 and 3

    assert locate_quotes(pages, [shared, "standard development split"]) == [
        Evidence(1, shared),
        Evidence(3, shared),
        Evidence(2, "standard development split"),
    ]
    assert locate_quotes(pages, [shared, "words that appear nowhere"]) is None


def test_a_quote_must_have_8_to_25_words() -> None:
    def quote(count: int) -> str:
        return " ".join(["word"] * count)

    assert not quote_length_ok(quote(MIN_QUOTE_WORDS - 1))
    assert quote_length_ok(quote(8))
    assert quote_length_ok(quote(25))
    assert not quote_length_ok(quote(MAX_QUOTE_WORDS + 1))


def test_quote_words_ignores_pipes_and_divider_cells() -> None:
    assert quote_words("| Base model | 65M | 27.3 |") == 4
    assert quote_words("| --- | :---: | ---: |") == 0
    assert quote_words("a plain sentence of six words") == 6
    # A symbol inside a word is part of the word.
    assert quote_words("state-of-the-art | 28.4") == 2


def test_a_table_row_is_measured_without_its_symbols() -> None:
    cells = [f"c{i}" for i in range(14)]
    row = "| " + " | ".join(cells) + " |"  # 14 words and 15 pipes: 29 tokens

    assert len(row.split()) > MAX_QUOTE_WORDS
    assert quote_words(row) == 14
    assert quote_length_ok(row)


# Quote extension (v2, change 4)


def test_a_short_prose_quote_is_extended_to_its_sentence() -> None:
    sentence = "The base model used eight attention heads in total!"
    doc = make_doc(paragraph(f"First sentence ends here. {sentence} Last one follows.", 1))

    assert extend_quote(doc, "eight attention heads") == sentence


def test_a_short_table_quote_is_extended_to_its_row() -> None:
    table = ParsedBlock(
        text="| name | params | bleu |\n| --- | --- | --- |\n"
        "| Base model | 65M | 27.3 |\n| Big model | 213M | 28.4 |",
        page=1,
        block_type=BlockType.TABLE,
    )

    assert extend_quote(make_doc(table), "213M | 28.4") == "| Big model | 213M | 28.4 |"


def test_a_long_sentence_is_cut_to_25_words_centred_on_the_quote() -> None:
    sentence = " ".join(f"w{i}" for i in range(40)) + "."
    doc = make_doc(paragraph(sentence, 1))

    extended = extend_quote(doc, "w19 w20 w21")

    assert extended == " ".join(f"w{i}" for i in range(8, 33))  # 11 words each side
    assert quote_words(extended) == MAX_QUOTE_WORDS


def test_extension_leaves_long_and_unplaceable_quotes_unchanged() -> None:
    two_sentences = "First part ends here. Second part starts now."
    doc = make_doc(paragraph(two_sentences, 1), paragraph(PAGE_1, 1))

    assert extend_quote(doc, QUOTE) == QUOTE  # already long enough
    assert extend_quote(doc, "ends here. Second part") == "ends here. Second part"  # two sentences
    assert extend_quote(doc, "nowhere to be found") == "nowhere to be found"  # not in the document


def test_a_quote_still_too_short_after_extension_is_rejected_before_the_judge() -> None:
    doc = make_doc(paragraph("Alpha beta gamma delta epsilon zeta. Other things here.", 1))
    llm = FakeLLM(replies=[generator_reply(evidence=["beta gamma delta"]), JUDGE_OK])

    records, stats = generate_for_document(llm, doc, [slot()], "p")

    assert records == []
    assert stats.rejected[QUOTE_LENGTH] == 1
    assert llm.calls == 1


def test_an_extended_quote_is_found_by_check_1_and_stored_instead_of_the_short_one() -> None:
    sentence = "In our setup the base model was trained for one hundred thousand steps."
    doc = make_doc(paragraph(f"Preface sentence. {sentence} Another sentence follows.", 1))
    short = "trained for one hundred thousand"
    llm = FakeLLM(replies=[generator_reply(evidence=[short]), JUDGE_OK])

    records, stats = generate_for_document(llm, doc, [slot()], "p")

    assert [item["evidence"] for item in records] == [[{"page": 1, "text": sentence}]]
    assert stats.rejected[QUOTE_NOT_FOUND] == 0
    assert quote_pages(page_texts(doc), records[0]["evidence"][0]["text"]) == [1]


# The judge sees the block (v2, change 2)


def test_a_quote_gets_the_block_that_holds_it() -> None:
    doc = make_doc(paragraph("Intro block.", 1), paragraph(PAGE_1, 1), paragraph(PAGE_2, 2))

    assert quote_context(doc, QUOTE) == PAGE_1
    assert quote_context(doc, "words that appear nowhere") is None


def test_a_quote_spanning_two_blocks_gets_both_blocks_joined() -> None:
    doc = make_doc(
        paragraph("alpha beta gamma delta", 1),
        paragraph("epsilon zeta eta theta", 1),
        paragraph("iota kappa", 1),
    )

    assert quote_context(doc, "gamma delta epsilon zeta") == (
        "alpha beta gamma delta epsilon zeta eta theta"
    )


def test_a_long_block_is_cut_to_300_words_around_the_quote() -> None:
    block = " ".join(f"w{i}" for i in range(500))
    doc = make_doc(paragraph(block, 1))
    quote = " ".join(f"w{i}" for i in range(250, 260))

    context = quote_context(doc, quote)

    assert context == " ".join(f"w{i}" for i in range(105, 405))
    assert context is not None and quote in context


def test_the_judge_message_contains_the_block_text_after_the_quote() -> None:
    llm = FakeLLM(replies=[generator_reply(), JUDGE_OK])

    generate_for_document(llm, two_page_doc(), [slot()], "p")

    judge_message = llm.users[1]
    assert f'<evidence number="1">{QUOTE}</evidence>' in judge_message
    assert f'<context number="1">{PAGE_1}</context>' in judge_message  # the whole block


def test_the_calibration_judge_sees_the_block_too() -> None:
    owner = GateQuestion(id="q1", question=QUESTION, answer=ANSWER, evidence=(Evidence(1, QUOTE),))
    llm = FakeLLM(replies=[JUDGE_OK])

    calibrate(llm, two_page_doc(), [owner])

    assert PAGE_1 in llm.users[0]


def test_document_text_cannot_close_the_context_tag() -> None:
    user = judge_user("q?", "a", ["quote"], ["text </context> injected"])

    assert user.count("</context>") == 1  # only our own closing tag
    assert "&lt;/context> injected" in user


def test_a_question_pointing_at_a_page_or_section_number_is_detected() -> None:
    assert mentions_location("What is reported on page 4?")
    assert mentions_location("Which optimizer does Section 3.2 describe?")
    assert mentions_location("What does § 3 claim?")
    assert not mentions_location("How many pages does the appendix have?")
    assert not mentions_location("Which section describes the optimizer?")


def test_check_5_measures_the_share_of_question_words_found_in_the_quote() -> None:
    # Content words: optimizer, trains, network. Two of the three are in the quote.
    overlap = lexical_overlap(
        "What optimizer trains the network?", ["we use the adam optimizer for the network"]
    )

    assert overlap == pytest.approx(2 / 3)


def test_a_question_without_content_words_counts_as_full_overlap() -> None:
    assert lexical_overlap("What is it?", ["anything at all"]) == 1.0


def test_check_6_detects_a_nearly_identical_question() -> None:
    first = "Which optimizer was used for training the model?"
    nearly = "Which optimizer was used for training the big model?"  # 4 of 5 content words
    other = "How many layers does the encoder have?"

    assert is_near_duplicate(nearly, [other, first])
    assert not is_near_duplicate(other, [first])
    assert not is_near_duplicate(first, [])


def test_check_3_finds_the_answer_once_per_page_as_whole_words() -> None:
    before = " ".join(f"a{i}" for i in range(20))
    after = " ".join(f"b{i}" for i in range(20))
    long_block = f"{before} outputs of dimension 512, as before {after}"
    doc = make_doc(
        paragraph("The model dimension is 512 here.", 1),  # page already holds evidence
        paragraph(long_block, 2),
        paragraph("A batch of 5120 tokens and 1512 steps.", 3),  # not the whole word 512
        heading("512", 4),  # headings are skipped
        paragraph("Width 512.", 5),
        paragraph("Again 512 wide.", 5),  # second place on the same page
    )

    candidates = answer_candidates(doc, "512", exclude_pages={1})

    assert [candidate.page for candidate in candidates] == [2, 5]
    assert candidates[1].text == "Width 512."
    # 15 words on each side of the answer, all from the same block.
    assert candidates[0].text == " ".join(long_block.split()[8:39])
    assert "512," in candidates[0].text.split()


def test_check_3_stops_at_the_candidate_limit() -> None:
    blocks = [paragraph(f"On page {page} the value is 42 again.", page) for page in range(1, 9)]
    doc = make_doc(*blocks)

    candidates = answer_candidates(doc, "42", exclude_pages=set())

    assert [candidate.page for candidate in candidates] == [1, 2, 3, 4, 5]


def test_check_3_matches_a_multi_word_answer_despite_punctuation() -> None:
    doc = make_doc(paragraph("We use the (Adam optimizer), with warmup.", 1))

    candidates = answer_candidates(doc, "Adam optimizer", exclude_pages=set())

    assert [candidate.text for candidate in candidates] == [
        "We use the (Adam optimizer), with warmup."
    ]
    assert answer_candidates(doc, "Adam with", exclude_pages=set()) == []


# Flows with FakeLLM


def test_an_accepted_question_is_stored_with_everything_the_stage_file_asks_for() -> None:
    llm = FakeLLM(replies=[generator_reply(), JUDGE_OK])

    records, stats = generate_for_document(llm, two_page_doc(), [slot()], "paper_x_r0")

    assert records == [
        {
            "id": "paper_x_r0_d01",
            "type": FACTUAL,
            "question": QUESTION,
            "answer": ANSWER,
            # The page comes from our search of the document, not from the LLM.
            "evidence": [{"page": 1, "text": QUOTE}],
            "verified": True,
            "verification": "machine",
            "checks_passed": list(CHECKS_PASSED),
            "generator_model": "fake-llm",
            "prompt_version": PROMPT_VERSION,
            "lexical_overlap": 0.167,
            "answerable_without_document": False,  # the judge said it needs the document
            "round": 0,
        }
    ]
    assert (stats.drafted, stats.accepted, sum(stats.rejected.values())) == (1, 1, 0)
    assert llm.calls == 2  # generator and judge; no other page holds the answer


def test_a_quote_missing_from_the_document_is_rejected_before_the_judge_is_paid() -> None:
    invented = "the base model was trained for nine hundred steps on two gpus"
    llm = FakeLLM(replies=[generator_reply(evidence=[invented]), JUDGE_OK])

    records, stats = generate_for_document(llm, two_page_doc(), [slot()], "p")

    assert records == []
    assert stats.rejected[QUOTE_NOT_FOUND] == 1
    assert llm.calls == 1


def test_invalid_generator_json_is_rejected_and_counted() -> None:
    llm = FakeLLM(replies=["I cannot write a question for this passage."])

    records, stats = generate_for_document(llm, two_page_doc(), [slot()], "p")

    assert records == []
    assert (stats.drafted, stats.rejected[INVALID_JSON]) == (1, 1)


def test_a_question_that_copies_its_evidence_is_rejected_for_overlap() -> None:
    copied = "Was the base model trained for one hundred thousand steps on eight gpus?"
    llm = FakeLLM(replies=[generator_reply(question=copied), JUDGE_OK])

    records, stats = generate_for_document(llm, two_page_doc(), [slot()], "p")

    assert records == []
    assert stats.rejected[LEXICAL_OVERLAP] == 1
    assert stats.overlaps == [1.0]
    assert llm.calls == 1


def test_each_judge_no_rejects_with_its_own_reason() -> None:
    cases = (
        (judge_reply(answerable=False), JUDGE_NOT_ANSWERABLE),
        (judge_reply(single_answer=False), JUDGE_NOT_SINGLE_ANSWER),
    )
    for verdict, reason in cases:
        llm = FakeLLM(replies=[generator_reply(), verdict])

        records, stats = generate_for_document(llm, two_page_doc(), [slot()], "p")

        assert records == []
        assert dict(stats.rejected) == {reason: 1}


def test_needs_document_no_longer_rejects_but_is_stored_as_a_label() -> None:
    for needs_document, label in ((False, True), (True, False)):
        llm = FakeLLM(replies=[generator_reply(), judge_reply(needs_document=needs_document)])

        records, stats = generate_for_document(llm, two_page_doc(), [slot()], "p")

        assert stats.accepted == 1
        assert sum(stats.rejected.values()) == 0
        assert records[0]["answerable_without_document"] is label


def test_an_unusable_judge_reply_rejects_the_draft() -> None:
    llm = FakeLLM(replies=[generator_reply(), "It seems fine to me."])

    records, stats = generate_for_document(llm, two_page_doc(), [slot()], "p")

    assert records == []
    assert stats.rejected[JUDGE_INVALID] == 1


def test_the_same_question_twice_is_kept_once() -> None:
    llm = FakeLLM(replies=[generator_reply(), JUDGE_OK, generator_reply()])

    records, stats = generate_for_document(llm, two_page_doc(), [slot(), slot()], "p")

    assert [item["id"] for item in records] == ["p_d01"]
    assert stats.rejected[NEAR_DUPLICATE] == 1
    assert llm.calls == 3  # the duplicate never reaches the judge


def test_questions_kept_in_an_earlier_round_count_as_duplicates_too() -> None:
    llm = FakeLLM(replies=[generator_reply(), JUDGE_OK])

    records, stats = generate_for_document(
        llm, two_page_doc(), [slot()], "p", round_number=1, earlier_questions=[QUESTION]
    )

    assert records == []
    assert stats.rejected[NEAR_DUPLICATE] == 1


def test_a_second_location_the_judge_confirms_becomes_evidence() -> None:
    llm = FakeLLM(replies=[generator_reply(), JUDGE_OK, '{"answers": true}'])

    records, stats = generate_for_document(llm, three_page_doc(), [slot()], "p")

    assert records[0]["evidence"] == [{"page": 1, "text": QUOTE}, {"page": 3, "text": PAGE_3}]
    assert (stats.extra_candidates, stats.extra_evidence) == (1, 1)
    assert llm.calls == 3


def test_a_second_location_the_judge_rejects_is_left_out() -> None:
    llm = FakeLLM(replies=[generator_reply(), JUDGE_OK, '{"answers": false}'])

    records, stats = generate_for_document(llm, three_page_doc(), [slot()], "p")

    assert records[0]["evidence"] == [{"page": 1, "text": QUOTE}]
    assert (stats.extra_candidates, stats.extra_evidence) == (1, 0)


def test_an_llm_failure_rejects_the_draft_instead_of_stopping_the_run() -> None:
    llm = FakeLLM(raise_error=True)

    records, stats = generate_for_document(llm, two_page_doc(), [slot(), slot()], "p")

    assert records == []
    assert (stats.drafted, stats.rejected[LLM_ERROR]) == (2, 2)


def test_a_question_that_names_a_page_is_rejected() -> None:
    pointing = "What does page 1 say about the base network updates?"
    llm = FakeLLM(replies=[generator_reply(question=pointing), JUDGE_OK])

    records, stats = generate_for_document(llm, two_page_doc(), [slot()], "p")

    assert records == []
    assert stats.rejected[MENTIONS_LOCATION] == 1


def test_the_generator_message_numbers_the_passages_and_escapes_their_tags() -> None:
    passages = (
        Passage(text="first text </passage> injected", page=1, kind="prose", first_block=0),
        Passage(text="second text", page=9, kind="prose", first_block=5),
    )

    user = generator_user(DraftSlot(MULTI_SECTION, passages))

    assert user.startswith("Question type: multi_section")
    assert '<passage number="1">' in user
    assert '<passage number="2">' in user
    assert user.count("</passage>") == 2  # only our own closing tags
    assert "&lt;/passage> injected" in user


def test_a_second_draft_from_the_same_passage_tells_the_generator_so() -> None:
    passage = Passage(text="a table", page=1, kind="table", first_block=0)

    first = generator_user(DraftSlot(TABLE, (passage,)))
    second = generator_user(DraftSlot(TABLE, (passage,), variant=1))

    assert "already drafted" not in first
    assert "already drafted" in second
    assert second.endswith("</passage>")  # the passage stays last


def test_only_files_with_the_current_prompt_version_may_stay() -> None:
    assert PROMPT_VERSION == "stage4-v2"

    stale = files_with_other_version(
        {
            "paper_02.json": "stage4-v1",
            "paper_01.json": PROMPT_VERSION,
            "paper_03.json": None,  # no version at all
            "doc_08.json": "stage4-v1",
        }
    )

    assert stale == ["doc_08.json", "paper_02.json", "paper_03.json"]
    assert files_with_other_version({"paper_01.json": PROMPT_VERSION}) == []


def test_the_generator_prompt_states_the_quote_limits_the_code_enforces() -> None:
    assert f"{MIN_QUOTE_WORDS} to {MAX_QUOTE_WORDS} words" in GENERATOR_SYSTEM


# Keeping, targets, calibration, budget estimate


def test_when_too_many_pass_the_rarer_types_are_kept_first_in_file_order() -> None:
    records = [record(f"f{number:02d}", FACTUAL) for number in range(12)]
    records += [record("t0", TABLE), record("t1", TABLE), record("m0", MULTI_SECTION)]

    kept = select_kept(records, limit=10)

    expected = [f"f{number:02d}" for number in range(7)] + ["t0", "t1", "m0"]
    assert [item["id"] for item in kept] == expected


def test_everything_is_kept_under_the_limit_except_unknown_types() -> None:
    records = [record("f0", FACTUAL), record("x0", "essay"), record("t0", TABLE)]

    assert [item["id"] for item in select_kept(records, limit=10)] == ["f0", "t0"]


def test_targets_report_what_is_met_and_what_is_missed() -> None:
    report = targets({FACTUAL: 50, TABLE: 15, MULTI_SECTION: 9})

    assert [(target.have, target.need, target.met) for target in report] == [
        (74, 70, True),
        (15, 15, True),
        (9, 10, False),
    ]


def test_calibration_counts_what_the_verifier_accepts_and_why_it_rejects() -> None:
    def question(question_id: str, evidence_text: str) -> GateQuestion:
        return GateQuestion(
            id=question_id, question=QUESTION, answer=ANSWER, evidence=(Evidence(1, evidence_text),)
        )

    questions = [
        question("q1", QUOTE),
        question("q2", "this evidence is not in the document"),  # check 1: no judge call
        question("q3", QUOTE),
    ]
    llm = FakeLLM(replies=[JUDGE_OK, judge_reply(answerable=False)])

    result = calibrate(llm, two_page_doc(), questions)

    assert (result.accepted, result.total) == (1, 3)
    assert dict(result.rejected) == {QUOTE_NOT_FOUND: 1, JUDGE_NOT_ANSWERABLE: 1}
    assert llm.calls == 2


def test_the_budget_estimate_counts_seven_calls_per_draft() -> None:
    # (10 drafts * 7 calls + 5 calibration calls) = 75 calls, at 3000 input and 500 output.
    assert estimate_usage(10, 5, 500) == (225_000, 37_500)


def test_a_generated_file_is_read_by_the_stage_2_loader(tmp_path: Path) -> None:
    llm = FakeLLM(replies=[generator_reply(), JUDGE_OK])
    records, _stats = generate_for_document(llm, two_page_doc(), [slot()], "paper_x_r0")
    path = tmp_path / "paper_x.json"
    path.write_text(json.dumps({"paper": "paper_x.pdf", "questions": records}), encoding="utf-8")

    scorable, skipped = load_questions(path)

    assert skipped == []
    assert [(item.id, item.evidence) for item in scorable] == [
        ("paper_x_r0_d01", (Evidence(1, QUOTE),))
    ]
    assert [item.type for item in load_eval_questions(path, "paper_x")] == [FACTUAL]
