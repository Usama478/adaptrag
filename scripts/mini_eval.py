"""Stage 4 mini evaluation: every strategy and the router on the dev documents (note 0025).

No arguments, dev documents only, no LLM calls (free). It:
1. makes sure each of the 9 dev documents is ingested (the router's choice is read from
   the database) and builds any missing chunk sets and embeddings for fixed_size,
   recursive and section_aware, plus recursive and section_aware with a table token
   budget of 512 (real bge-small, exact search);
2. retrieves the top 10 chunks for every question under every variant and finds the rank
   of the first relevant chunk (evaluation/relevance.py);
3. prints the results table, the breakdowns and the outcome of each rule fixed before the
   run (evaluation/decisions.py), and writes eval/results/stage4/results.json and
   summary.md. The same inputs give the same output: nothing in them depends on the clock.

Needs the question files from scripts/generate_questions.py.

Usage: python scripts/mini_eval.py
"""

import argparse
import json
import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from adaptrag.chunking.splitter import word_count
from adaptrag.config import Settings, get_settings
from adaptrag.db import get_engine
from adaptrag.embedding.base import Embedder
from adaptrag.embedding.sentence_transformer import get_embedder, get_vocabulary
from adaptrag.evaluation.analysis import (
    HEADLINE_K,
    MetricRow,
    best_single_strategy,
    best_strategy_per_document,
    compose,
    metric_row,
    paired_outcome,
    pooled,
)
from adaptrag.evaluation.corpus import TEST_DIR, CorpusEntry, load_corpus
from adaptrag.evaluation.decisions import (
    VERY_SHORT_MIN_DOCUMENTS,
    VERY_SHORT_WORDS,
    BudgetComparison,
    DocumentOutcome,
    propose_threshold,
    router_decision,
    table_budget_decision,
    tied_pairs,
    very_short_decision,
    very_short_documents,
)
from adaptrag.evaluation.gate import QuestionFileError
from adaptrag.evaluation.glued import CASE_JOIN, TWO_WORDS, find_glued, vocabulary_words
from adaptrag.evaluation.metrics import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    Rank,
    hits_at_k,
    mrr,
    recall_at_k,
)
from adaptrag.evaluation.passages import FACTUAL, MULTI_SECTION, TABLE
from adaptrag.evaluation.questions import EvalQuestion, load_eval_questions
from adaptrag.evaluation.relevance import MISSING, SPLIT, evidence_status, first_relevant_rank
from adaptrag.ingest import embed_chunk_set, run_ingest_job, save_chunk_set
from adaptrag.models import Chunk, Document, Job
from adaptrag.parsing.blocks import ParsedDocument
from adaptrag.parsing.pdf import PARSER_VERSION, parse_pdf
from adaptrag.retrieval.results import ScoredChunk
from adaptrag.routes.documents import find_by_hash
from adaptrag.routing.profile import DocumentProfile, profile_document
from adaptrag.storage import compute_sha256, save_upload
from adaptrag.strategies.base import Strategy
from adaptrag.strategies.recursive import RecursiveStrategy
from adaptrag.strategies.registry import build_strategy
from adaptrag.strategies.section_aware import (
    SectionAwareStrategy,
    heading_prefixes,
    strip_heading_prefix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "eval" / "corpus.json"
GENERATED_DIR = REPO_ROOT / "eval" / "questions" / "generated"
OWNER_SET = REPO_ROOT / "eval" / "questions" / "paper_01.json"
OWNER_DOCUMENT = "paper_01"
RESULTS_DIR = REPO_ROOT / "eval" / "results" / "stage4"

TOP_K = 10
TABLE_TOKEN_BUDGET = 512
# The three strategies of the benchmark, in the order that wins ties (control first).
BASE = ("fixed_size", "recursive", "section_aware")
# (strategy without the budget, the same strategy with it)
BUDGET_PAIRS = (
    ("recursive", "recursive+tb512"),
    ("section_aware", "section_aware+tb512"),
)
VARIANTS = (*BASE, *(with_budget for _, with_budget in BUDGET_PAIRS))
QUESTION_TYPES = (FACTUAL, TABLE, MULTI_SECTION)

BOUNDARY_CHUNKS = 15
BOUNDARY_CHARS = 60
FAILURE_TOP = 3
FAILURE_CHARS = 200
GLUED_EXAMPLES = 8
SCORE_BIN = 0.05
RECURSIVE_BOUNDARY_DOCUMENT = "doc_08"

# Fixed questions that no dev document answers: the low end of the top-1 score scale.
OFF_TOPIC_QUESTIONS = (
    "What is the capital of France?",
    "How do I bake sourdough bread at home?",
    "Who won the football World Cup in 1998?",
    "What is the boiling point of water at sea level?",
    "How many strings does a violin have?",
    "Which planet is closest to the sun?",
    "What is a good recipe for tomato soup?",
    "How long does it take to walk the Camino de Santiago?",
    "Who painted the Mona Lisa?",
    "What are the rules of cricket?",
    "How do I change a flat bicycle tyre?",
    "What is the tallest mountain in Africa?",
    "When was the Eiffel Tower built?",
    "How do bees make honey?",
    "What is the best way to remove a red wine stain?",
    "Which river flows through Cairo?",
    "How many players are on a basketball team?",
    "What causes the northern lights?",
    "How do I train a puppy to sit?",
    "What is the currency of Japan?",
)


@dataclass
class StoredChunk:
    index: int
    text: str
    page_start: int
    page_end: int


@dataclass
class DevDocument:
    """One dev document with everything the evaluation needs to know about it."""

    entry: CorpusEntry
    parsed: ParsedDocument
    profile: DocumentProfile
    database_id: int
    routed: str
    prefixes: frozenset[str]
    chunk_set_ids: dict[str, int] = field(default_factory=dict)
    chunks: dict[str, list[StoredChunk]] = field(default_factory=dict)


@dataclass
class Run:
    """One question under one variant: what was retrieved and where the evidence ranked."""

    question: EvalQuestion
    retrieved: list[ScoredChunk]
    rank: Rank


# Small formatting helpers


def number(value: float) -> str:
    return f"{value:.3f}"


def interval(pair: tuple[float, float]) -> str:
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def pages(start: int, end: int) -> str:
    return f"p{start}" if start == end else f"p{start}-{end}"


def cell(text: str, limit: int) -> str:
    """Text on one line, cut to `limit` characters, safe inside a Markdown table."""
    return " ".join(text.split())[:limit].replace("|", "\\|")


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """A Markdown table as lines (also readable in the terminal)."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def is_table_chunk(text: str) -> bool:
    return any(line.startswith("|") for line in text.split("\n"))


# Setup: documents, chunk sets, questions


def dev_entries() -> list[CorpusEntry]:
    """The dev documents of the manifest. A test-set path can never get through."""
    entries = [entry for entry in load_corpus(CORPUS) if entry.split == "dev"]
    for entry in entries:
        if entry.path.startswith(f"{TEST_DIR}/"):
            raise SystemExit(f"{entry.id}: test documents are not allowed here.")
        if not (REPO_ROOT / entry.path).exists():
            raise SystemExit(f"{entry.id}: {entry.path} is missing.")
    return entries


def load_question_sets(
    entries: list[CorpusEntry],
) -> tuple[dict[str, list[EvalQuestion]], list[EvalQuestion]]:
    """The generated questions per document, and the owner-verified paper_01 set."""
    generated: dict[str, list[EvalQuestion]] = {}
    try:
        for entry in entries:
            path = GENERATED_DIR / f"{entry.id}.json"
            if not path.exists():
                raise SystemExit(
                    f"{path.name} is missing. Run: python scripts/generate_questions.py"
                )
            generated[entry.id] = load_eval_questions(path, entry.id)
        owner = load_eval_questions(OWNER_SET, OWNER_DOCUMENT)
    except QuestionFileError as exc:
        raise SystemExit(f"Cannot load questions: {exc}") from exc
    if OWNER_DOCUMENT not in generated:
        raise SystemExit(f"{OWNER_DOCUMENT} is not a dev document in {CORPUS.name}.")
    return generated, owner


def ensure_ingested(
    engine: Engine, settings: Settings, embedder: Embedder, entry: CorpusEntry
) -> tuple[int, str]:
    """(database id, routed strategy) of a dev document, ingesting it first if needed.

    Uses the same run_ingest_job() as the API, so the route is the router's own decision.
    """
    data = (REPO_ROOT / entry.path).read_bytes()
    file_hash = compute_sha256(data)
    if entry.sha256 is not None and file_hash != entry.sha256:
        raise SystemExit(f"{entry.id}: the file's SHA-256 differs from {CORPUS.name}.")
    upload_dir = settings.data_dir / "uploads"

    with Session(engine) as session:
        document = find_by_hash(session, file_hash)
        if document is not None and document.status == "ready" and document.strategy in BASE:
            return document.id, document.strategy
        if document is None:
            document = Document(
                file_hash=file_hash, filename=Path(entry.path).name, size_bytes=len(data)
            )
            session.add(document)
            session.flush()
        job = Job(document_id=document.id)
        session.add(job)
        session.commit()
        job_id = job.id

    print(f"{entry.id}: not ready in the database, ingesting it now")
    save_upload(data, file_hash, upload_dir)
    run_ingest_job(engine, upload_dir, job_id, embedder)

    with Session(engine) as session:
        document = find_by_hash(session, file_hash)
        if document is None or document.status != "ready" or document.strategy not in BASE:
            raise SystemExit(f"{entry.id}: ingestion failed; see the job's error in the database.")
        return document.id, document.strategy


def prepare_document(
    engine: Engine, settings: Settings, embedder: Embedder, entry: CorpusEntry
) -> DevDocument:
    database_id, routed = ensure_ingested(engine, settings, embedder, entry)
    parsed = parse_pdf(REPO_ROOT / entry.path)
    return DevDocument(
        entry=entry,
        parsed=parsed,
        profile=profile_document(parsed),
        database_id=database_id,
        routed=routed,
        prefixes=heading_prefixes(parsed),
    )


def make_strategies(embedder: Embedder) -> dict[str, Strategy]:
    """Variant label -> strategy. The three base ones come from the registry."""
    strategies: dict[str, Strategy] = {name: build_strategy(name, embedder) for name in BASE}
    strategies["recursive+tb512"] = RecursiveStrategy(
        embedder, table_token_budget=TABLE_TOKEN_BUDGET
    )
    strategies["section_aware+tb512"] = SectionAwareStrategy(
        embedder, table_token_budget=TABLE_TOKEN_BUDGET
    )
    return strategies


def build_chunk_sets(
    session: Session, document: DevDocument, strategies: dict[str, Strategy], embedder: Embedder
) -> None:
    """Create whatever is missing for each variant, then read the stored chunks back."""
    for label, strategy in strategies.items():
        chunk_set_id, _created = save_chunk_set(
            session,
            document.database_id,
            strategy.name,
            strategy.params,
            document.parsed.parser_version,
            strategy.chunk(document.parsed),
        )
        embed_chunk_set(session, document.database_id, chunk_set_id, embedder)
        session.commit()
        rows = session.execute(
            select(Chunk.chunk_index, Chunk.text, Chunk.page_start, Chunk.page_end)
            .where(Chunk.chunk_set_id == chunk_set_id)
            .order_by(Chunk.chunk_index)
        ).all()
        document.chunk_set_ids[label] = chunk_set_id
        document.chunks[label] = [
            StoredChunk(index, text, page_start, page_end)
            for index, text, page_start, page_end in rows
        ]


def body(strategy: Strategy, document: DevDocument, text: str) -> str:
    """The chunk text the relevance rule sees: without the section-aware heading prefix."""
    if strategy.name == SectionAwareStrategy.name:
        return strip_heading_prefix(text, document.prefixes)
    return text


def retrieve_runs(
    session: Session,
    document: DevDocument,
    label: str,
    strategy: Strategy,
    questions: list[EvalQuestion],
) -> list[Run]:
    runs: list[Run] = []
    for question in questions:
        retrieved = strategy.retrieve(
            session, document.chunk_set_ids[label], question.question, TOP_K
        )
        bodies = [body(strategy, document, chunk.text) for chunk in retrieved]
        rank = first_relevant_rank(bodies, question.evidence_texts)
        runs.append(Run(question=question, retrieved=retrieved, rank=rank))
    return runs


# Report helpers. main() builds the report as a list of Markdown lines.


def row_cells(row: MetricRow, with_intervals: bool = True) -> list[str]:
    no_interval = "n/a (1 document)"
    return [
        row.name,
        number(row.recall_at_1),
        number(row.recall_at_5),
        number(row.recall_at_10),
        number(row.mrr),
        interval(row.recall_at_5_interval) if with_intervals else no_interval,
        interval(row.mrr_interval) if with_intervals else no_interval,
        str(row.questions),
    ]


def row_json(row: MetricRow) -> dict[str, Any]:
    return {
        "name": row.name,
        "questions": row.questions,
        "recall_at_1": round(row.recall_at_1, 4),
        "recall_at_5": round(row.recall_at_5, 4),
        "recall_at_10": round(row.recall_at_10, 4),
        "mrr": round(row.mrr, 4),
        "recall_at_5_interval": [round(value, 4) for value in row.recall_at_5_interval],
        "mrr_interval": [round(value, 4) for value in row.mrr_interval],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 mini evaluation (dev documents).")
    parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")
    settings = get_settings()
    engine = get_engine(settings.database_url)

    entries = dev_entries()
    # Questions first: a missing file should stop the run before the model loads.
    generated, owner_questions = load_question_sets(entries)

    embedder = get_embedder(
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_query_instruction,
    )
    words = vocabulary_words(
        get_vocabulary(
            settings.embedding_model,
            settings.embedding_device,
            settings.embedding_query_instruction,
        )
    )
    strategies = make_strategies(embedder)

    documents = [prepare_document(engine, settings, embedder, entry) for entry in entries]
    by_id = {document.entry.id: document for document in documents}
    routed = {document.entry.id: document.routed for document in documents}

    # runs[variant][document id] = one Run per generated question, in file order.
    runs: dict[str, dict[str, list[Run]]] = {label: {} for label in VARIANTS}
    owner_runs: dict[str, list[Run]] = {}
    off_topic_scores: list[float] = []
    with Session(engine) as session:
        for document in documents:
            build_chunk_sets(session, document, strategies, embedder)
            for label in VARIANTS:
                runs[label][document.entry.id] = retrieve_runs(
                    session, document, label, strategies[label], generated[document.entry.id]
                )
            routed_strategy = strategies[document.routed]
            for question in OFF_TOPIC_QUESTIONS:
                top = routed_strategy.retrieve(
                    session, document.chunk_set_ids[document.routed], question, 1
                )
                if top:
                    off_topic_scores.append(top[0].score)
            print(f"{document.entry.id}: retrieved ({len(generated[document.entry.id])} questions)")
        owner_document = by_id[OWNER_DOCUMENT]
        for label in BASE:
            owner_runs[label] = retrieve_runs(
                session, owner_document, label, strategies[label], owner_questions
            )

    ranks: dict[str, dict[str, list[Rank]]] = {
        label: {name: [run.rank for run in doc_runs] for name, doc_runs in runs[label].items()}
        for label in VARIANTS
    }

    # 1. The results table
    best_single = best_single_strategy(ranks, BASE)
    best_per_document = best_strategy_per_document(ranks, BASE)
    rows = [metric_row(name, ranks[name]) for name in BASE]
    rows.append(metric_row("router (rule v1)", compose(ranks, routed)))
    rows.append(metric_row(f"best single ({best_single})", ranks[best_single]))
    rows.append(metric_row("oracle (upper bound)", compose(ranks, best_per_document)))
    budget_rows = [metric_row(with_budget, ranks[with_budget]) for _, with_budget in BUDGET_PAIRS]
    owner_rows = []
    for label in BASE:
        marker = " (routed)" if label == owner_document.routed else ""
        owner_ranks = {OWNER_DOCUMENT: [run.rank for run in owner_runs[label]]}
        owner_rows.append(metric_row(f"owner paper_01: {label}{marker}", owner_ranks, resamples=1))

    headers = ["row", "R@1", "R@5", "R@10", "MRR", "95% R@5", "95% MRR", "N"]
    report: list[str] = ["# Stage 4 mini evaluation (dev documents)", ""]
    report.append(
        f"Embedding model {embedder.model_name}, parser {PARSER_VERSION}, top {TOP_K}, exact "
        f"search. Intervals: cluster bootstrap over documents, {BOOTSTRAP_RESAMPLES} resamples, "
        f"seed {BOOTSTRAP_SEED}."
    )
    report.extend(["", "## Results (machine-verified questions)", ""])
    report.extend(table(headers, [row_cells(row) for row in rows]))
    report.extend(["", "Best single strategy and oracle are chosen on these same dev numbers."])
    report.extend(["", "## Table token budget variants", ""])
    report.extend(table(headers, [row_cells(row) for row in budget_rows]))
    report.extend(["", "## Owner-verified paper_01 set (10 questions, one document)", ""])
    report.extend(table(headers, [row_cells(row, with_intervals=False) for row in owner_rows]))

    # 2. Breakdown per question type
    def router_runs(document_id: str) -> list[Run]:
        return runs[routed[document_id]][document_id]

    def type_ranks(label: str | None, question_type: str) -> list[Rank]:
        """Ranks of one question type under a variant, or under the router for None."""
        found: list[Rank] = []
        for document in documents:
            name = document.entry.id
            doc_runs = router_runs(name) if label is None else runs[label][name]
            found.extend(run.rank for run in doc_runs if run.question.type == question_type)
        return found

    type_rows: list[list[str]] = []
    type_json: dict[str, dict[str, Any]] = {}
    for question_type in QUESTION_TYPES:
        type_json[question_type] = {}
        for label in (*BASE, None):
            name = "router" if label is None else label
            found = type_ranks(label, question_type)
            hits = hits_at_k(found, HEADLINE_K)
            type_rows.append(
                [question_type, name, str(len(found)), f"{hits}/{len(found)}", number(mrr(found))]
            )
            type_json[question_type][name] = {
                "questions": len(found),
                "hits_at_5": hits,
                "mrr": round(mrr(found), 4),
            }
    report.extend(["", "## Per question type", ""])
    report.extend(table(["type", "row", "N", "hits@5", "MRR"], type_rows))

    # 3. Breakdown per document, and the router rule
    document_rows: list[list[str]] = []
    outcomes: list[DocumentOutcome] = []
    for document in documents:
        name = document.entry.id
        hits = {label: hits_at_k(ranks[label][name], HEADLINE_K) for label in BASE}
        outcomes.append(
            DocumentOutcome(document_id=name, routed=document.routed, hits_by_strategy=hits)
        )
        total = len(ranks[BASE[0]][name])
        document_rows.append(
            [
                name,
                document.entry.category,
                str(total),
                *(f"{hits[label]}/{total} ({number(mrr(ranks[label][name]))})" for label in BASE),
                document.routed,
                best_per_document[name],
            ]
        )
    report.extend(["", "## Per document: hits@5 (MRR), routed and best strategy", ""])
    report.extend(table(["document", "category", "N", *BASE, "routed", "best"], document_rows))

    profiles = {document.entry.id: document.profile.to_dict() for document in documents}
    router = router_decision(outcomes, profiles)
    report.extend(["", "## Router rule v1", ""])
    report.append(
        "Rule fixed before the run: propose a rule v2 only if the routed strategy loses on at "
        "least 3 of the documents (another strategy has more hits@5) AND one profile threshold "
        "separates those documents."
    )
    losing = ", ".join(router.losing_documents) or "none"
    report.append(f"Routed strategy loses on {len(router.losing_documents)} documents: {losing}.")
    report.append(f"Outcome: **{router.verdict}** (the router is not changed in this stage).")

    # 4. Paired wins and losses, and ties
    pair_rows: list[list[str]] = []
    pair_json: list[dict[str, Any]] = []
    for position, first in enumerate(BASE):
        for second in BASE[position + 1 :]:
            outcome = paired_outcome(pooled(ranks[first]), pooled(ranks[second]))
            pair_rows.append(
                [
                    f"{first} vs {second}",
                    str(outcome.first_only),
                    str(outcome.second_only),
                    str(outcome.both),
                    str(outcome.neither),
                ]
            )
            pair_json.append(
                {
                    "first": first,
                    "second": second,
                    "first_only": outcome.first_only,
                    "second_only": outcome.second_only,
                    "both": outcome.both,
                    "neither": outcome.neither,
                }
            )
    report.extend(["", f"## Paired outcomes per question (top {HEADLINE_K})", ""])
    report.extend(table(["pair", "only first", "only second", "both", "neither"], pair_rows))

    def only_found_by(first: str, second: str) -> list[str]:
        """Ids of the questions `first` answers in the top 5 and `second` does not."""
        found: list[str] = []
        for document in documents:
            name = document.entry.id
            for first_run, second_run in zip(runs[first][name], runs[second][name], strict=True):
                first_hit = first_run.rank is not None and first_run.rank <= HEADLINE_K
                second_hit = second_run.rank is not None and second_run.rank <= HEADLINE_K
                if first_hit and not second_hit:
                    found.append(first_run.question.id)
        return found

    recall_by_strategy = {row.name: row.recall_at_5 for row in rows[: len(BASE)]}
    ties = tied_pairs(recall_by_strategy)
    report.extend(["", "## Ties", ""])
    if not ties:
        report.append("No two strategies are within 0.01 on Recall@5.")
    for first, second in ties:
        first_ids = ", ".join(only_found_by(first, second)) or "none"
        second_ids = ", ".join(only_found_by(second, first)) or "none"
        report.append(
            f"TIE: {first} ({number(recall_by_strategy[first])}) and {second} "
            f"({number(recall_by_strategy[second])}) are within 0.01 on Recall@5. "
            f"Found only by {first}: {first_ids}. Found only by {second}: {second_ids}."
        )

    # 5. Evidence split across chunks
    split_rows: list[list[str]] = []
    split_json: dict[str, dict[str, int]] = {}
    for label in VARIANTS:
        counts: Counter[str] = Counter()
        for document in documents:
            bodies = [
                body(strategies[label], document, chunk.text) for chunk in document.chunks[label]
            ]
            for question in generated[document.entry.id]:
                counts[evidence_status(bodies, question.evidence_texts)] += 1
        split_rows.append([label, str(counts[SPLIT]), str(counts[MISSING])])
        split_json[label] = {"split": counts[SPLIT], "missing": counts[MISSING]}
    report.extend(["", "## Evidence split across chunk boundaries", ""])
    report.append(
        "split = no chunk of the set holds an evidence entry whole, but one holds at least "
        "half of it. missing = no chunk of the set is relevant at all."
    )
    report.append("")
    report.extend(table(["variant", "questions split", "questions missing"], split_rows))

    # 6. Glued words (measured only; no parser change in this stage)
    glued_rows: list[list[str]] = []
    glued_json: dict[str, dict[str, Any]] = {}
    for document in documents:
        text = " ".join(block.text for block in document.parsed.blocks)
        found_tokens = find_glued(text, words)
        kinds = Counter(token.kind for token in found_tokens)
        common = Counter(token.token for token in found_tokens)
        by_frequency = sorted(common.items(), key=lambda item: (-item[1], item[0]))
        examples = [token for token, _ in by_frequency]
        glued_rows.append(
            [
                document.entry.id,
                str(len(found_tokens)),
                str(len(common)),
                str(kinds[TWO_WORDS]),
                str(kinds[CASE_JOIN]),
                cell(", ".join(examples[:GLUED_EXAMPLES]), 200),
            ]
        )
        glued_json[document.entry.id] = {
            "candidates": len(found_tokens),
            "distinct": len(common),
            "two_words": kinds[TWO_WORDS],
            "case_join": kinds[CASE_JOIN],
            "examples": examples[:GLUED_EXAMPLES],
        }

    def with_glued_evidence(questions: list[EvalQuestion]) -> int:
        return sum(
            1
            for question in questions
            if any(find_glued(text, words) for text in question.evidence_texts)
        )

    all_generated = [
        question for document in documents for question in generated[document.entry.id]
    ]
    glued_generated = with_glued_evidence(all_generated)
    glued_owner = with_glued_evidence(owner_questions)
    report.extend(["", "## Glued-word candidates (general detector)", ""])
    report.append(
        "A candidate is a run of letters that is not in the embedding model's vocabulary and "
        "splits into two vocabulary words (two words), or has a lowercase letter followed by an "
        "uppercase one (case join). Candidates include real compounds and product names."
    )
    report.append("")
    report.extend(
        table(
            ["document", "candidates", "distinct", "two words", "case join", "most frequent"],
            glued_rows,
        )
    )
    report.append("")
    report.append(
        f"Questions whose evidence contains a candidate: {glued_generated} of "
        f"{len(all_generated)} machine-verified, {glued_owner} of {len(owner_questions)} "
        "owner-verified."
    )

    # 7. Table token budget
    def over_limit(label: str) -> tuple[int, int]:
        """(table chunks over the budget, table chunks) of a variant, all documents."""
        over = 0
        tables = 0
        for document in documents:
            for chunk in document.chunks[label]:
                if is_table_chunk(chunk.text):
                    tables += 1
                    if embedder.count_tokens(chunk.text) > TABLE_TOKEN_BUDGET:
                        over += 1
        return over, tables

    comparisons: list[BudgetComparison] = []
    budget_table: list[list[str]] = []
    for without, with_budget in BUDGET_PAIRS:
        over_without, tables_without = over_limit(without)
        over_with, tables_with = over_limit(with_budget)
        table_ranks_without = type_ranks(without, TABLE)
        table_ranks_with = type_ranks(with_budget, TABLE)
        comparison = BudgetComparison(
            strategy=without,
            over_limit_without=over_without,
            over_limit_with=over_with,
            table_hits_without=hits_at_k(table_ranks_without, HEADLINE_K),
            table_hits_with=hits_at_k(table_ranks_with, HEADLINE_K),
        )
        comparisons.append(comparison)
        budget_table.append(
            [
                without,
                f"{over_without}/{tables_without}",
                f"{over_with}/{tables_with}",
                f"{comparison.table_hits_without}/{len(table_ranks_without)}",
                f"{comparison.table_hits_with}/{len(table_ranks_with)}",
            ]
        )
    budget_verdict = table_budget_decision(comparisons)
    report.extend(["", f"## Table token budget ({TABLE_TOKEN_BUDGET} tokens)", ""])
    report.extend(
        table(
            [
                "strategy",
                "table chunks over 512, no budget",
                "with budget",
                "table hits@5, no budget",
                "with budget",
            ],
            budget_table,
        )
    )
    report.append("")
    report.append(
        "Rule fixed before the run: ADOPT if table chunks over 512 tokens drop to 0 AND hits@5 "
        "on table questions drop by no more than 1 question, for both strategies; otherwise KEEP."
    )
    report.append(f"Outcome: **{budget_verdict}** (the default changes only in a follow-up).")

    # 8. Very short documents
    words_by_document = {document.entry.id: document.profile.words for document in documents}
    short_rows = [
        [document.entry.id, str(document.profile.words), str(document.profile.page_count)]
        for document in documents
    ]
    short_verdict = very_short_decision(words_by_document)
    short_ids = ", ".join(very_short_documents(words_by_document)) or "none"
    report.extend(["", "## Very short documents", ""])
    report.extend(table(["document", "words", "pages"], short_rows))
    report.append("")
    report.append(
        f'Rule fixed before the run: no "very short" rule unless at least '
        f"{VERY_SHORT_MIN_DOCUMENTS} dev documents are under {VERY_SHORT_WORDS} words. "
        f"Under the limit: {short_ids}. Outcome: **{short_verdict}**."
    )

    # 9. Top-1 scores and the not-found threshold
    answerable_scores = [
        run.retrieved[0].score
        for document in documents
        for run in router_runs(document.entry.id)
        if run.retrieved
    ]
    proposal = propose_threshold(answerable_scores, off_topic_scores)
    report.extend(["", "## Top-1 scores under the router, and the not-found threshold", ""])
    score_rows: list[list[str]] = []
    bins = round(1 / SCORE_BIN)
    for position in range(bins):
        # Both edges from the position, so one bin's top is exactly the next bin's bottom.
        low = position * SCORE_BIN
        high = (position + 1) * SCORE_BIN
        last = position == bins - 1
        in_bin = [
            sum(1 for score in scores if low <= score < high or (last and score >= high))
            for scores in (answerable_scores, off_topic_scores)
        ]
        if any(in_bin):
            score_rows.append([f"{low:.2f}-{high:.2f}", str(in_bin[0]), str(in_bin[1])])
    report.extend(table(["top-1 score", "answerable questions", "off-topic questions"], score_rows))
    report.append("")
    for name, scores in (("answerable", answerable_scores), ("off-topic", off_topic_scores)):
        if scores:
            report.append(
                f"{name}: n {len(scores)}, min {min(scores):.3f}, "
                f"median {statistics.median(scores):.3f}, max {max(scores):.3f}."
            )
    current = settings.retrieval_min_score
    below_current = sum(1 for score in answerable_scores if score < current)
    above_current = sum(1 for score in off_topic_scores if score >= current)
    report.append(
        f"Current retrieval_min_score {current}: {below_current} answerable questions fall below "
        f"it, {above_current} off-topic questions pass it."
    )
    if proposal is None:
        report.append("No threshold proposal: the scores do not form two groups.")
    else:
        report.append(
            f"Proposal: {proposal.threshold:.3f} puts {proposal.correct} of {proposal.total} "
            "scores on the right side (answerable at or above, off-topic below). The setting is "
            "NOT changed here: the owner decides. Off-topic questions are the easy case; "
            "in-domain questions without an answer score much higher (Stage 2 notes)."
        )

    # Everything above is also printed; the sections below go to summary.md only.
    print("\n".join(report))

    # 10. Section-aware: boundaries vs prefix
    report.extend(["", "## Section-aware: boundaries or prefix?", ""])
    report.append(
        "Stage 4 cannot separate the two. Section-aware chunks differ from recursive chunks in "
        "where they are cut AND in the heading prefix that is embedded with them. The relevance "
        "rule ignores the prefix, but retrieval does not. The Stage 5 switch (the same "
        "boundaries with and without the prefix) will separate them."
    )

    # 11. Chunk boundaries side by side
    section_routed = [d.entry.id for d in documents if d.routed == SectionAwareStrategy.name]
    boundary_ids = section_routed[:1]
    if RECURSIVE_BOUNDARY_DOCUMENT in by_id:
        boundary_ids.append(RECURSIVE_BOUNDARY_DOCUMENT)
    for name in boundary_ids:
        document = by_id[name]
        report.extend(
            [
                "",
                f"## Chunk boundaries: {name} (routed: {document.routed}), "
                f"first {BOUNDARY_CHUNKS} chunks of each strategy",
                "",
            ]
        )
        boundary_rows: list[list[str]] = []
        for position in range(BOUNDARY_CHUNKS):
            cells = [str(position)]
            for label in BASE:
                stored = document.chunks[label]
                if position < len(stored):
                    chunk = stored[position]
                    start = cell(chunk.text, BOUNDARY_CHARS)
                    where = pages(chunk.page_start, chunk.page_end)
                    cells.append(f"{where}, {word_count(chunk.text)}w: {start}")
                else:
                    cells.append("")
            boundary_rows.append(cells)
        report.extend(table(["#", *BASE], boundary_rows))
        totals = ", ".join(f"{label} {len(document.chunks[label])}" for label in BASE)
        report.extend(["", f"Chunks in total: {totals}."])

    # 12. Every question the router failed (no relevant chunk in the top 5)
    report.extend(["", f"## Failed questions under the router (not in the top {HEADLINE_K})", ""])
    failures = 0
    for document in documents:
        name = document.entry.id
        for position, run in enumerate(router_runs(name)):
            if run.rank is not None and run.rank <= HEADLINE_K:
                continue
            failures += 1
            found_by = [
                label
                for label in BASE
                if (rank := runs[label][name][position].rank) is not None and rank <= HEADLINE_K
            ]
            question = run.question
            rank_text = "none in the top 10" if run.rank is None else str(run.rank)
            report.append(f"### {question.id} ({question.type}, {name}, routed: {document.routed})")
            report.append("")
            report.append(f"- Question: {question.question}")
            report.append(f"- Answer: {question.answer}")
            for item in question.evidence:
                report.append(f"- Evidence p{item.page}: {cell(item.text, 400)}")
            report.append(f"- Router rank of the first relevant chunk: {rank_text}")
            finders = ", ".join(found_by) or "no strategy"
            report.append(f"- Found in the top {HEADLINE_K} by: {finders}")
            for chunk_rank, chunk in enumerate(run.retrieved[:FAILURE_TOP], start=1):
                where = pages(chunk.page_start, chunk.page_end)
                report.append(
                    f"- Retrieved {chunk_rank}: score {chunk.score:.3f}, {where}: "
                    f"{cell(chunk.text, FAILURE_CHARS)}"
                )
            report.append("")
    if failures == 0:
        report.append("None.")

    results: dict[str, Any] = {
        "stage": "stage4",
        "embedding_model": embedder.model_name,
        "parser_version": PARSER_VERSION,
        "top_k": TOP_K,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES},
        "rows": [row_json(row) for row in rows],
        "budget_rows": [row_json(row) for row in budget_rows],
        "owner_rows": [row_json(row) for row in owner_rows],
        "best_single": best_single,
        "by_type": type_json,
        "by_document": {
            document.entry.id: {
                "routed": document.routed,
                "best": best_per_document[document.entry.id],
                "questions": len(ranks[BASE[0]][document.entry.id]),
                "hits_at_5": {
                    label: hits_at_k(ranks[label][document.entry.id], HEADLINE_K)
                    for label in VARIANTS
                },
                "recall_at_5": {
                    label: round(recall_at_k(ranks[label][document.entry.id], HEADLINE_K), 4)
                    for label in VARIANTS
                },
                "words": document.profile.words,
                "pages": document.profile.page_count,
            }
            for document in documents
        },
        "paired": pair_json,
        "ties": [list(pair) for pair in ties],
        "evidence_split": split_json,
        "glued": {
            "documents": glued_json,
            "questions_with_glued_evidence": glued_generated,
            "owner_questions_with_glued_evidence": glued_owner,
        },
        "table_budget": {
            "budget": TABLE_TOKEN_BUDGET,
            "comparisons": [
                {
                    "strategy": comparison.strategy,
                    "over_limit_without": comparison.over_limit_without,
                    "over_limit_with": comparison.over_limit_with,
                    "table_hits_without": comparison.table_hits_without,
                    "table_hits_with": comparison.table_hits_with,
                }
                for comparison in comparisons
            ],
            "decision": budget_verdict,
        },
        "very_short": {
            "under_limit": very_short_documents(words_by_document),
            "decision": short_verdict,
        },
        "router": {
            "losing_documents": list(router.losing_documents),
            "decision": router.verdict,
        },
        "threshold": {
            "current": current,
            "proposal": None if proposal is None else round(proposal.threshold, 4),
            "correct": None if proposal is None else proposal.correct,
            "total": None if proposal is None else proposal.total,
            "answerable_top1": [round(score, 4) for score in answerable_scores],
            "off_topic_top1": [round(score, 4) for score in off_topic_scores],
        },
        "questions": [
            {
                "id": run.question.id,
                "document": document.entry.id,
                "type": run.question.type,
                "ranks": {
                    label: runs[label][document.entry.id][position].rank for label in VARIANTS
                },
            }
            for document in documents
            for position, run in enumerate(runs[BASE[0]][document.entry.id])
        ],
        "owner_questions": [
            {
                "id": run.question.id,
                "ranks": {label: owner_runs[label][position].rank for label in BASE},
            }
            for position, run in enumerate(owner_runs[BASE[0]])
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.json"
    summary_path = RESULTS_DIR / "summary.md"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print()
    print(f"router failures listed in the summary: {failures}")
    print(f"wrote {results_path.relative_to(REPO_ROOT)} and {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
