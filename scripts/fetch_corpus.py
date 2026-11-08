"""Download missing benchmark documents and record their SHA-256 in the manifest.

Files that already exist are not downloaded again, so re-running the script is the retry
after a failure. A recorded hash is never overwritten: a different file is reported as a
mismatch (note 0017).

Test documents are parsed only to print page and word counts. Their text is never
printed. Do not open them or use them for tuning until the final evaluation.

Usage: python scripts/fetch_corpus.py [--manifest eval/corpus.json]
"""

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from urllib.parse import urlsplit

from adaptrag.evaluation.corpus import CATEGORIES, SPLITS, CorpusEntry, parse_corpus
from adaptrag.parsing.pdf import ParseError, parse_pdf
from adaptrag.storage import compute_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]

USER_AGENT = "adaptrag-corpus-fetch/1.0 (research benchmark; https://github.com/Usama478/adaptrag)"
TIMEOUT_SECONDS = 60
MAX_BYTES = 50 * 1024 * 1024
PDF_MAGIC = b"%PDF-"
ARXIV_HOST = "arxiv.org"
ARXIV_MIN_GAP_SECONDS = 3.0
FIRST_TEXT_CHARS = 150

RECORDED = "recorded"
OK = "ok"
FAILED = "FAILED"


class FetchError(Exception):
    """This document failed. The message is the short reason printed on its line."""


@dataclass(slots=True)
class Result:
    """What happened to one document."""

    status: str = FAILED
    detail: str = ""
    downloaded: bool = False
    new_hash: str | None = None
    first_text: str | None = None


class ArxivPacer:
    """Keeps requests to arxiv.org at least ARXIV_MIN_GAP_SECONDS apart."""

    def __init__(self) -> None:
        self._last_request: float | None = None

    def wait_if_needed(self, url: str) -> None:
        """Sleep if `url` is on arXiv and the last arXiv request was too recent."""
        if urlsplit(url).hostname != ARXIV_HOST:
            return
        if self._last_request is not None:
            remaining = ARXIV_MIN_GAP_SECONDS - (time.monotonic() - self._last_request)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request = time.monotonic()


def download(url: str, path: Path, pacer: ArxivPacer) -> None:
    """Fetch `url` and save it at `path`. Raises FetchError; never leaves a half file."""
    pacer.wait_if_needed(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            content: bytes = response.read(MAX_BYTES + 1)
            declared_length: str | None = response.headers.get("Content-Length")
    except (OSError, HTTPException) as exc:  # URLError and timeouts are both OSError
        raise FetchError(f"download failed: {exc}") from exc

    if len(content) > MAX_BYTES:
        raise FetchError(f"larger than {MAX_BYTES // (1024 * 1024)} MB")
    # read(n) returns fewer bytes without an error if the connection drops part-way.
    if declared_length is not None and declared_length != str(len(content)):
        raise FetchError("incomplete download")
    if not content.startswith(PDF_MAGIC):
        raise FetchError("not a PDF")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        part_path = path.with_suffix(".part")
        part_path.write_bytes(content)
        part_path.replace(path)
    except OSError as exc:
        raise FetchError(f"could not save the file: {exc}") from exc


def count_pages_and_words(entry: CorpusEntry, path: Path, result: Result) -> None:
    """Fill in the counts. Only dev entries with no source URL also get their first text."""
    try:
        parsed = parse_pdf(path)
    except ParseError as exc:
        raise FetchError(str(exc)) from exc

    words = sum(len(block.text.split()) for block in parsed.blocks)
    result.detail = f"pages={parsed.page_count} words={words}"
    if entry.split == "dev" and entry.source_url is None:
        result.first_text = parsed.blocks[0].text[:FIRST_TEXT_CHARS]


def process_entry(entry: CorpusEntry, pacer: ArxivPacer) -> Result:
    """Download (if needed), check the hash, and parse one document."""
    result = Result()
    path = REPO_ROOT / entry.path
    try:
        if not path.exists():
            if entry.source_url is None:
                raise FetchError("missing, no source URL")
            download(entry.source_url, path, pacer)
            result.downloaded = True

        digest = compute_sha256(path.read_bytes())
        if entry.sha256 is None:
            result.new_hash = digest
        elif digest != entry.sha256:
            raise FetchError("hash mismatch: file changed or new version at source")

        count_pages_and_words(entry, path, result)
        result.status = RECORDED if result.new_hash is not None else OK
    except FetchError as exc:
        result.status = FAILED
        result.detail = str(exc)
    return result


def format_line(entry: CorpusEntry, result: Result) -> str:
    """One output line for a document."""
    line = (
        f"{entry.split:<5} {entry.category:<18} {entry.id:<9} {result.status:<9} "
        f"{result.detail}  {entry.path}"
    )
    if result.first_text is not None:
        line += f'  first_text="{result.first_text}"'
    return line


def print_summary(entries: list[CorpusEntry], results: list[Result]) -> None:
    """Counts per split and category, then the totals."""
    counts = Counter((entry.split, entry.category) for entry in entries)
    print("\nDocuments per split and category:")
    for split in SPLITS:
        for category in CATEGORIES:
            print(f"  {split:<5} {category:<18} {counts[(split, category)]}")

    downloaded = sum(result.downloaded for result in results)
    recorded = sum(result.new_hash is not None for result in results)
    ok = sum(result.status == OK for result in results)
    failed = sum(result.status == FAILED for result in results)
    print(f"\nTotals: downloaded={downloaded} recorded={recorded} ok={ok} failed={failed}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download corpus PDFs and record their SHA-256.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "eval" / "corpus.json",
        help="default: eval/corpus.json",
    )
    args = parser.parse_args()

    manifest_path: Path = args.manifest
    # The raw JSON is kept so it can be written back with the same keys in the same order.
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = parse_corpus(data)

    pacer = ArxivPacer()
    results: list[Result] = []
    for entry, raw_document in zip(entries, data["documents"], strict=True):
        result = process_entry(entry, pacer)
        if result.new_hash is not None:
            raw_document["sha256"] = result.new_hash
        results.append(result)
        print(format_line(entry, result))

    if any(result.new_hash is not None for result in results):
        manifest_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print_summary(entries, results)
    return 1 if any(result.status == FAILED for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
