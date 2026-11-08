"""Print the document profile of every PDF in a dev folder (developer tool).

Usage:
    python scripts/profile_report.py data/samples

The router's rules are written from these numbers, so only dev documents may be
measured: any path inside data/test_set is refused (note 0017).
"""

import argparse
import sys
from pathlib import Path

from adaptrag.parsing.blocks import BlockType
from adaptrag.parsing.pdf import ParseError, parse_pdf
from adaptrag.routing.profile import profile_document
from adaptrag.routing.router import route

TEST_SET = (Path(__file__).resolve().parents[1] / "data" / "test_set").resolve()

HEADINGS_SHOWN = 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Print document profiles for a dev folder.")
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    folder = args.folder.resolve()
    if folder.is_relative_to(TEST_SET):
        print("Error: this report is for dev documents only; data/test_set is not allowed.")
        sys.exit(1)

    for path in sorted(folder.glob("*.pdf")):
        try:
            doc = parse_pdf(path)
        except ParseError as exc:
            print(f"{path.name}: REJECTED ({type(exc).__name__}) {exc}")
            continue

        profile = profile_document(doc)
        fields = " ".join(f"{key}={value}" for key, value in profile.to_dict().items())
        print(f"{path.name} {fields}")

        decision = route(profile)
        print(f"  route: {decision.strategy} ({decision.rule}) {decision.reason}")

        headings = [block.text for block in doc.blocks if block.block_type is BlockType.HEADING]
        for text in headings[:HEADINGS_SHOWN]:
            print(f"  {text}")


if __name__ == "__main__":
    main()
