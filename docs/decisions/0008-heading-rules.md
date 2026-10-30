# 0008: Heading detection rules v1

Date: 2026-10-30
Status: Accepted

## Context
The section-aware chunker and the router's main rule ("headings found")
depend on finding headings. Inspection (scripts/inspect_pdf.py) showed that
no single signal works alone: subsections are body-sized, some bold text is
not a heading, and some large text (the arXiv stamp) is not a heading.

## Decision
1. Body size = most common span size in the whole document, weighted by
   characters, rounded to the nearest 0.5.
2. A block is a heading if: every span is bold, its size >= body size, and it
   has at most 15 words.
3. Level from numbering: "3" -> 1, "3.1" -> 2, "3.1.2" -> 3; no number -> 1.
4. heading_path from a stack: a level-L heading drops entries at level L or
   deeper, then is added. A heading block's path ends with itself.

## Evidence
- paper_01: sections 12.0 bold; subsections "3.1", "3.2" 10.0 bold (= body);
  "Encoder:" bold but inside a paragraph block; permission notice 12.0 not
  bold; arXiv stamp 20.0 not bold.
- paper_02: "1. Introduction", "2. Related Work" 12.0 bold; "Residual
  Representations." bold but inside a paragraph block.
- paper_03: "2 Related work", "4 Theoretical Results" 12.0 bold.

## Rejected
- "Bigger than body text": misses every subsection (paper_01 "3.1").
- "Any bold text": catches run-in labels ("Encoder:", "Residual
  Representations.") and author names.
- "Largest font on the page": picks the arXiv stamp (20.0).

## Known limits
- IEEE-style headings (paper_05: "I. INTRODUCTION", not bold, inside a
  paragraph block) are not detected. Revisit if the benchmark has many.
- The title and "Abstract" become level-1 headings.
- A heading with a non-bold footnote marker would be missed.
