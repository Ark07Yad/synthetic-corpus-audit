"""Tokenization for pre-tokenizer-training analysis.

Design constraint that drives this whole module: chaff runs *before* a tokenizer is
trained on the corpus. It therefore cannot use one. Every view of the text here is
produced by deterministic unicode-aware regex, so the numbers chaff reports describe
the corpus itself rather than the artifacts of some particular BPE vocabulary.

A ``TextView`` is built once per document and handed to every metric family. Six
families re-tokenizing the same document was the obvious hot spot, so it is cached
up front instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, List, Sequence, Tuple

# A "word" is a run of letters, optionally joined by internal apostrophes
# (straight or curly) so "don't" and "don’t" survive as single tokens rather than
# fragmenting into three. Digits and underscores are excluded from word-hood on
# purpose: numeric density is its own signal and should not dilute lexical metrics.
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)

# Full tokenization: words, numbers (with internal separators), or a single
# punctuation/symbol character. Punctuation is kept as its own token because
# formatting-artifact detection in phase 4 depends on punctuation rhythm.
TOKEN_RE = re.compile(
    r"[^\W\d_]+(?:['’][^\W\d_]+)*"  # words
    r"|\d+(?:[.,:/]\d+)*"                # numbers, dates, versions
    r"|[^\w\s]",                         # single punctuation / symbol
    re.UNICODE,
)

# Abbreviations that end in a period without ending a sentence. Not exhaustive —
# it does not need to be, since sentence segmentation only feeds step-level
# reasoning analysis where an occasional bad split is noise, not a failure.
_ABBREVIATIONS = frozenset(
    """mr mrs ms dr prof sr jr st vs etc al fig eq no approx dept est inc ltd co
    jan feb mar apr jun jul aug sep sept oct nov dec i.e e.g cf ca vol pp ed""".split()
)

_SENT_BOUNDARY = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")
_WS_RUN = re.compile(r"[ \t]{2,}")


def tokenize(text: str) -> List[str]:
    """Words, numbers and punctuation, in order, preserving surface form."""
    return TOKEN_RE.findall(text)


def tokenize_words(text: str, lower: bool = True) -> List[str]:
    """Letter-only tokens. Lowercased by default for type/frequency counting."""
    words = WORD_RE.findall(text)
    return [w.lower() for w in words] if lower else words


def split_sentences(text: str) -> List[str]:
    """Split into sentences with a light abbreviation and decimal guard.

    Deliberately rule-based: pulling in an NLP dependency to improve an already
    adequate split would violate the zero-dependency constraint (see OBJECTIVE §2 G2).
    """
    if not text.strip():
        return []

    pieces = _SENT_BOUNDARY.split(text)
    out: List[str] = []
    buffer = ""

    for piece in pieces:
        candidate = (buffer + " " + piece).strip() if buffer else piece.strip()
        if not candidate:
            continue
        if _ends_on_abbreviation(candidate):
            buffer = candidate
            continue
        out.append(candidate)
        buffer = ""

    if buffer:
        out.append(buffer)
    return out


def _ends_on_abbreviation(chunk: str) -> bool:
    """True if ``chunk`` ends with a known abbreviation or a bare initial."""
    stripped = chunk.rstrip("\"'”’)]")
    if not stripped.endswith("."):
        return False
    tail = stripped[:-1].split()
    if not tail:
        return False
    last = tail[-1].lower().strip("([\"'")
    if last in _ABBREVIATIONS:
        return True
    # A single capital letter before the period is an initial ("J. R. R.").
    return len(last) == 1 and last.isalpha()


def split_lines(text: str) -> List[str]:
    """Non-empty lines, right-stripped. Leading whitespace is kept: markdown
    indentation is itself a formatting-artifact signal."""
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()]


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def ngrams(seq: Sequence[str], n: int) -> Iterator[Tuple[str, ...]]:
    """Yield overlapping n-grams. Empty when the sequence is shorter than ``n``."""
    if n <= 0:
        raise ValueError("n must be positive")
    for i in range(len(seq) - n + 1):
        yield tuple(seq[i : i + n])


@dataclass
class TextView:
    """All cached views of one document's text.

    Built once by :func:`build_view` and passed read-only to every metric family.
    Treat every field as immutable; metrics that need a transformed view should
    derive it locally rather than mutating shared state.
    """

    raw: str
    tokens: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)

    @property
    def n_chars(self) -> int:
        return len(self.raw)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def n_types(self) -> int:
        return len(set(self.words))

    @property
    def is_analysable(self) -> bool:
        """Below roughly 50 words every distributional metric is dominated by
        sampling noise, so short documents are reported rather than scored."""
        return self.n_words >= MIN_ANALYSABLE_WORDS


MIN_ANALYSABLE_WORDS = 50


def build_view(text: str) -> TextView:
    """Produce the shared, cached tokenization of one document."""
    return TextView(
        raw=text,
        tokens=tokenize(text),
        words=tokenize_words(text),
        sentences=split_sentences(text),
        lines=split_lines(text),
        paragraphs=split_paragraphs(text),
    )
