"""N-gram repetition and compressibility signals (distributional family).

Where ``entropy.py`` asks whether the *vocabulary* is impoverished, this module asks
whether the *phrasing* is. Generated text reuses multi-word constructions far more than
human writing does, at two distinct scales:

* **phrase scale (4-grams)** — "it is important to", "plays a crucial role"
* **passage scale (8-grams)** — whole clauses restated with minor substitutions

These are separate phenomena, so they are separate signals rather than one averaged
"repetition" number: a document can be phrase-repetitive without being passage-
repetitive (a technical reference), and the reverse (a looping argument).

Compressibility is carried alongside them as a cheap, assumption-free cross-check.
``zlib`` finds redundancy that fixed-width n-grams miss — repeated structure at any
scale and offset, including whitespace and formatting rhythm.
"""

from __future__ import annotations

import zlib
from typing import List, Optional, Sequence

from ..document import Document
from ..tokenization import TextView, ngrams
from . import FAMILY_DISTRIBUTIONAL, Signal, register

#: Compression level. 6 is zlib's default: higher levels search harder for distant
#: matches, which would make the ratio depend on the encoder's effort budget rather
#: than on the text. The level is pinned so scores are reproducible (OBJECTIVE G5).
COMPRESSION_LEVEL = 6

#: A document needs several times ``n`` tokens before an n-gram repetition rate means
#: anything — with 20 tokens there are only 13 8-grams and the rate is quantised to
#: thirteenths.
MIN_NGRAM_MULTIPLE = 10

#: zlib needs a few hundred bytes before its dictionary warms up; below this the ratio
#: measures encoder overhead rather than textual redundancy.
MIN_COMPRESSION_CHARS = 500


def repetition_rate(tokens: Sequence[str], n: int) -> Optional[float]:
    """Share of n-grams that are repeats: ``1 - distinct / total``.

    0.0 means every n-gram in the document is unique; values approaching 1.0 mean the
    document is built from a small set of recycled spans. Returns ``None`` when the
    document is too short for the rate to carry information.
    """
    if len(tokens) < n * MIN_NGRAM_MULTIPLE:
        return None
    grams = list(ngrams(tokens, n))
    if not grams:
        return None
    return 1.0 - (len(set(grams)) / float(len(grams)))


def compression_ratio(text: str, level: int = COMPRESSION_LEVEL) -> Optional[float]:
    """Compressed size divided by raw size, both in UTF-8 bytes.

    Lower means more redundant. This is deliberately a whole-text measurement rather
    than a token-level one: it sees repeated markdown scaffolding, indentation patterns
    and punctuation rhythm that a word n-gram never will.

    Length caveat: the ratio falls as documents get longer, because zlib's dictionary
    has more history to exploit. It is therefore only comparable between documents of
    similar length — which corpus-relative z-scoring handles partially, and which
    phase 5 will need to handle properly with length conditioning.
    """
    raw = text.encode("utf-8")
    if len(raw) < MIN_COMPRESSION_CHARS:
        return None
    return len(zlib.compress(raw, level)) / float(len(raw))


@register("ngram_repetition", FAMILY_DISTRIBUTIONAL)
def ngram_repetition(doc: Document, view: TextView) -> List[Signal]:
    """Emit phrase-scale and passage-scale repetition plus compressibility."""
    signals: List[Signal] = []
    # Repetition is measured over the full token stream, punctuation included:
    # synthetic text reuses punctuation shape ("**X:** ...") as much as wording.
    tokens = view.tokens

    rep4 = repetition_rate(tokens, 4)
    if rep4 is not None:
        signals.append(
            Signal(
                name="repetition_4gram",
                value=rep4,
                family=FAMILY_DISTRIBUTIONAL,
                direction=1,
                description="Share of 4-grams that are repeats. Higher means recycled phrasing.",
            )
        )

    rep8 = repetition_rate(tokens, 8)
    if rep8 is not None:
        signals.append(
            Signal(
                name="repetition_8gram",
                value=rep8,
                family=FAMILY_DISTRIBUTIONAL,
                direction=1,
                description="Share of 8-grams that are repeats. Higher means whole clauses are being restated.",
            )
        )

    ratio = compression_ratio(view.raw)
    if ratio is not None:
        signals.append(
            Signal(
                name="compression_ratio",
                value=ratio,
                family=FAMILY_DISTRIBUTIONAL,
                direction=-1,
                description="zlib compressed size over raw size. Lower means more redundant structure.",
            )
        )

    return signals
