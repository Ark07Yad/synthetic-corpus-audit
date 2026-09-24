"""Zipf's law, frequency spectrum, and Heaps' law signals (distributional family).

Natural language obeys Zipf's law: a word's frequency is roughly inversely proportional
to its frequency rank, giving a near-straight line of slope about -1 on a log-log plot.
The part that matters for contamination is the **tail** — the rare words. Nucleus and
top-k sampling suppress low-probability tokens at every decoding step, so generated text
has a thinner tail than natural language does.

Three complementary measurements, each of a different property:

``zipf_slope``      the exponent of the rank/frequency curve — *how steeply* frequency
                    falls with rank.
``spectrum_slope``  the exponent of the frequency spectrum (how many types occur once,
                    twice, three times ...) — *how much mass sits in the tail*.
``heaps_beta``      vocabulary growth — *how fast new words keep arriving*.

Why the rank/frequency tail is not fitted directly
--------------------------------------------------
The obvious implementation — split the rank/frequency curve into a head and a tail,
fit both, compare — was built and then removed, because it does not work at any corpus
size chaff will realistically see. Word frequencies are small integers, so the rare half
of the vocabulary takes only a handful of distinct values. At 20,000 tokens over 3,000
types the tail's frequencies are literally ``{1, 2}``: a two-valued step function. A
least-squares line through it either fails the R-squared guard or returns a confident,
meaningless slope.

The frequency spectrum is the standard way around this and is what this module fits
instead. Instead of asking "what frequency does rank *r* have" it asks "how many types
have frequency *i*", which is smooth and well-behaved at moderate sample sizes.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional, Sequence, Tuple

from ..document import Document
from ..stats import linear_regression
from ..tokenization import TextView
from . import FAMILY_DISTRIBUTIONAL, Signal, register

#: Minimum distinct word types before a full-range Zipf slope is fitted at all.
MIN_ZIPF_TYPES = 40

#: Minimum tokens before the frequency spectrum is fitted. Measured, not guessed: on
#: controlled Zipfian corpora the separation between a full and a truncated tail is
#: ~3.0 at 20k tokens, ~2.0 at 5k, ~0.56 at 2k and ~0.04 at 500. Below roughly 2,000
#: tokens the signal is indistinguishable from noise, so it is withheld. This is why
#: the short documents in ``data/samples/`` produce no spectrum slope — correct
#: behaviour, not a gap.
MIN_SPECTRUM_WORDS = 2000

#: Frequency-spectrum points fitted (types occurring 1..N times).
SPECTRUM_MAX_I = 10
MIN_SPECTRUM_POINTS = 4

#: A fit is only trusted when the relationship is actually linear.
MIN_FIT_R2 = 0.50

#: Heaps' law needs several vocabulary-growth observations to fit an exponent.
MIN_HEAPS_WORDS = 100
HEAPS_SAMPLE_POINTS = 12


def _rank_frequency(words: Sequence[str]) -> List[int]:
    """Word frequencies sorted descending — the Zipf curve's y-axis."""
    return sorted(Counter(words).values(), reverse=True)


def zipf_slope(words: Sequence[str]) -> Optional[Tuple[float, float]]:
    """Full-range Zipf exponent and its R-squared, or ``None`` if not estimable.

    Verified to recover a known exponent: synthetic corpora generated at true
    exponents of 0.8 / 1.0 / 1.2 / 1.5 measure at -0.92 / -1.04 / -1.11 / -1.27.
    Finite sampling compresses the estimate toward -1, but the ordering is preserved,
    which is all a corpus-relative z-score needs.
    """
    freqs = _rank_frequency(words)
    if len(freqs) < MIN_ZIPF_TYPES or len(set(freqs)) < 2:
        return None
    xs = [math.log(i + 1) for i in range(len(freqs))]
    ys = [math.log(f) for f in freqs]
    slope, _intercept, r2 = linear_regression(xs, ys)
    return slope, r2


def spectrum_slope(words: Sequence[str]) -> Optional[Tuple[float, float]]:
    """Frequency-spectrum exponent: fit ``log(V_i) ~ log(i)`` for ``i = 1..10``.

    ``V_i`` is the number of word types occurring exactly ``i`` times. A corpus with a
    rich tail is dominated by hapax legomena, so ``V_1 >> V_2 >> V_3`` and the slope is
    steeply negative. When sampling truncates the tail, the hapax count collapses
    toward the counts of more frequent types, the spectrum flattens, and the slope rises
    toward zero and beyond.

    On controlled corpora: 3,000 types gives -1.60, 1,200 types gives -0.59, 600 types
    gives +1.11, 300 types gives +1.44. Monotonic across the whole range, which is what
    makes it usable as a directed signal.

    Known blind spot: a pathologically truncated document — thousands of tokens over a
    few hundred types — can leave the ``i = 1..10`` bins empty altogether, so fewer than
    :data:`MIN_SPECTRUM_POINTS` remain and this returns ``None`` exactly where the
    evidence is strongest. ``heaps_beta`` covers that case (0.32 against 0.68 on the
    same pair of corpora), which is the practical argument for carrying several
    partially-redundant signals in one family rather than one clever one.
    """
    spectrum = Counter(Counter(words).values())
    points = [(i, spectrum[i]) for i in range(1, SPECTRUM_MAX_I + 1) if spectrum.get(i, 0) > 0]
    if len(points) < MIN_SPECTRUM_POINTS:
        return None
    xs = [math.log(i) for i, _ in points]
    ys = [math.log(v) for _, v in points]
    slope, _intercept, r2 = linear_regression(xs, ys)
    return slope, r2


def heaps_beta(words: Sequence[str]) -> Optional[float]:
    """Heaps' law exponent: vocabulary ``V`` grows as ``V ~ K * n^beta``.

    Measures the rate at which new words keep arriving as a document proceeds — a
    direct read on tail richness that, unlike a raw type count, is not a function of
    document length. Human prose sits around 0.4-0.6. Text whose vocabulary saturates
    early yields a lower exponent; on controlled corpora a full tail gives 0.68 against
    0.32 for a truncated one, at identical token counts.

    Sampled at log-spaced prefixes so the early, fast-growing part of the curve is not
    swamped by the long flat part, which is what uniform spacing would do.
    """
    n = len(words)
    if n < MIN_HEAPS_WORDS:
        return None

    start, stop = math.log(50), math.log(n)
    step = (stop - start) / (HEAPS_SAMPLE_POINTS - 1)
    positions = sorted({int(math.exp(start + i * step)) for i in range(HEAPS_SAMPLE_POINTS)})
    positions = [p for p in positions if 50 <= p <= n]
    if len(positions) < 5:
        return None

    xs: List[float] = []
    ys: List[float] = []
    seen = set()
    cursor = 0
    for position in positions:
        while cursor < position:
            seen.add(words[cursor])
            cursor += 1
        if seen:
            xs.append(math.log(position))
            ys.append(math.log(len(seen)))

    if len(xs) < 5:
        return None
    slope, _intercept, r2 = linear_regression(xs, ys)
    return slope if r2 >= MIN_FIT_R2 else None


@register("zipf_profile", FAMILY_DISTRIBUTIONAL)
def zipf_profile(doc: Document, view: TextView) -> List[Signal]:
    """Emit Zipf, spectrum and Heaps signals, withholding any the document is too
    short to support. Grouped into one extractor because they share a frequency count."""
    words = view.words
    signals: List[Signal] = []

    fit = zipf_slope(words)
    if fit is not None and fit[1] >= MIN_FIT_R2:
        signals.append(
            Signal(
                name="zipf_slope",
                value=fit[0],
                family=FAMILY_DISTRIBUTIONAL,
                direction=-1,
                description="Log-log rank/frequency exponent. More negative means frequency falls away faster with rank.",
            )
        )

    beta = heaps_beta(words)
    if beta is not None:
        signals.append(
            Signal(
                name="heaps_beta",
                value=beta,
                family=FAMILY_DISTRIBUTIONAL,
                direction=-1,
                description="Heaps' law vocabulary-growth exponent. Lower means new words stop arriving sooner.",
            )
        )

    if len(words) >= MIN_SPECTRUM_WORDS:
        spectrum = spectrum_slope(words)
        if spectrum is not None and spectrum[1] >= MIN_FIT_R2:
            signals.append(
                Signal(
                    name="spectrum_slope",
                    value=spectrum[0],
                    family=FAMILY_DISTRIBUTIONAL,
                    direction=1,
                    description="Frequency-spectrum exponent. Higher (flatter) means the rare-word tail has been thinned.",
                )
            )

    return signals
