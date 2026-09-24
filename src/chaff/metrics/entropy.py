"""Lexical diversity and entropy signals (distributional family).

These answer CORE_PROBLEM §3.1: does this document's vocabulary behave like natural
language, or like language that has been through a sampler that truncates the
low-probability tail at every decoding step?

Length sensitivity is the recurring trap in this family. Type/token ratio falls as a
document gets longer *for purely mechanical reasons*, so a naive TTR signal would rank
documents by length and call it contamination. Every metric here is either
length-robust by construction (MTLD, Yule's K) or is normalised before it is emitted.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional, Sequence

from ..document import Document
from ..stats import shannon_entropy
from ..tokenization import TextView
from . import FAMILY_DISTRIBUTIONAL, Signal, register

#: MTLD's standard factor threshold. A "factor" ends when the running type/token ratio
#: falls to this value; the metric is the mean token count per factor. 0.72 is the
#: value from McCarthy & Jarvis (2010) and is kept rather than tuned — a bespoke
#: threshold would make chaff's numbers incomparable with published lexical-diversity
#: work for no measurable gain.
MTLD_THRESHOLD = 0.72

#: Below this many words MTLD returns its degenerate fallback, so it is not emitted.
MIN_MTLD_WORDS = 50

#: Conditional bigram entropy needs at least this many repeated contexts before it is
#: measuring branching rather than sampling noise. See the docstring for why measuring
#: over all contexts produces a non-monotonic, unusable signal.
MIN_BRANCHING_CONTEXTS = 5


def _mtld_one_direction(words: Sequence[str], threshold: float = MTLD_THRESHOLD) -> float:
    """MTLD over a single pass of the token sequence."""
    factors = 0.0
    types = set()
    count = 0

    for word in words:
        count += 1
        types.add(word)
        if len(types) / count <= threshold:
            factors += 1.0
            types.clear()
            count = 0

    if count > 0:
        ttr = len(types) / count
        # The trailing partial factor is scaled by how far it got toward the
        # threshold, otherwise documents would be penalised for their tail length.
        if ttr < 1.0:
            factors += (1.0 - ttr) / (1.0 - threshold)

    if factors <= 0.0:
        # Never dropped to the threshold: maximally diverse over this length. The
        # conventional fallback is the token count itself.
        return float(len(words))
    return len(words) / factors


def mtld(words: Sequence[str]) -> float:
    """Bidirectional Measure of Textual Lexical Diversity.

    Averaging a forward and a reverse pass removes the ordering artifact that makes
    single-direction MTLD sensitive to where repetition happens to sit in a document.
    Higher means more lexically diverse.
    """
    if not words:
        return 0.0
    forward = _mtld_one_direction(words)
    backward = _mtld_one_direction(list(reversed(words)))
    return (forward + backward) / 2.0


def yules_k(words: Sequence[str]) -> float:
    """Yule's K characteristic: ``10^4 * (sum(i^2 * V_i) - N) / N^2``.

    ``V_i`` is the number of types occurring exactly ``i`` times and ``N`` is the token
    count. K measures the probability that two tokens drawn at random are the same
    word, so it rises with repetition. It is asymptotically independent of document
    length, which is why it is here and raw TTR is not.
    """
    n = len(words)
    if n < 2:
        return 0.0
    spectrum = Counter(Counter(words).values())
    m2 = sum((i * i) * v for i, v in spectrum.items())
    return 10_000.0 * (m2 - n) / float(n * n)


def conditional_bigram_entropy(words: Sequence[str]) -> Optional[float]:
    """``H(w_i | w_{i-1})`` in bits, measured **only over contexts seen more than once**.

    The average branching factor of the text: given the previous word, how many
    different words genuinely follow it? Decoded text has fewer plausible
    continuations per context than human text does.

    The restriction to repeated contexts is load-bearing, not a detail. Computed over
    *all* bigrams this quantity is non-monotonic and therefore unusable as a directed
    signal: it goes to 0 for deterministically repetitive text (every context has one
    continuation) **and** to 0 for maximally diverse text (every context occurs once,
    so its observed continuation is also unique). Both extremes score identically.

    A context seen once carries no information about branching — there was never a
    choice to observe. Restricting to contexts with count >= 2 measures branching only
    where branching is observable, which restores monotonicity: the metric now falls
    as text becomes more templated and is undefined (not zero) when there is no
    evidence either way.

    Returns ``None`` when fewer than :data:`MIN_BRANCHING_CONTEXTS` contexts repeat.
    ``None`` and ``0.0`` mean different things and must not be collapsed: ``None`` is
    "no evidence", while ``0.0`` is the strong observation that every repeated context
    has exactly one continuation — the most templated text possible, and precisely what
    this family exists to catch.
    """
    if len(words) < 3:
        return None

    bigrams = Counter(zip(words, words[1:]))
    contexts = Counter(words[:-1])
    repeated = {c for c, n in contexts.items() if n >= 2}
    if len(repeated) < MIN_BRANCHING_CONTEXTS:
        return None

    # Renormalise over the retained contexts so the result is a proper weighted
    # average of per-context entropies rather than a fraction of the full total.
    total = float(sum(n for (c, _), n in bigrams.items() if c in repeated))
    if total <= 0:
        return None

    entropy = 0.0
    for (context, _nxt), joint_count in bigrams.items():
        if context not in repeated:
            continue
        p_joint = joint_count / total
        p_conditional = joint_count / contexts[context]
        entropy -= p_joint * math.log(p_conditional, 2)
    return entropy


def hapax_ratio(words: Sequence[str]) -> float:
    """Share of the vocabulary occurring exactly once.

    The most direct symptom of a truncated distribution tail. Human prose typically
    sits around 0.4-0.6 at document scale; systematically lower is the signal.
    """
    if not words:
        return 0.0
    counts = Counter(words)
    return sum(1 for c in counts.values() if c == 1) / float(len(counts))


@register("lexical_profile", FAMILY_DISTRIBUTIONAL)
def lexical_profile(doc: Document, view: TextView) -> List[Signal]:
    """Emit the entropy and lexical-diversity signals for one document.

    Grouped into a single extractor because all six share one frequency distribution;
    computing it once is the only reason this is not six separate functions.
    """
    words = view.words
    if len(words) < 3:
        return []

    counts = Counter(words)
    n_types = len(counts)

    unigram_h = shannon_entropy(counts.values())
    # Entropy efficiency: how close the distribution is to uniform over its own
    # vocabulary. Normalising by log2(V) removes the vocabulary-size dependence that
    # makes raw entropy track document length.
    normalized_h = unigram_h / math.log(n_types, 2) if n_types > 1 else 0.0

    signals = [
        Signal(
            name="unigram_entropy",
            value=unigram_h,
            family=FAMILY_DISTRIBUTIONAL,
            direction=-1,
            description="Shannon entropy of the word distribution, in bits. Lower means a more compressed vocabulary.",
        ),
        Signal(
            name="normalized_entropy",
            value=normalized_h,
            family=FAMILY_DISTRIBUTIONAL,
            direction=-1,
            description="Unigram entropy divided by log2(vocabulary). Length-corrected; lower means more concentrated word use.",
        ),
        Signal(
            name="hapax_ratio",
            value=hapax_ratio(words),
            family=FAMILY_DISTRIBUTIONAL,
            direction=-1,
            description="Share of vocabulary seen exactly once. Lower means a thinner distribution tail.",
        ),
        Signal(
            name="yules_k",
            value=yules_k(words),
            family=FAMILY_DISTRIBUTIONAL,
            direction=1,
            description="Yule's K: length-independent repetition. Higher means more word reuse.",
        ),
    ]

    branching = conditional_bigram_entropy(words)
    if branching is not None:
        # None means "not measurable on this document" (too few repeated contexts).
        # A measured 0.0 is the opposite: maximally templated. Only the former is
        # withheld from fusion.
        signals.append(
            Signal(
                name="conditional_bigram_entropy",
                value=branching,
                family=FAMILY_DISTRIBUTIONAL,
                direction=-1,
                description="H(word | previous word) over repeated contexts, in bits. Lower means fewer plausible continuations.",
            )
        )

    if len(words) >= MIN_MTLD_WORDS:
        signals.append(
            Signal(
                name="mtld",
                value=mtld(words),
                family=FAMILY_DISTRIBUTIONAL,
                direction=-1,
                description="Bidirectional MTLD lexical diversity. Lower means less diverse.",
            )
        )
    return signals
