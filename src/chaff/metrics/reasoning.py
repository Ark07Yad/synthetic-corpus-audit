"""Repetitive-reasoning signals (reasoning family).

CORE_PROBLEM §3.3. The subtlest failure: a chain of steps, each grammatical and each
*appearing* to advance an argument, that introduces no new logical state. Step 4 is
step 2 with different connectives; the conclusion restates the question. This is worse
than useless in a pre-training corpus because it is reasoning-*shaped*: exactly the text
a data pipeline would upweight as chain-of-thought, while teaching the form of reasoning
without its substance.

The proxy for "logical state" is **content**: the non-function words a step contributes.
A step that advances an argument brings in something not in play before — a quantity, an
entity, a relation. A step that loops re-uses the vocabulary of the steps around it.
This is a proxy, not an understanding of logic; it cannot tell a valid inference from an
invalid one, and it is not meant to. It detects *stalled* chains, which is what
generated pseudo-reasoning looks like at the surface.

Novelty is measured against a sliding window of recent steps, not against everything
before. Measured against all prior steps it would fall mechanically as a document got
longer — the vocabulary already seen keeps growing (Heaps' law) — and long human documents
would look like loops. A window makes it a local property.

What measurement removed
------------------------
A third signal, ``restatement`` — each step's highest similarity to *any* earlier step —
was built and dropped. On the guaranteed-human baseline (``benchmarks/human_baseline.py``,
1,702 documents with enough steps) it correlated with the distributional family at
|rho| 0.51-0.74 across seven signals: -0.74 with compression ratio, -0.71 with hapax
ratio and Zipf slope, +0.68 with 4-gram repetition. It was largely re-measuring lexical
repetition. That matters beyond redundancy: the corroboration rule (OBJECTIVE §4.3)
counts two firing families as two pieces of evidence, so a signal that duplicates
another family lets one lexically repetitive human document masquerade as corroborated.
The two windowed signals that remain stay below |rho| 0.5 (at most 0.49), which is
weaker coupling but not zero — phase 5 must fuse with the measured correlation in mind.
"""

from __future__ import annotations

from typing import FrozenSet, List, Sequence

from ..document import Document
from ..stats import mean
from ..tokenization import TextView, tokenize_words
from . import FAMILY_REASONING, Signal, register

#: English function words: carry syntax, not state. Content is everything else.
STOPWORDS = frozenset("""
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing done down during
    each either else ever every few for from further had has have having he her here hers
    herself him himself his how however i if in into is it its itself just let lets like
    may me might more most much must my myself neither no nor not now of off on once only
    or other our ours ourselves out over own per quite rather same shall she should since
    so some still such than that the their theirs them themselves then there these they
    this those though through thus to too under until up upon us very via was we were
    what when where whether which while who whom whose why will with within without would
    yet you your yours yourself yourselves one two first second third next finally step
    therefore hence so let's we'll it's that's there's here's
""".split())

#: Steps with fewer content words than this are fragments ("Step 1:", "Yes.") and carry
#: no measurable state either way.
MIN_STEP_CONTENT = 2

#: A chain needs this many content-bearing steps before its shape means anything.
MIN_STEPS = 5

#: Novelty is measured against this many immediately preceding steps.
NOVELTY_WINDOW = 3

#: A step is "stalled" when fewer than this share of its content words are new
#: relative to the window.
STALL_THRESHOLD = 0.25

_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied",
             "ness", "edly", "ed", "es", "ly", "s")


def stem(word: str) -> str:
    """Strip one common suffix, keeping a stem of at least three letters.

    Deliberately crude. Its only job is to stop "cost" / "costs" / "costing" counting as
    three new pieces of state; a real stemmer would be a dependency (OBJECTIVE G2) for a
    marginal gain on a proxy measure.

    Possessives are stripped first. Without that, "component's" stemmed to "component'"
    and every possessive counted as new content — inflating novelty for exactly the
    text (human prose) that uses possessives most.
    """
    for possessive in ("'s", "\u2019s", "'", "\u2019"):
        if word.endswith(possessive) and len(word) > len(possessive):
            word = word[: -len(possessive)]
            break
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_of(sentence: str) -> FrozenSet[str]:
    return frozenset(
        stem(w) for w in tokenize_words(sentence)
        if w not in STOPWORDS and len(w) > 1
    )


def steps_of(view: TextView) -> List[FrozenSet[str]]:
    """Content sets of the document's sentences, dropping fragments."""
    steps = [content_of(s) for s in view.sentences]
    return [c for c in steps if len(c) >= MIN_STEP_CONTENT]


def windowed_novelty(steps: Sequence[FrozenSet[str]], window: int = NOVELTY_WINDOW) -> List[float]:
    """For each step after the first, the share of its content absent from the
    preceding ``window`` steps."""
    out: List[float] = []
    for i in range(1, len(steps)):
        recent = frozenset().union(*steps[max(0, i - window) : i])
        out.append(len(steps[i] - recent) / float(len(steps[i])))
    return out


@register("reasoning_profile", FAMILY_REASONING)
def reasoning_profile(doc: Document, view: TextView) -> List[Signal]:
    """Emit chain-shape signals when the document has enough content-bearing steps."""
    steps = steps_of(view)
    if len(steps) < MIN_STEPS:
        return []
    novelty = windowed_novelty(steps)
    return [
        Signal(
            name="step_novelty",
            value=mean(novelty),
            family=FAMILY_REASONING,
            direction=-1,
            description="Mean share of each sentence's content that is new relative to the previous three. Lower means the argument is not moving.",
        ),
        Signal(
            name="stalled_step_ratio",
            value=sum(1 for n in novelty if n < STALL_THRESHOLD) / float(len(novelty)),
            family=FAMILY_REASONING,
            direction=1,
            description="Share of sentences that add almost nothing new to the ones just before. Higher means a stalled chain.",
        ),
    ]
