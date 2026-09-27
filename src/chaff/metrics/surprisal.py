"""Surprisal signals from the corpus-internal language model (surprisal family).

Every word in a document gets a surprisal — ``-log2 P(word | two previous words)``
under the corpus-internal LM, with the document itself left out of the model. The
signals here describe the *shape* of that sequence rather than just its level.

The shape is the point. Decoding picks locally likely tokens at every step, so generated
text is uniformly unsurprising. Human text is **bursty**: a predictable stretch, then a
word nobody would have guessed, then predictable again — and at sentence scale, some
sentences are routine while others carry the new idea. Measuring only the *mean*
surprisal (plain perplexity) conflates two different kinds of low-surprisal text:
synthetic prose and genuinely formulaic human writing like contracts. Its spread is
where the two are supposed to separate.

What was measured, and what it changed
--------------------------------------
Candidates were tested on controlled corpora: one random Markov-chain language, "human"
documents sampled from its full distribution and "synthetic" ones through nucleus
(top-p 0.9) truncation — identical language and sentence lengths, only the sampler
differs. AUC against the model's **bigram coverage** (see ``NgramLM.bigram_coverage``):

==========  =====  ==========================  ======================  =============
coverage    mean   std                         sentence std            cv
==========  =====  ==========================  ======================  =============
0.41-0.51   0.90   0.20-0.38 (**inverted**)    ~0.45 (chance)          0.11 (inverted)
0.68-0.70   0.97   0.89-0.93                   ~0.54                   0.13 (inverted)
0.77-0.79   0.99   0.97-0.99                   ~0.64                   0.19-0.29
0.85-0.87   0.99   0.99-1.00                   0.65-0.72               0.40-0.51
0.92-0.99   1.00   1.00                        0.76-0.84               0.79-0.99
==========  =====  ==========================  ======================  =============

Three consequences, all implemented below:

1. **The mean is the robust signal**, strong at every coverage tested. This corrects
   the project's original thesis, which held that spread would beat the level.
2. **Spread measures are only valid on a well-estimated model**, and below that they
   do not merely weaken — they invert, because backoff noise dominates the spread of
   *both* classes. They are therefore gated on corpus-level coverage, with thresholds
   read off this table. The gate is corpus-level on purpose: a per-document gate
   would withhold spread signals more often from human documents (which use rarer
   transitions) than from synthetic ones, making missingness itself correlate with
   the label.
3. **Coefficient of variation was dropped.** It is ``std / mean``, so it carries no
   information the other two lack, and it has the worst failure mode of the set —
   inverted across most of the coverage range.

The toy language has no topic structure, so this experiment cannot test the popular
claim that human text is bursty *because of* topic shifts and idiosyncratic word
choice. Only real data can; that is a phase 6 measurement, not an assumption here.

This extractor is contextual: it cannot run on a single document in isolation, because
surprisal only means something relative to a model. It is registered as a factory that
receives the :class:`~chaff.context.CorpusContext` after pass 1.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..context import CorpusContext
from ..document import Document
from ..stats import mean, stdev
from ..tokenization import TextView, tokenize_words
from . import FAMILY_SURPRISAL, Extractor, Signal, register_contextual

#: A token is "highly predictable" when the model gave it more than even odds.
PREDICTABLE_BITS = 1.0

#: A run of predictable tokens must be at least this long to count. Short runs are
#: ordinary — "of the", "in order to". Long ones are recycled spans: phrasing that
#: this document shares verbatim with other documents in the corpus, which the
#: leave-one-out model can only predict because *someone else* wrote it too.
MIN_PREDICTABLE_RUN = 5

#: Sentence-level spread needs enough sentences to have a spread.
MIN_SENTENCES = 4
MIN_SENTENCE_WORDS = 3

#: Minimum corpus bigram coverage before token-level surprisal spread is emitted.
#: Measured: AUC 0.20-0.38 (inverted) at <= 0.51, 0.89-0.93 at 0.68-0.70.
MIN_COVERAGE_TOKEN_SPREAD = 0.70

#: Minimum coverage for sentence-level spread, which needs a better model still:
#: chance up to ~0.79, 0.65-0.72 at 0.85, 0.76-0.84 above 0.92.
MIN_COVERAGE_SENTENCE_SPREAD = 0.85


def predictable_run_fraction(bits: Sequence[float]) -> float:
    """Share of tokens inside runs of at least :data:`MIN_PREDICTABLE_RUN` consecutive
    highly predictable tokens."""
    if not bits:
        return 0.0
    covered = 0
    run = 0
    for value in list(bits) + [float("inf")]:  # sentinel flushes the final run
        if value < PREDICTABLE_BITS:
            run += 1
            continue
        if run >= MIN_PREDICTABLE_RUN:
            covered += run
        run = 0
    return covered / float(len(bits))


def sentence_means(view: TextView, bits: Sequence[float]) -> Optional[List[float]]:
    """Mean surprisal of each sentence, aligned to the document's word stream.

    Alignment is by re-tokenizing each sentence with the same word tokenizer that
    built ``view.words``. If the per-sentence word counts do not add up to the
    document's word count — the sentence splitter and the word tokenizer disagreed
    somewhere — this returns ``None`` rather than guessing at a misaligned split.
    """
    counts = [len(tokenize_words(sentence)) for sentence in view.sentences]
    if sum(counts) != len(bits):
        return None
    out: List[float] = []
    cursor = 0
    for n in counts:
        if n >= MIN_SENTENCE_WORDS:
            out.append(mean(bits[cursor : cursor + n]))
        cursor += n
    return out if len(out) >= MIN_SENTENCES else None


def surprisal_signals(
    view: TextView,
    bits: Sequence[float],
    *,
    coverage: float,
) -> List[Signal]:
    """Turn a per-word surprisal sequence into signals.

    ``coverage`` is the model's corpus-level bigram coverage. Spread signals are
    withheld below their measured thresholds — see the module docstring for why a
    spread measured on an under-trained model is worse than no measurement. It is
    deliberately required: a permissive default would let a caller emit spread
    signals from a model nobody measured.

    Split out from the extractor so it can be tested against hand-built surprisal
    sequences without building a model.
    """
    if not bits:
        return []

    signals = [
        Signal(
            name="mean_surprisal",
            value=mean(bits),
            family=FAMILY_SURPRISAL,
            direction=-1,
            description="Mean per-word surprisal in bits (log-perplexity). Lower means more predictable given the rest of the corpus.",
        ),
        Signal(
            name="predictable_run_fraction",
            value=predictable_run_fraction(bits),
            family=FAMILY_SURPRISAL,
            direction=1,
            description="Share of words inside long runs the model predicts at better than even odds. Higher means recycled spans.",
        ),
    ]

    if coverage >= MIN_COVERAGE_TOKEN_SPREAD:
        signals.append(
            Signal(
                name="surprisal_std",
                value=stdev(bits),
                family=FAMILY_SURPRISAL,
                direction=-1,
                description="Standard deviation of per-word surprisal, in bits. Lower means flatter, less bursty text.",
            )
        )

    if coverage >= MIN_COVERAGE_SENTENCE_SPREAD:
        per_sentence = sentence_means(view, bits)
        if per_sentence is not None:
            signals.append(
                Signal(
                    name="sentence_surprisal_std",
                    value=stdev(per_sentence),
                    family=FAMILY_SURPRISAL,
                    direction=-1,
                    description="Spread of mean surprisal across sentences. Lower means every sentence is equally routine.",
                )
            )
    return signals


@register_contextual("surprisal_profile", FAMILY_SURPRISAL)
def surprisal_profile_factory(context: CorpusContext) -> Optional[Extractor]:
    """Bind the corpus LM into a per-document extractor, or opt out without one."""
    if context.lm is None:
        return None
    lm = context.lm
    loo = context.leave_one_out
    # Read once at bind time: coverage is a property of the corpus, identical for
    # every document, which is what keeps the spread gate label-neutral.
    coverage = lm.bigram_coverage

    def surprisal_profile(doc: Document, view: TextView) -> List[Signal]:
        return surprisal_signals(view, lm.surprisals(view.words, loo=loo), coverage=coverage)

    return surprisal_profile
