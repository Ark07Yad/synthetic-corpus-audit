"""The surprisal family (phase 3).

As in phase 2, discrimination is tested on controlled corpora rather than on the
150-word sample documents, which are far too small to train a meaningful model. The
controlled corpus is a random Markov-chain language where "human" documents sample
the full distribution and "synthetic" ones sample through nucleus truncation — same
language, same sentence lengths, only the sampler differs.
"""

from __future__ import annotations

import random

import pytest

from chaff.context import SOURCE_SELF, CorpusContext
from chaff.document import Document
from chaff.lm import NgramLM
from chaff.metrics.surprisal import (
    MIN_COVERAGE_SENTENCE_SPREAD,
    MIN_COVERAGE_TOKEN_SPREAD,
    MIN_PREDICTABLE_RUN,
    predictable_run_fraction,
    sentence_means,
    surprisal_profile_factory,
    surprisal_signals,
)
from chaff.tokenization import build_view

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _word(i):
    out, i = "", i + 1
    while i:
        out, i = _ALPHABET[i % 26] + out, i // 26
    return out


def _language(rng, n_vocab, support, exponent):
    vocab = [_word(i) for i in range(n_vocab)]
    chain = {}
    for w in vocab:
        weights = [1.0 / (r ** exponent) for r in range(1, support + 1)]
        total = sum(weights)
        chain[w] = (rng.sample(vocab, support), [x / total for x in weights])
    return vocab, chain


def _document(rng, vocab, chain, n_words, top_p=None):
    words, sentences, w = [], [], rng.choice(vocab)
    while len(words) < n_words:
        sentence = []
        for _ in range(rng.randint(8, 20)):
            successors, probs = chain[w]
            if top_p is not None:
                kept, acc = [], 0.0
                for s, p in zip(successors, probs):
                    kept.append((s, p))
                    acc += p
                    if acc >= top_p:
                        break
                successors, probs = [s for s, _ in kept], [p for _, p in kept]
            w = rng.choices(successors, weights=probs, k=1)[0]
            sentence.append(w)
        sentences.append(" ".join(sentence) + ".")
        words += sentence
    return " ".join(sentences)


def _auc(human, synthetic, direction):
    wins = sum((1.0 if (b - a) * direction > 0 else 0.5 if a == b else 0.0)
               for a in human for b in synthetic)
    return wins / (len(human) * len(synthetic))


def _controlled(seed, n_vocab, support, exponent, n_human=60, n_synth=30, n_words=300):
    rng = random.Random(seed)
    vocab, chain = _language(rng, n_vocab, support, exponent)
    texts = [(_document(rng, vocab, chain, n_words), "human") for _ in range(n_human)]
    texts += [(_document(rng, vocab, chain, n_words, top_p=0.9), "synthetic") for _ in range(n_synth)]
    views = [(build_view(t), label) for t, label in texts]
    lm = NgramLM()
    for view, _ in views:
        lm.observe(view.words)
    lm.finalize(prune=True)
    extractor = surprisal_profile_factory(CorpusContext(source=SOURCE_SELF, n_documents=len(views), lm=lm))
    per = {}
    for view, label in views:
        for s in extractor(Document("d", view.raw), view):
            per.setdefault(s.name, {"human": [], "synthetic": [], "dir": s.direction})[label].append(s.value)
    return lm, per


# ------------------------------------------------------------ predictable runs

def test_run_fraction_counts_only_runs_at_or_above_the_minimum():
    low, high = 0.1, 5.0
    exactly = [low] * MIN_PREDICTABLE_RUN + [high] * MIN_PREDICTABLE_RUN
    too_short = [low] * (MIN_PREDICTABLE_RUN - 1) + [high] * (MIN_PREDICTABLE_RUN + 1)
    assert predictable_run_fraction(exactly) == pytest.approx(0.5)
    assert predictable_run_fraction(too_short) == 0.0


def test_run_fraction_flushes_a_run_that_ends_the_document():
    assert predictable_run_fraction([5.0] + [0.1] * 9) == pytest.approx(0.9)


def test_run_fraction_of_empty_sequence_is_zero():
    assert predictable_run_fraction([]) == 0.0


# ------------------------------------------------------------ sentence alignment

def test_sentence_means_align_to_the_word_stream():
    text = " ".join("One two three four." for _ in range(4))
    view = build_view(text)
    bits = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0]
    assert sentence_means(view, bits) == [1.0, 2.0, 3.0, 4.0]


def test_sentence_means_refuse_a_misaligned_split():
    view = build_view("One two three four. Five six seven eight. A b c d. E f g h.")
    assert sentence_means(view, [1.0] * 3) is None


# ------------------------------------------------------------------ the gate

def _bits_and_view():
    text = " ".join("alpha beta gamma delta epsilon." for _ in range(6))
    return build_view(text), [float(i % 7) for i in range(30)]


@pytest.mark.parametrize("coverage,expected", [
    (MIN_COVERAGE_TOKEN_SPREAD - 0.01, {"mean_surprisal", "predictable_run_fraction"}),
    (MIN_COVERAGE_TOKEN_SPREAD, {"mean_surprisal", "predictable_run_fraction", "surprisal_std"}),
    (MIN_COVERAGE_SENTENCE_SPREAD, {"mean_surprisal", "predictable_run_fraction",
                                    "surprisal_std", "sentence_surprisal_std"}),
])
def test_spread_signals_are_gated_on_model_coverage(coverage, expected):
    view, bits = _bits_and_view()
    assert {s.name for s in surprisal_signals(view, bits, coverage=coverage)} == expected


def test_coverage_has_no_permissive_default():
    view, bits = _bits_and_view()
    with pytest.raises(TypeError):
        surprisal_signals(view, bits)


def test_coefficient_of_variation_is_not_emitted():
    """Dropped after measurement: it is std/mean, adds nothing independent, and was
    inverted across most of the coverage range."""
    view, bits = _bits_and_view()
    assert "surprisal_cv" not in {s.name for s in surprisal_signals(view, bits, coverage=1.0)}


# --------------------------------------------------------------- the factory

def test_factory_opts_out_without_a_language_model():
    assert surprisal_profile_factory(CorpusContext(source=SOURCE_SELF, n_documents=0)) is None


# ----------------------------------------------------------- discrimination

def test_well_estimated_model_separates_truncated_sampling():
    """On a language the model can estimate well, both the level and the spread of
    surprisal separate full sampling from nucleus-truncated sampling."""
    lm, per = _controlled(seed=1, n_vocab=300, support=25, exponent=1.5)
    assert lm.bigram_coverage >= MIN_COVERAGE_SENTENCE_SPREAD
    for name in ("mean_surprisal", "surprisal_std"):
        d = per[name]
        assert _auc(d["human"], d["synthetic"], d["dir"]) > 0.95, name


def test_sparse_model_withholds_the_spread_signals_that_would_invert():
    """The regression this gate exists for. On an under-trained model, surprisal_std
    measured AUC 0.20-0.38 — confidently backwards. It must not be emitted there, and
    the mean must still work."""
    lm, per = _controlled(seed=2, n_vocab=2000, support=150, exponent=1.1)
    assert lm.bigram_coverage < MIN_COVERAGE_TOKEN_SPREAD
    assert "surprisal_std" not in per
    assert "sentence_surprisal_std" not in per
    d = per["mean_surprisal"]
    assert _auc(d["human"], d["synthetic"], d["dir"]) > 0.8
