"""The distributional family (phase 2).

Two kinds of test here, and the distinction matters:

* **Behaviour tests** pin down known values and guard conditions.
* **Discrimination tests** assert that a signal actually separates a rich-tailed corpus
  from a truncated one. These use controlled Zipfian corpora rather than the sample
  documents in ``data/samples/``, because those documents are ~150 words — far below
  the length at which most of these metrics carry information. Asserting
  discriminative power on data that cannot support it would be the exact
  self-deception this project is built to avoid.
"""

from __future__ import annotations

import random

import pytest

from chaff.document import Document
from chaff.metrics import FAMILIES, Signal
from chaff.metrics.entropy import (
    MIN_BRANCHING_CONTEXTS,
    conditional_bigram_entropy,
    hapax_ratio,
    lexical_profile,
    mtld,
    yules_k,
)
from chaff.metrics.ngram import (
    MIN_COMPRESSION_CHARS,
    compression_ratio,
    ngram_repetition,
    repetition_rate,
)
from chaff.metrics.zipf import (
    MIN_SPECTRUM_WORDS,
    heaps_beta,
    spectrum_slope,
    zipf_profile,
    zipf_slope,
)
from chaff.tokenization import build_view, tokenize

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _word(i: int) -> str:
    """Alphabetic vocabulary. Digits are deliberately excluded from word-hood by the
    tokenizer, so a ``w0``-style vocabulary would collapse to a single type."""
    out, i = "", i + 1
    while i:
        out, i = _ALPHABET[i % 26] + out, i // 26
    return out


def _zipfian(n_tokens, n_types=3000, exponent=1.0, seed=1):
    rng = random.Random(seed)
    weights = [1.0 / (r ** exponent) for r in range(1, n_types + 1)]
    return rng.choices([_word(i) for i in range(n_types)], weights=weights, k=n_tokens)


# --------------------------------------------------------------- lexical diversity

def test_yules_k_is_zero_when_no_word_repeats():
    assert yules_k([_word(i) for i in range(200)]) == pytest.approx(0.0)


def test_yules_k_is_large_when_everything_repeats():
    assert yules_k(["same"] * 200) > 9000


def test_yules_k_is_broadly_length_independent():
    # This is the whole reason Yule's K is here instead of raw TTR.
    base = _zipfian(2000, seed=7)
    doubled = base + _zipfian(2000, seed=8)
    assert yules_k(doubled) == pytest.approx(yules_k(base), rel=0.35)


def test_mtld_orders_diverse_above_repetitive():
    diverse = [_word(i) for i in range(200)]
    cyclic = ["a", "b", "c", "d"] * 50
    identical = ["a"] * 200
    assert mtld(diverse) > mtld(cyclic) > mtld(identical)


def test_mtld_falls_back_to_token_count_when_never_below_threshold():
    words = [_word(i) for i in range(120)]
    assert mtld(words) == pytest.approx(120.0)


def test_mtld_is_direction_symmetric():
    # Bidirectional averaging exists so that where repetition sits in a document
    # does not change the score.
    words = ["a"] * 40 + [_word(i) for i in range(80)]
    assert mtld(words) == pytest.approx(mtld(list(reversed(words))))


def test_hapax_ratio_bounds():
    assert hapax_ratio([_word(i) for i in range(50)]) == pytest.approx(1.0)
    assert hapax_ratio(["a"] * 50) == pytest.approx(0.0)
    assert hapax_ratio([]) == 0.0


# ----------------------------------------------------------- conditional entropy

def test_branching_entropy_is_none_without_enough_repeated_contexts():
    # Every context occurs once: there was never a choice to observe.
    assert conditional_bigram_entropy([_word(i) for i in range(200)]) is None


def test_branching_entropy_none_and_zero_mean_different_things():
    """A measured 0.0 is the strongest templated-text evidence there is; ``None`` is
    the absence of evidence. Collapsing them would discard the signal exactly where
    it is most informative."""
    # A closed cycle: every word has exactly one possible successor, so there are
    # plenty of repeated contexts and zero branching at all of them.
    cycle = [_word(i) for i in range(MIN_BRANCHING_CONTEXTS + 3)] * 10
    measured = conditional_bigram_entropy(cycle)
    assert measured is not None
    assert measured == pytest.approx(0.0)


def test_branching_entropy_is_higher_for_genuinely_branching_text():
    rng = random.Random(3)
    vocab = [_word(i) for i in range(10)]
    branching = [rng.choice(vocab) for _ in range(600)]
    templated = ["the", "system", "is", "important", "and", "so"] * 100
    assert conditional_bigram_entropy(branching) > conditional_bigram_entropy(templated)


# ------------------------------------------------------------------ zipf / heaps

def test_zipf_slope_recovers_exponent_ordering():
    measured = [zipf_slope(_zipfian(20000, exponent=e, seed=4))[0] for e in (0.8, 1.0, 1.2, 1.5)]
    assert measured == sorted(measured, reverse=True), measured


def test_zipf_slope_is_none_below_the_type_threshold():
    assert zipf_slope(["a", "b", "c"] * 5) is None


def test_spectrum_slope_flattens_as_the_tail_is_truncated():
    """The core Zipf-violation signal: thinning the rare-word tail raises the
    frequency-spectrum exponent toward and past zero."""
    rich = spectrum_slope(_zipfian(20000, n_types=3000, seed=5))[0]
    thin = spectrum_slope(_zipfian(20000, n_types=1200, seed=5))[0]
    assert thin > rich


def test_spectrum_slope_is_withheld_on_short_documents():
    short = " ".join(_zipfian(MIN_SPECTRUM_WORDS // 4, seed=6))
    emitted = {s.name for s in zipf_profile(Document("d", short), build_view(short))}
    assert "spectrum_slope" not in emitted


def test_heaps_beta_is_higher_when_new_words_keep_arriving():
    rich = heaps_beta(_zipfian(20000, n_types=3000, seed=9))
    thin = heaps_beta(_zipfian(20000, n_types=300, seed=9))
    assert rich > thin


def test_heaps_beta_is_none_below_the_word_threshold():
    assert heaps_beta(["a"] * 40) is None


# ----------------------------------------------------------- n-gram / compression

def test_repetition_rate_is_zero_when_every_ngram_is_unique():
    assert repetition_rate([_word(i) for i in range(200)], 4) == pytest.approx(0.0)


def test_repetition_rate_is_high_for_a_looped_clause():
    assert repetition_rate(tokenize("It is important to note that. " * 30), 4) > 0.9


def test_repetition_rate_is_none_when_too_short_to_mean_anything():
    assert repetition_rate(["a", "b", "c"], 4) is None


def test_compression_ratio_is_lower_for_redundant_text():
    looped = "It is important to note that this is important. " * 30
    varied = " ".join(_zipfian(400, seed=11))
    assert compression_ratio(looped) < compression_ratio(varied)


def test_compression_ratio_is_none_below_the_warmup_threshold():
    assert compression_ratio("x" * (MIN_COMPRESSION_CHARS - 1)) is None


# ---------------------------------------------------------------- contract tests

def _all_signals(text):
    doc = Document("d", text)
    view = build_view(text)
    return list(lexical_profile(doc, view)) + list(zipf_profile(doc, view)) + list(
        ngram_repetition(doc, view)
    )


def test_every_emitted_signal_satisfies_the_contract():
    for signal in _all_signals(" ".join(_zipfian(5000, seed=12))):
        assert isinstance(signal, Signal)
        assert signal.family in FAMILIES
        assert signal.direction in (1, -1)
        assert signal.description, "{0} has no user-facing description".format(signal.name)
        assert signal.value == signal.value, "{0} emitted NaN".format(signal.name)


def test_signal_names_are_unique_within_a_document():
    names = [s.name for s in _all_signals(" ".join(_zipfian(5000, seed=13)))]
    assert len(names) == len(set(names))


def test_extractors_do_not_mutate_the_shared_view():
    """Every family reads one shared TextView; mutating it would corrupt the families
    that run after. There is no enforcement, so it is asserted."""
    text = " ".join(_zipfian(3000, seed=14))
    view = build_view(text)
    before = (list(view.words), list(view.tokens), list(view.sentences), view.raw)
    doc = Document("d", text)
    for extractor in (lexical_profile, zipf_profile, ngram_repetition):
        extractor(doc, view)
    assert (view.words, view.tokens, view.sentences, view.raw) == before


def test_extractors_are_safe_on_degenerate_input():
    for text in ("", "   ", "a", "a b c", "!!!", "\n\n\n"):
        doc = Document("d", text)
        view = build_view(text)
        for extractor in (lexical_profile, zipf_profile, ngram_repetition):
            assert isinstance(extractor(doc, view), list)
