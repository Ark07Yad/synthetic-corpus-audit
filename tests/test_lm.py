"""The corpus-internal language model.

The properties asserted here are the ones every surprisal signal silently depends on:
probabilities that genuinely sum to one, leave-one-out scoring that is *exactly* the
same as removing the document, and pruning that changes nothing it should not.
"""

from __future__ import annotations

import random

import pytest

from chaff.lm import BOS, NgramLM, StreamMismatchError

VOCAB = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta"]


def _docs(seed=0, n=12):
    rng = random.Random(seed)
    return [[rng.choice(VOCAB) for _ in range(rng.randint(20, 60))] for _ in range(n)]


def _model(docs, prune):
    lm = NgramLM()
    for d in docs:
        lm.observe(d)
    lm.finalize(prune=prune)
    return lm


@pytest.mark.parametrize("history", [("alpha", "beta"), (BOS, BOS), ("zeta", "never-seen")])
def test_probabilities_sum_to_one_including_the_oov_bucket(history):
    lm = _model(_docs(), prune=False)
    total = sum(lm._prob(history + (w,), None) for w in VOCAB)
    total += lm._prob(history + ("__oov__",), None)
    assert total == pytest.approx(1.0, abs=1e-12)


def test_probabilities_sum_to_one_under_leave_one_out():
    """Checked on the unpruned model on purpose. Normalisation is a property of the
    full conditional distribution, which means asking about n-grams the document does
    *not* contain — including singletons from other documents, which pruning deletes
    and cannot reconstruct. Scoring never asks those questions (see the next test),
    so the pruned model is exact for everything it is used for, but it cannot answer
    this one."""
    docs = _docs()
    lm = _model(docs, prune=False)
    local = lm._local_counts(docs[3])
    for history in (("alpha", "beta"), (BOS, BOS)):
        total = sum(lm._prob(history + (w,), local) for w in VOCAB)
        total += lm._prob(history + ("__oov__",), local)
        assert total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("prune", [False, True])
def test_leave_one_out_equals_physically_removing_the_document(prune):
    """The defining property. Not approximately equal: identical."""
    docs = _docs()
    lm = _model(docs, prune=prune)
    for i, doc in enumerate(docs):
        held_out = _model([d for j, d in enumerate(docs) if j != i], prune=False)
        assert lm.surprisals(doc, loo=True) == pytest.approx(
            held_out.surprisals(doc, loo=False), abs=1e-12
        )


def test_a_document_does_not_get_to_predict_itself():
    """Why leave-one-out exists: a phrase that only this document uses must look
    surprising, not memorised."""
    docs = _docs()
    unique_phrase = ["zyx", "wvu", "tsr", "qpo"] * 5
    docs.append(unique_phrase + docs[0])
    lm = _model(docs, prune=True)
    with_self = _model(docs, prune=False).surprisals(docs[-1], loo=False)
    without_self = lm.surprisals(docs[-1], loo=True)
    phrase_len = len(unique_phrase)
    assert sum(without_self[:phrase_len]) > 3 * sum(with_self[:phrase_len])


def test_pruning_removes_most_entries_on_zipfian_text():
    rng = random.Random(1)
    vocab = ["w" + chr(97 + i % 26) + chr(97 + i // 26 % 26) for i in range(600)]
    weights = [1.0 / r for r in range(1, 601)]
    docs = [rng.choices(vocab, weights=weights, k=300) for _ in range(40)]
    assert _model(docs, prune=True).summary()["pruned_share"] > 0.5


def test_pruned_model_refuses_to_score_documents_outside_training():
    lm = _model(_docs(), prune=True)
    with pytest.raises(ValueError):
        lm.surprisals(["alpha", "beta"], loo=False)


def test_scoring_an_untrained_document_leave_one_out_raises():
    """Pass 1 and pass 2 disagreeing about the corpus must fail loudly."""
    lm = _model(_docs(), prune=True)
    with pytest.raises(StreamMismatchError):
        lm.surprisals(["alpha", "beta", "gamma", "delta"] * 10, loo=True)


def test_surprisal_per_word_and_empty_input():
    docs = _docs()
    lm = _model(docs, prune=True)
    assert len(lm.surprisals(docs[0], loo=True)) == len(docs[0])
    assert lm.surprisals([], loo=True) == []


def test_lifecycle_is_enforced():
    lm = NgramLM()
    lm.observe(["a", "b"])
    with pytest.raises(RuntimeError):
        lm.surprisals(["a"], loo=True)
    lm.finalize()
    with pytest.raises(RuntimeError):
        lm.observe(["c"])


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_discount_must_lie_strictly_inside_the_unit_interval(bad):
    with pytest.raises(ValueError):
        NgramLM(discount=bad)


def test_bigram_coverage_is_unchanged_by_pruning():
    docs = _docs()
    assert _model(docs, prune=True).bigram_coverage == pytest.approx(
        _model(docs, prune=False).bigram_coverage
    )


def test_bigram_coverage_matches_a_manual_count():
    docs = [["a", "b", "a", "b"], ["a", "b", "c"]]
    lm = _model(docs, prune=False)
    # Bigram tokens (7 words, BOS-prefixed): (<s>,a) x2 [seen twice], (a,b) x3,
    # (b,a) x1, (b,c) x1. Repeated types cover 2 + 3 = 5 of 7 positions.
    assert lm.bigram_coverage == pytest.approx(5 / 7)


def test_bigram_coverage_of_an_empty_model_is_zero():
    lm = NgramLM()
    lm.finalize()
    assert lm.bigram_coverage == 0.0
