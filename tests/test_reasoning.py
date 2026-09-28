"""The reasoning family (phase 4)."""

from __future__ import annotations

import pytest

from chaff.document import Document
from chaff.metrics.reasoning import (
    MIN_STEPS,
    STALL_THRESHOLD,
    content_of,
    reasoning_profile,
    stem,
    steps_of,
    windowed_novelty,
)
from chaff.tokenization import build_view

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _word(i):
    out, i = "", i + 1
    while i:
        out, i = _ALPHABET[i % 26] + out, i // 26
    return "q" + out  # prefix keeps generated words clear of the stopword list


def _chain(n_steps, new_per_step, recycled_per_step=0, words_per_step=6):
    """A chain where each step introduces ``new_per_step`` never-seen content words and
    re-uses ``recycled_per_step`` from the previous step. The first step is seeded at
    full width, so early steps are not artificially short (and so artificially novel)."""
    fresh = iter(range(10_000))
    steps, previous = [], []
    for i in range(n_steps):
        width = words_per_step if i == 0 else new_per_step
        new = [_word(next(fresh)) for _ in range(width)]
        reused = previous[:recycled_per_step] if i > 0 else []
        words = (new + reused)[:words_per_step]
        steps.append(" ".join(words).capitalize() + ".")
        previous = words
    return " ".join(steps)


def _signals(text):
    return {s.name: s.value for s in reasoning_profile(Document("d", text), build_view(text))}


def test_stem_strips_one_suffix_and_keeps_three_letters():
    assert stem("costs") == "cost"
    assert stem("costing") == "cost"
    assert stem("bus") == "bus"      # would fall below three letters
    assert stem("was") == "was"


def test_possessives_do_not_create_new_content():
    # Regression: "component's" once stemmed to "component'", a spurious new word.
    assert stem("component's") == "component"
    assert stem("component\u2019s") == "component"
    assert stem("users'") == "user"


def test_content_drops_function_words_and_merges_inflections():
    assert content_of("The costs of the component, and the component's cost.") == {"cost", "component"}


def test_novelty_is_one_when_every_step_is_new():
    view = build_view(_chain(8, new_per_step=6))
    assert all(n == pytest.approx(1.0) for n in windowed_novelty(steps_of(view)))


def test_novelty_falls_monotonically_as_steps_recycle_content():
    """Detector validity: the measured novelty tracks the constructed novelty."""
    measured = [_signals(_chain(10, new_per_step=6 - r, recycled_per_step=r))["step_novelty"]
                for r in range(0, 6)]
    assert measured == sorted(measured, reverse=True), measured
    assert measured[0] == pytest.approx(1.0)


def test_a_looping_chain_is_mostly_stalled():
    signals = _signals(_chain(10, new_per_step=1, recycled_per_step=5))
    assert signals["stalled_step_ratio"] > 0.8
    assert signals["step_novelty"] < STALL_THRESHOLD


def test_novelty_does_not_penalise_long_documents_for_being_long():
    """The reason novelty is windowed: measured against *all* prior steps it would fall
    with document length alone (Heaps' law). A long chain of genuinely new steps must
    score like a short one."""
    short = _signals(_chain(6, new_per_step=5, recycled_per_step=1))["step_novelty"]
    long = _signals(_chain(60, new_per_step=5, recycled_per_step=1))["step_novelty"]
    assert long == pytest.approx(short, abs=0.02)


def test_withheld_below_the_minimum_number_of_steps():
    assert reasoning_profile(Document("d", _chain(MIN_STEPS - 1, 6)),
                             build_view(_chain(MIN_STEPS - 1, 6))) == []


def test_fragments_do_not_count_as_steps():
    text = "Step 1: " + " Step 2: ".join(["x."] * 8)
    assert steps_of(build_view(text)) == []


def test_restatement_was_removed():
    """Dropped after it correlated at |rho| up to 0.74 with the distributional family on
    the human baseline — it was re-measuring lexical repetition, which would let one
    repetitive human document count as corroborated by two families."""
    assert set(_signals(_chain(10, 3, 3))) == {"step_novelty", "stalled_step_ratio"}
