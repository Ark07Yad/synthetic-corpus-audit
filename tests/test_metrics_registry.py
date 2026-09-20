"""The registry contract that phases 2-4 build on."""

from __future__ import annotations

import pytest

from chaff.document import Document
from chaff.metrics import (
    FAMILY_ARTIFACT,
    FAMILY_DISTRIBUTIONAL,
    Signal,
    clear_registry,
    extract_signals,
    register,
    registered,
)
from chaff.tokenization import build_view


@pytest.fixture(autouse=True)
def clean_registry():
    clear_registry()
    yield
    clear_registry()


def _doc(text="word " * 80):
    return Document(doc_id="d", text=text)


def test_signal_rejects_unknown_family():
    with pytest.raises(ValueError):
        Signal(name="x", value=1.0, family="nonsense", direction=1)


def test_signal_rejects_ambiguous_direction():
    # Direction must be declared so fusion never guesses a metric's polarity.
    with pytest.raises(ValueError):
        Signal(name="x", value=1.0, family=FAMILY_DISTRIBUTIONAL, direction=0)


def test_registered_extractor_runs_and_returns_signals():
    @register("dummy_entropy", FAMILY_DISTRIBUTIONAL)
    def _extract(doc, view):
        return [Signal("dummy_entropy", float(view.n_words), FAMILY_DISTRIBUTIONAL, -1)]

    doc = _doc()
    signals = extract_signals(doc, build_view(doc.text))
    assert [s.name for s in signals] == ["dummy_entropy"]
    assert signals[0].value == 80.0


def test_family_filter_selects_only_requested_families():
    @register("dist_one", FAMILY_DISTRIBUTIONAL)
    def _dist(doc, view):
        return [Signal("dist_one", 1.0, FAMILY_DISTRIBUTIONAL, 1)]

    @register("art_one", FAMILY_ARTIFACT)
    def _art(doc, view):
        return [Signal("art_one", 2.0, FAMILY_ARTIFACT, 1)]

    doc = _doc()
    view = build_view(doc.text)
    names = [s.name for s in extract_signals(doc, view, families=[FAMILY_ARTIFACT])]
    assert names == ["art_one"]
    assert len(extract_signals(doc, view)) == 2


def test_duplicate_registration_is_rejected():
    @register("dupe", FAMILY_DISTRIBUTIONAL)
    def _one(doc, view):
        return []

    with pytest.raises(ValueError):
        @register("dupe", FAMILY_DISTRIBUTIONAL)
        def _two(doc, view):
            return []


def test_registered_listing_is_sorted():
    for name in ("zeta", "alpha", "mid"):
        register(name, FAMILY_DISTRIBUTIONAL)(lambda doc, view: [])
    assert [n for n, _ in registered()] == ["alpha", "mid", "zeta"]
