"""Two-pass orchestration: when pass 1 runs, what it builds, and how it fails."""

from __future__ import annotations

import io
import json
import os
import random

import pytest

from chaff.context import SOURCE_REFERENCE, SOURCE_SELF
from chaff.lm import StreamMismatchError
from chaff.metrics import FAMILY_DISTRIBUTIONAL
from chaff.pipeline import build_context, profile_corpus


def _corpus(tmp_path, name="c.jsonl", n=8, seed=0):
    rng = random.Random(seed)
    vocab = "the a of cat dog ran sat fast slow red blue green tree house river stone".split()
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            text = ". ".join(" ".join(rng.choice(vocab) for _ in range(12)) for _ in range(8)) + "."
            fh.write(json.dumps({"id": "{0}-{1}".format(name, i), "text": text}) + "\n")
    return path


def test_contextual_family_triggers_a_first_pass(tmp_path):
    profile = profile_corpus(_corpus(tmp_path))
    assert profile.context_source == SOURCE_SELF
    assert profile.lm_summary["documents"] == 8
    assert "surprisal_profile" in profile.metrics_active
    assert any(s.name == "mean_surprisal" for s in profile.rows[0].signals)


def test_no_contextual_family_means_no_first_pass(tmp_path):
    profile = profile_corpus(_corpus(tmp_path), families=[FAMILY_DISTRIBUTIONAL])
    assert profile.lm_summary == {}
    assert profile.context_source is None
    assert "surprisal_profile" not in profile.metrics_active


def test_stdin_skips_surprisal_with_an_explicit_note(tmp_path, monkeypatch):
    with open(_corpus(tmp_path), encoding="utf-8") as fh:
        monkeypatch.setattr("sys.stdin", io.StringIO(fh.read()))
    profile = profile_corpus("-")
    assert profile.n_documents == 8
    assert "surprisal_profile" not in profile.metrics_active
    assert any("stdin" in note for note in profile.notes)


def test_reference_mode_scores_documents_outside_the_model(tmp_path):
    reference = _corpus(tmp_path, "ref.jsonl", seed=1)
    audited = _corpus(tmp_path, "audit.jsonl", seed=2)
    profile = profile_corpus(audited, reference=reference)
    assert profile.context_source == SOURCE_REFERENCE
    assert all(any(s.name == "mean_surprisal" for s in r.signals) for r in profile.rows)


def test_reference_model_is_not_pruned(tmp_path):
    # Pruning is only lossless under leave-one-out; the reference model scores
    # documents it never saw.
    context = build_context(_corpus(tmp_path), source=SOURCE_REFERENCE)
    assert context.lm.pruned is False


def test_passes_that_disagree_about_the_corpus_fail_loudly(tmp_path):
    path = _corpus(tmp_path, n=8)
    stale = build_context(path, limit=5)
    with pytest.raises(StreamMismatchError):
        profile_corpus(path, context=stale)
