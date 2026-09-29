"""End-to-end scoring: passes, spill, fusion, outputs and CLI."""

from __future__ import annotations

import json
import os
import random

import pytest

from chaff.cli import main
from chaff.pipeline import CorpusProfile, coverage_drops, score_corpus
from chaff.report import render_html, render_json, render_markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "..", "data", "samples", "mixed_sample.jsonl")


def _corpus(tmp_path, n=12, seed=0, name="c.jsonl", label=None):
    rng = random.Random(seed)
    vocab = "the a of cat dog ran sat fast slow red blue green tree house river stone".split()
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            text = ". ".join(" ".join(rng.choice(vocab) for _ in range(12)) for _ in range(8)) + "."
            rec = {"id": "{0}-{1}".format(name, i), "text": text}
            if label:
                rec["label"] = label
            fh.write(json.dumps(rec) + "\n")
    return path


def test_scores_file_has_one_line_per_document_and_a_meta_sidecar(tmp_path):
    out = os.path.join(str(tmp_path), "scores.jsonl")
    run = score_corpus(_corpus(tmp_path), scores_path=out)
    lines = [json.loads(l) for l in open(out)]
    assert len(lines) == 12
    meta = json.load(open(out + ".meta.json"))
    assert sum(meta["tiers"].values()) == meta["documents"] == 12
    assert [d.score for d in run.top] == sorted((d.score for d in run.top), reverse=True)


def test_small_reference_is_flagged(tmp_path):
    run = score_corpus(_corpus(tmp_path, n=8))
    assert any("tiny sample" in n for n in run.meta["notes"])


def test_labels_are_reported_but_only_as_a_breakdown(tmp_path):
    run = score_corpus(_corpus(tmp_path, label="human"))
    assert sum(run.meta["labels"]["human"].values()) == 12


def test_reference_mode_normalises_against_the_reference(tmp_path):
    ref = _corpus(tmp_path, n=20, seed=1, name="ref.jsonl")
    run = score_corpus(_corpus(tmp_path, seed=2, name="aud.jsonl"), reference=ref)
    assert run.meta["normalisation"] == "reference corpus"
    assert run.meta["profile"]["audited_coverage"] is not None


@pytest.mark.parametrize("coverage,dropped", [
    (None, set()),
    (0.5, {"surprisal_std", "sentence_surprisal_std"}),
    (0.75, {"sentence_surprisal_std"}),
    (0.95, set()),
])
def test_r4_audited_coverage_gates_spread_signals(coverage, dropped):
    profile = CorpusProfile(path="x")
    profile.audited_coverage = coverage
    got, note = coverage_drops(profile)
    assert got == dropped
    assert (note is None) == (not dropped)


def test_sample_corpus_tiers(tmp_path):
    """Mechanism, not accuracy: ten hand-written documents. The adversarial human cases
    must stay CLEAN and the looping reasoning document must corroborate."""
    out = os.path.join(str(tmp_path), "s.jsonl")
    score_corpus(SAMPLE, scores_path=out)
    tiers = {r["doc_id"]: r["tier"] for r in map(json.loads, open(out))}
    assert tiers["h_changelog_01"] == tiers["h_legal_01"] == "CLEAN"
    assert tiers["s_reasoning_01"] == "LIKELY_SYNTHETIC"
    assert all(t == "CLEAN" for d, t in tiers.items() if d.startswith("h_"))


def test_cli_score_explain_report_round_trip(tmp_path, capsys):
    out = os.path.join(str(tmp_path), "s.jsonl")
    assert main(["score", SAMPLE, "--out", out]) == 0
    assert main(["explain", out, "s_reasoning_01"]) == 0
    assert "LIKELY_SYNTHETIC" in capsys.readouterr().out
    for fmt in ("md", "json", "html"):
        target = os.path.join(str(tmp_path), "r." + fmt)
        assert main(["report", out, "--format", fmt, "--output", target]) == 0
        assert os.path.getsize(target) > 500


def test_explain_unknown_document_fails_cleanly(tmp_path):
    out = os.path.join(str(tmp_path), "s.jsonl")
    main(["score", SAMPLE, "--out", out])
    assert main(["explain", out, "no-such-doc"]) == 1


def test_html_report_is_self_contained(tmp_path):
    out = os.path.join(str(tmp_path), "s.jsonl")
    run = score_corpus(SAMPLE, scores_path=out)
    page = render_html(run.meta, [d.to_dict() for d in run.top])
    assert "<script" not in page and "http://" not in page and "https://" not in page
    assert "prefers-color-scheme" in page


def test_every_report_carries_its_caveats(tmp_path):
    run = score_corpus(SAMPLE)
    top = [d.to_dict() for d in run.top]
    for text in (render_markdown(run.meta, top), render_json(run.meta, top), render_html(run.meta, top)):
        assert "triage instrument" in text and "True-positive rates are not yet measured" in text
