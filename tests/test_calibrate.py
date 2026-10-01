"""`chaff calibrate`: thresholds learned from a user's own trusted human corpus."""

from __future__ import annotations

import json
import os
import random

import pytest

from chaff.calibrate import (
    MIN_DOCUMENTS,
    CalibrationError,
    calibrate_rows,
    half_of,
)
from chaff.cli import main
from chaff.pipeline import score_corpus
from chaff.scoring import (
    TIER_LIKELY,
    TIER_SUSPECT,
    Calibration,
    Row,
    SignalInfo,
    build_reference_stats,
    load_calibration,
    score_row,
)

CATALOG = {
    "dist_a": SignalInfo("distributional", -1),
    "dist_b": SignalInfo("distributional", 1),
    "surp": SignalInfo("surprisal", -1),
    "reas": SignalInfo("reasoning", -1),
    "echo": SignalInfo("artifact", 1),
}


def _rows(n=1200, seed=0, prefix="d"):
    rng = random.Random(seed)
    return [Row(doc_id="{0}{1:05d}".format(prefix, i), n_words=rng.randint(100, 4000), analysable=True,
                signals={"dist_a": rng.gauss(5, 1), "dist_b": rng.gauss(5, 1),
                         "surp": rng.gauss(8, 1), "reas": rng.gauss(0.7, 0.1),
                         "echo": float(rng.random() < 0.1) * rng.random()})
            for i in range(n)]


def test_half_of_is_deterministic_and_balanced():
    ids = ["doc-{0}".format(i) for i in range(4000)]
    assert [half_of(d) for d in ids] == [half_of(d) for d in ids]
    share_a = sum(half_of(d) == "A" for d in ids) / len(ids)
    assert 0.45 < share_a < 0.55


def test_calibration_meets_its_target_and_records_the_held_out_half():
    run = calibrate_rows(_rows(), CATALOG, likely_target=0.01)
    cal, ev = run.calibration, run.calibration.meta["evaluation"]
    assert ev["A"]["likely_rate"] <= 0.01
    assert ev["A"]["documents"] + ev["B"]["documents"] == 1200
    assert set(cal.thresholds) == {"distributional", "surprisal", "reasoning", "artifact"}
    # two-sided families get a lower threshold too; the artifact family does not
    assert set(cal.thresholds_low) == {"distributional", "surprisal", "reasoning"}
    assert cal.score_null == sorted(cal.score_null)
    # the largest alpha on the grid that still meets the target
    assert cal.meta["alpha"] == max(a for a, _s, _l, ok in run.alpha_search if ok)


def test_held_out_rate_matches_what_scoring_produces():
    """The rates calibrate records for half B must be the rates `chaff score` produces on
    those same documents: same normalisation, same rounding, same strict thresholds."""
    rows = _rows()
    cal = calibrate_rows(rows, CATALOG).calibration
    stats = build_reference_stats(rows, CATALOG)
    b = [r for r in rows if half_of(r.doc_id) == "B"]
    tiers = [score_row(r, CATALOG, stats, cal).tier for r in b]
    recorded = cal.meta["evaluation"]["B"]
    assert tiers.count(TIER_LIKELY) / len(b) == pytest.approx(recorded["likely_rate"], abs=1e-5)
    assert tiers.count(TIER_SUSPECT) / len(b) == pytest.approx(recorded["suspect_rate"], abs=1e-5)


def test_a_calibration_survives_a_round_trip_through_json(tmp_path):
    cal = calibrate_rows(_rows(), CATALOG).calibration
    path = os.path.join(str(tmp_path), "cal.json")
    with open(path, "w") as fh:
        json.dump(cal.to_dict(), fh)
    again = load_calibration(path)
    assert again.thresholds == cal.thresholds and again.thresholds_low == cal.thresholds_low
    assert again.score_null == cal.score_null


def test_unanalysable_rows_are_left_out():
    rows = _rows() + [Row("short{0}".format(i), 20, False, {}) for i in range(300)]
    ev = calibrate_rows(rows, CATALOG).calibration.meta["evaluation"]
    assert ev["A"]["documents"] + ev["B"]["documents"] == 1200


def test_too_few_documents_is_refused():
    with pytest.raises(CalibrationError, match="at least {0}".format(MIN_DOCUMENTS)):
        calibrate_rows(_rows(MIN_DOCUMENTS - 1), CATALOG)


def test_small_corpus_is_calibrated_with_a_warning():
    run = calibrate_rows(_rows(400), CATALOG)
    assert any("fewer than the recommended" in n for n in run.notes)


def test_a_corpus_already_split_by_the_same_hash_is_refused():
    rows = [r for r in _rows(2000) if half_of(r.doc_id) == "A"]
    with pytest.raises(CalibrationError, match="fresh ids"):
        calibrate_rows(rows, CATALOG)


def test_an_unreachable_target_is_refused():
    with pytest.raises(CalibrationError, match="no alpha"):
        calibrate_rows(_rows(), CATALOG, likely_target=-1.0)


def test_groups_get_their_own_held_out_rates():
    rows = _rows()
    for i, r in enumerate(rows):
        r.group = "man" if i % 3 else "docs"
    ev = calibrate_rows(rows, CATALOG, stratify=True, stratify_by="genre").calibration.meta
    assert ev["stratify_by"] == "genre"
    assert {"B_man", "B_docs"} <= set(ev["evaluation"])
    assert ev["evaluation"]["B_man"]["documents"] + ev["evaluation"]["B_docs"]["documents"] \
        == ev["evaluation"]["B"]["documents"]


def test_scoring_against_a_calibration_built_differently_is_noted(tmp_path):
    cal = load_calibration()
    cal.meta = dict(cal.meta, stratify_by="genre")
    path = os.path.join(str(tmp_path), "c.jsonl")
    with open(path, "w") as fh:
        for i in range(10):
            fh.write(json.dumps({"id": str(i), "text": "the cat sat on the mat today. " * 12}) + "\n")
    notes = score_corpus(path, calibration=cal).meta["notes"]
    assert any("built with --stratify-by genre" in n and "unstratified" in n for n in notes)


def _human_corpus(tmp_path, n, name="trusted.jsonl"):
    rng = random.Random(5)
    vocab = ("the a of and to in is it that was for on are as with be by at this from or an but not "
             "river stone house garden window light morning evening letter road field market bridge "
             "walked carried opened closed watched wrote found kept turned").split()
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            text = ". ".join(" ".join(rng.choice(vocab) for _ in range(rng.randint(8, 18)))
                             for _ in range(rng.randint(6, 12))) + "."
            fh.write(json.dumps({"id": "h{0}".format(i), "text": text}) + "\n")
    return path


def test_cli_calibrate_then_score_with_it(tmp_path, capsys):
    corpus = _human_corpus(tmp_path, 260)
    out = os.path.join(str(tmp_path), "mine.json")
    assert main(["calibrate", corpus, "--out", out]) == 0
    printed = capsys.readouterr().out
    assert "held-out half B" in printed and "--calibration " + out in printed
    cal = load_calibration(out)
    assert isinstance(cal, Calibration) and cal.meta["baseline"] == corpus
    assert main(["score", corpus, "--reference", corpus, "--calibration", out]) == 0


def test_cli_calibrate_refuses_a_tiny_corpus(tmp_path, capsys):
    corpus = _human_corpus(tmp_path, 30)
    assert main(["calibrate", corpus, "--out", os.path.join(str(tmp_path), "x.json")]) == 2
    assert "a calibration needs at least" in capsys.readouterr().err
