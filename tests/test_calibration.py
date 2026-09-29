"""The shipped calibration file (built by benchmarks/human_baseline.py calibrate)."""

from __future__ import annotations

from chaff.document import Document
from chaff.metrics import FAMILIES
from chaff.metrics.artifacts import artifact_profile
from chaff.scoring import Calibration, load_calibration
from chaff.tokenization import build_view


def test_shipped_calibration_covers_every_family():
    cal = load_calibration()
    assert set(cal.thresholds) == set(FAMILIES)
    for family in FAMILIES:
        null = cal.family_null[family]
        assert null == sorted(null) and len(null) > 100


def test_every_artifact_signal_has_a_human_reference():
    """R5: an artifact signal missing from the human reference is silently ignored by
    fusion. Adding a detector without recalibrating must fail here, not in production."""
    cal = load_calibration()
    text = "- **A:** b\n" * 6 + "I hope this helps. " * 20
    emitted = {s.name for s in artifact_profile(Document("d", text), build_view(text))}
    assert emitted and emitted <= set(cal.human.counts)


def test_recorded_held_out_rate_meets_the_target():
    cal = load_calibration()
    held = cal.meta["evaluation"]["B"]
    assert held["likely_rate"] <= 2 * cal.meta["likely_target"]
    assert cal.meta["alpha"] > 0


def test_calibration_round_trips():
    cal = load_calibration()
    again = Calibration.from_dict(cal.to_dict())
    assert again.thresholds == cal.thresholds
    assert again.human.tail_p("assistant_echo_rate", 1.0, 1) == cal.human.tail_p("assistant_echo_rate", 1.0, 1)


def test_calibration_contains_numbers_not_text():
    """The baseline text must never ship: only distributions of numbers."""
    cal = load_calibration()
    for pairs in cal.human.counts.values():
        for value, count in pairs:
            assert isinstance(value, (int, float)) and isinstance(count, int)


def test_score_is_calibrated_so_typical_human_text_scores_about_fifty():
    """Regression: an uncalibrated statistic sat near 70 for clean text (max and
    second-max of four uniform draws), so human documents read as suspicious."""
    held = load_calibration().meta["evaluation"]["B"]
    assert 40 <= held["score_median"] <= 60
    assert held["score_p95"] >= 85


def test_score_is_monotone_in_the_evidence_statistic():
    cal = load_calibration()
    scores = [cal.score(x / 2.0) for x in range(0, 81)]
    assert scores == sorted(scores) and scores[0] < 5 and scores[-1] > 95


def test_tier_and_score_agree_on_held_out_human_text():
    """Every held-out document that fires a family scores above the clean median."""
    by_tier = load_calibration().meta["evaluation"]["B"]["score_by_tier_min_median"]
    clean_median = by_tier["CLEAN"][1]
    assert by_tier["SUSPECT"][0] > clean_median
    assert by_tier["LIKELY"][0] > by_tier["SUSPECT"][1]
