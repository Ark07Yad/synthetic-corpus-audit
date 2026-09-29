"""Fusion (phase 5). Each requirement R1-R7 and each bug found while building fusion
has at least one test here, named for what it protects."""

from __future__ import annotations

import math
import random

import pytest

from chaff.scoring import (
    BASIS_CORPUS,
    BASIS_HUMAN,
    CALIBRATION_PRECISION,
    MAX_BINS,
    MIN_BIN,
    TIER_CLEAN,
    TIER_LIKELY,
    TIER_SUSPECT,
    TIER_UNSCORED,
    Z_CAP,
    Calibration,
    Evidence,
    HumanReference,
    Row,
    SignalInfo,
    aggregate,
    build_reference_stats,
    build_signal_stats,
    normalise,
    score_row,
)

CATALOG = {
    "dist_a": SignalInfo("distributional", -1),
    "dist_b": SignalInfo("distributional", 1),
    "surp": SignalInfo("surprisal", -1),
    "reas": SignalInfo("reasoning", -1),
    "echo": SignalInfo("artifact", 1),
    "noise": SignalInfo("artifact", -1),
}


def _spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def _calibration(thresholds=None, null=None):
    human = HumanReference.from_values({"echo": [0.0] * 97 + [0.4, 0.6, 0.9],
                                        "noise": [0.0] * 60 + [2.0] * 40})
    # One generator for all draws. The original built a new Random(1) per draw, so every
    # null was 201 copies of one number — invisible under one-sided scoring, and exposed
    # the moment two-sided scoring put every document in an extreme tail.
    rng_null = random.Random(1)
    null = null or {f: sorted(rng_null.gauss(0, 1) for _ in range(201))
                    for f in ("distributional", "surprisal", "reasoning", "artifact")}
    rng = random.Random(2)
    # Fisher's X under an independent null is chi-squared with 8 degrees of freedom.
    score_null = sorted(sum(rng.expovariate(0.5) for _ in range(4)) for _ in range(201))
    return Calibration(thresholds=thresholds or {"distributional": 2.0, "surprisal": 2.0,
                                                  "reasoning": 2.0, "artifact": 2.0},
                       family_null=null, human=human, score_null=score_null)


def _stats(n=400, seed=0):
    rng = random.Random(seed)
    rows = [Row(doc_id=str(i), n_words=rng.randint(100, 4000), analysable=True,
                signals={"dist_a": rng.gauss(5, 1), "dist_b": rng.gauss(5, 1),
                         "surp": rng.gauss(8, 1), "reas": rng.gauss(0.7, 0.1)})
            for i in range(n)]
    return build_reference_stats(rows, CATALOG)


# --------------------------------------------------------------- R2: strata

def test_bin_count_follows_min_bin_and_max_bins():
    assert len(build_signal_stats([(i, 1.0) for i in range(MIN_BIN - 1)]).edges) == 1
    assert len(build_signal_stats([(i, 1.0) for i in range(MIN_BIN * 3)]).edges) == 3
    assert len(build_signal_stats([(i, 1.0) for i in range(MIN_BIN * 100)]).edges) == MAX_BINS


def test_length_strata_remove_a_pure_length_confound():
    """R2: a signal that is a function of length plus noise must not rank documents by
    length after normalisation. Unstratified, it does exactly that."""
    rng = random.Random(3)
    pairs = [(n, math.log(n) + rng.gauss(0, 0.05)) for n in (rng.randint(80, 8000) for _ in range(1000))]
    stats = build_signal_stats(pairs)
    stratified = [(v - stats.locate(n)[0]) / stats.locate(n)[1] for n, v in pairs]
    unstratified = [(v - stats.global_median) / stats.global_mad for _, v in pairs]
    lengths = [n for n, _ in pairs]
    assert abs(_spearman(lengths, unstratified)) > 0.9
    assert abs(_spearman(lengths, stratified)) < 0.1


def test_zero_mad_stratum_falls_back_to_global_scale():
    stats = build_signal_stats([(n, 1.0 if n < 100 else float(n)) for n in range(1, 200)])
    _, scale = stats.locate(10)
    assert scale == stats.global_mad > 0


# ----------------------------------------------------- R5: human reference

def test_no_evidence_on_a_zero_inflated_signal_is_neutral():
    human = HumanReference.from_values({"echo": [0.0] * 97 + [0.5] * 3})
    assert human.tail_p("echo", 0.0, 1) == pytest.approx(0.5, abs=0.03)


def test_a_value_beyond_every_human_document_is_rare_but_not_impossible():
    human = HumanReference.from_values({"echo": [0.0] * 100})
    assert 0 < human.tail_p("echo", 5.0, 1) < 0.01


def test_tail_direction_is_mirrored_for_negative_signals():
    human = HumanReference.from_values({"noise": [0.0] * 50 + [2.0] * 50})
    # Direction -1: *less* noise than humans is the suspicious tail.
    assert human.tail_p("noise", 0.0, -1) < human.tail_p("noise", 2.0, -1)


def test_artifacts_are_normalised_against_humans_not_the_corpus():
    ev = {e.signal: e for e in normalise({"echo": 3.0, "dist_a": 5.0}, 500, CATALOG, _stats(), _calibration())}
    assert ev["echo"].basis == BASIS_HUMAN and ev["echo"].z > 2
    assert ev["dist_a"].basis == BASIS_CORPUS


# ------------------------------------------------------------ normalisation

def test_direction_is_applied():
    stats, cal = _stats(), _calibration()
    low = {e.signal: e.z for e in normalise({"dist_a": 1.0, "dist_b": 1.0}, 500, CATALOG, stats, cal)}
    assert low["dist_a"] > 0 > low["dist_b"]   # dist_a: lower is suspicious; dist_b: higher is


def test_z_is_clipped():
    ev = normalise({"dist_a": -1e9}, 500, CATALOG, _stats(), _calibration())
    assert ev[0].z == Z_CAP


def test_dropped_and_unknown_signals_are_skipped():
    ev = normalise({"surp": 1.0, "mystery": 3.0}, 500, CATALOG, _stats(), _calibration(), dropped={"surp"})
    assert ev == []


# --------------------------------------------------------------- R1: families

def _ev(z, family="artifact"):
    return Evidence("s", family, 0.0, z, "x")


def test_continuous_families_average_their_signals():
    assert aggregate([_ev(3.0, "distributional"), _ev(1.0, "distributional")])["distributional"] == (2.0, 2)


def test_artifact_family_takes_the_strongest_tell_with_a_multiple_look_correction():
    one = aggregate([_ev(3.0)])["artifact"][0]
    among_six = aggregate([_ev(3.0)] + [_ev(0.0)] * 5)["artifact"][0]
    assert among_six < one            # looking six times makes one hit less surprising
    assert among_six > 2.0            # ... but a strong tell is still strong


def test_more_detectors_do_not_inflate_a_family():
    few = aggregate([_ev(1.0)] * 2)["artifact"][0]
    many = aggregate([_ev(1.0)] * 12)["artifact"][0]
    assert many <= few + 1e-9


# ---------------------------------------------------------- R6/R7: tiers

def _row(**signals):
    return Row(doc_id="d", n_words=500, analysable=True, signals=signals)


def test_unanalysable_documents_are_unscored():
    doc = score_row(Row("d", 10, False, {}), CATALOG, _stats(), _calibration())
    assert doc.tier == TIER_UNSCORED and doc.score is None


def test_tiers_count_fired_families():
    stats = _stats()
    strong_low = -100.0   # direction -1 signals: very low = very synthetic
    cal = _calibration()
    assert score_row(_row(dist_a=5.0, surp=8.0), CATALOG, stats, cal).tier == TIER_CLEAN
    assert score_row(_row(dist_a=strong_low, surp=8.0), CATALOG, stats, cal).tier == TIER_SUSPECT
    assert score_row(_row(dist_a=strong_low, surp=strong_low), CATALOG, stats, cal).tier == TIER_LIKELY


def test_a_single_family_can_never_reach_likely_synthetic():
    """OBJECTIVE §4.3: however strong, one family is SUSPECT at most."""
    doc = score_row(_row(echo=500.0), CATALOG, _stats(), _calibration())
    assert doc.tier == TIER_SUSPECT


def test_threshold_is_strict():
    """Regression: with ">=", a threshold landing on the artifact null's giant tie would
    fire on every tied document at once. A family z exactly at the threshold must not
    fire; a hair above must."""
    stats = _stats()
    z = score_row(_row(echo=0.6), CATALOG, stats, _calibration()).families["artifact"].z
    base = {"distributional": 9.0, "surprisal": 9.0, "reasoning": 9.0}
    at = score_row(_row(echo=0.6), CATALOG, stats, _calibration(thresholds=dict(base, artifact=z)))
    below = score_row(_row(echo=0.6), CATALOG, stats,
                      _calibration(thresholds=dict(base, artifact=z - 10 ** -CALIBRATION_PRECISION)))
    assert not at.families["artifact"].fired
    assert below.families["artifact"].fired


def test_percentile_of_a_tie_is_its_midpoint():
    """Regression: 'share strictly below' put every no-evidence document at the top of
    the artifact null's tie (the 89.5th percentile) and inflated clean scores."""
    cal = _calibration(null={"artifact": [-3.0] * 10 + [-2.0] * 180 + [1.0] * 11})
    assert cal.percentile("artifact", -2.0) == pytest.approx((10 + 190) / 2 / 201)


def test_family_z_is_compared_at_calibration_precision():
    """Regression, through score_row itself: the family z is computed at full precision,
    the calibration stores ties at 6 dp, and the unrounded z sat a hair off the tie —
    landing at the 0th or 100th percentile instead of the middle of it."""
    stats, probe_cal = _stats(), _calibration()
    evidence = normalise({"echo": 0.6, "noise": 2.0}, 500, CATALOG, stats, probe_cal)
    z_full = aggregate(evidence)["artifact"][0]
    tie = round(z_full, CALIBRATION_PRECISION)
    assert z_full != tie, "probe must differ from its rounded value to exercise the bug"
    cal = _calibration(null={f: [tie] * 201 for f in ("artifact", "distributional", "surprisal", "reasoning")})
    doc = score_row(_row(echo=0.6, noise=2.0), CATALOG, stats, cal)
    assert doc.families["artifact"].percentile == pytest.approx(0.5)


def test_more_evidence_scores_higher():
    stats, cal = _stats(), _calibration()
    one = score_row(_row(dist_a=-100.0, surp=8.0, reas=0.7), CATALOG, stats, cal)
    two = score_row(_row(dist_a=-100.0, surp=-100.0, reas=0.7), CATALOG, stats, cal)
    assert two.score > one.score


def test_a_document_that_fires_outranks_one_with_no_evidence():
    """Regression: the mean-of-top-two statistic scored documents firing one extreme
    family (e.g. an assistant echo) *below* a typical clean document, so a triage queue
    sorted by score would never surface them. Tier and score must agree."""
    stats, cal = _stats(), _calibration()
    typical = score_row(_row(dist_a=5.0, dist_b=5.0, surp=8.0, reas=0.7), CATALOG, stats, cal)
    one_tell = score_row(_row(dist_a=5.0, dist_b=5.0, surp=8.0, reas=0.7, echo=50.0), CATALOG, stats, cal)
    assert one_tell.tier == TIER_SUSPECT and typical.tier == TIER_CLEAN
    assert one_tell.score > typical.score


def test_score_is_none_without_a_calibrated_score_null():
    cal = _calibration()
    cal.score_null = []
    assert score_row(_row(dist_a=5.0), CATALOG, _stats(), cal).score is None


def test_evidence_is_sorted_strongest_first():
    doc = score_row(_row(dist_a=-100.0, dist_b=5.0, surp=8.0), CATALOG, _stats(), _calibration())
    strengths = [e.strength for e in doc.evidence]
    assert strengths == sorted(strengths, reverse=True)


def test_two_sided_families_fire_in_either_direction():
    """Phase 6: modern models write *more* diverse text than human genre norms, the
    opposite of the original hypothesis. A continuous family must flag both extremes."""
    stats = _stats()
    cal = _calibration()
    cal.thresholds_low = {"distributional": -2.0, "surprisal": -2.0, "reasoning": -2.0}
    too_compressed = score_row(_row(dist_a=-100.0), CATALOG, stats, cal).families["distributional"]
    too_diverse = score_row(_row(dist_a=100.0), CATALOG, stats, cal).families["distributional"]
    assert (too_compressed.fired, too_compressed.side) == (True, "high")
    assert (too_diverse.fired, too_diverse.side) == (True, "low")


def test_a_too_diverse_document_scores_as_atypical():
    """What SIDEDNESS actually controls is the score: a document at the *opposite*
    extreme from the original hypothesis (richer, less repetitive than its genre) must
    score as strong evidence. One-sided scoring gave it the lowest score possible."""
    stats, cal = _stats(), _calibration()
    typical = score_row(_row(dist_a=5.0, dist_b=5.0, surp=8.0, reas=0.7), CATALOG, stats, cal)
    too_diverse = score_row(_row(dist_a=100.0, dist_b=-100.0, surp=8.0, reas=0.7), CATALOG, stats, cal)
    assert too_diverse.families["distributional"].z < -3
    assert too_diverse.score > typical.score + 20


def test_evidence_in_the_unexpected_direction_ranks_first():
    doc = score_row(_row(dist_a=100.0, dist_b=5.0, surp=8.0), CATALOG, _stats(), _calibration())
    assert doc.evidence[0].signal == "dist_a" and doc.evidence[0].z < -3


def test_artifact_family_stays_one_sided():
    """A tell only means something in one direction: an unusually *low* hedging rate is
    not evidence of anything."""
    cal = _calibration()
    cal.thresholds_low = {"distributional": -2.0}
    doc = score_row(_row(noise=50.0), CATALOG, _stats(), cal)   # lots of human noise
    assert not doc.families["artifact"].fired


def test_evidence_records_which_way_a_signal_deviates():
    doc = score_row(_row(dist_a=-100.0), CATALOG, _stats(), _calibration())
    assert {e.signal: e.deviation for e in doc.evidence}["dist_a"] == "below"


def test_labels_never_affect_scoring():
    """The leakage firewall: identical signals, different labels, identical outcome."""
    stats, cal = _stats(), _calibration()
    signals = {"dist_a": 2.0, "surp": 6.0, "echo": 0.6}
    results = [score_row(Row("d", 500, True, dict(signals), label=label), CATALOG, stats, cal)
               for label in ("human", "synthetic", None)]
    assert len({(r.tier, r.score) for r in results}) == 1


# ------------------------------------------------------- phase 6: stratify-by

def _grouped_rows():
    rng = random.Random(4)
    rows = []
    for i in range(300):          # majority genre: dist_a around 5
        rows.append(Row(str(i), 500, True, {"dist_a": rng.gauss(5, 1)}, group="major"))
    for i in range(60):           # minority genre: dist_a around 9
        rows.append(Row("m%d" % i, 500, True, {"dist_a": rng.gauss(9, 1)}, group="minor"))
    for i in range(10):           # too small to stand alone
        rows.append(Row("t%d" % i, 500, True, {"dist_a": rng.gauss(9, 1)}, group="tiny"))
    return rows


def test_stratified_stats_keep_a_group_entry_only_when_it_is_large_enough():
    from chaff.scoring import stats_for
    stats = build_reference_stats(_grouped_rows(), CATALOG, stratify=True)
    assert ("major", "dist_a") in stats and ("minor", "dist_a") in stats
    assert ("tiny", "dist_a") not in stats
    assert stats_for(stats, "dist_a", "tiny") is stats["dist_a"]    # falls back to corpus-wide


def test_stratifying_removes_the_minority_genre_penalty():
    """The minority genre is unusual only relative to the majority: normalised within its
    own group, a typical minority document is typical."""
    cal = _calibration()
    plain = build_reference_stats(_grouped_rows(), CATALOG)
    strat = build_reference_stats(_grouped_rows(), CATALOG, stratify=True)
    doc = Row("x", 500, True, {"dist_a": 9.0}, group="minor")
    z_plain = normalise(doc.signals, 500, CATALOG, plain, cal, group=doc.group)[0].z
    z_strat = normalise(doc.signals, 500, CATALOG, strat, cal, group=doc.group)[0].z
    assert abs(z_plain) > 2 and abs(z_strat) < 0.5
