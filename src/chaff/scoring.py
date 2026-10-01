"""Fusion: per-document signals -> calibrated family evidence -> tier, score, evidence.

Everything upstream of this module *measures*. This module decides what the
measurements mean, and every decision in it traces to a requirement found in an earlier
phase (PROJECT.md R1-R7) rather than to taste.

The four steps
--------------
1. **Normalise** each signal into an oriented z (positive = more synthetic-like).

   * Continuous signals (distributional, surprisal, reasoning) use a robust z —
     median and MAD — **within length strata** of the reference distribution (R2).
     Measured on 1,748 human documents, eight distributional signals correlate with
     document length at |rho| 0.48-0.95; unstratified, "corpus-relative" would mostly
     mean "ranked by length". Equal-frequency length bins of at least 50 documents
     (up to 20) bring every signal to |rho| <= 0.06.
   * Artifact signals use a tail probability under the **human baseline** distribution
     shipped in the calibration (R5). Most artifact rates are zero for most documents,
     so median = MAD = 0 and a robust z is undefined. Normalising them against the
     audited corpus would also fail exactly when it matters: in a crawl that is 20%
     synthetic, hedging is "normal". Human text is the right null for a tell.

2. **Aggregate within each family** (R1). Distributional, surprisal and reasoning
   signals are repeated measurements of one property, so they are averaged. Artifact
   signals are distinct tells, where one strong one is evidence on its own, so the
   strongest is taken with a Sidak correction for having looked at several.

3. **Fire** a family when its aggregate exceeds a threshold calibrated on half A of the
   human baseline (R7), chosen so that at most 1% of clean documents reach
   ``LIKELY_SYNTHETIC``. That target is set on the *measured* joint firing rate, which
   includes whatever dependence the families really have (R6) — not on an independence
   assumption that phase 4 showed is false.

**Two-sided since phase 6.** Distributional, surprisal and reasoning families flag a
document that is atypical for its genre in *either* direction; only the artifact family is
one-sided. The project began from a one-sided premise — synthetic text has a compressed,
repetitive vocabulary — and evaluation on real model output showed that holds weakly for
older base models and is **inverted** for modern instruction-tuned ones, which write with
*richer* vocabulary than the human text of the same genre (per-signal AUC as low as 0.12
on Claude-written technical documentation, 0.20 on GPT-4). One-sided scoring ranked
Claude's technical docs as more human than human docs (AUC 0.06). Two-sided scoring,
pre-registered and confirmed on 1,442 MAGE documents no analysis had touched, took AUC
from 0.61 to 0.88 there. See ``benchmarks/EVALUATION.md``.

4. **Tier and score.** Two or more families firing is ``LIKELY_SYNTHETIC``, one is
   ``SUSPECT``, none is ``CLEAN`` (OBJECTIVE §4.3) — corroboration is the *tier's* job.
   The 0-100 score ranks combined evidence strength: Fisher's method over families,
   ``X = -2 * sum(ln(tail_f))``, reported as a **percentile among clean human
   documents**. 50 is typical of human text; 99 means more evidence than 99% of it.

   Two statistics were built and replaced before this one:

   * *Mean of the two strongest families' percentiles*, uncalibrated, sat near 70 for
     clean text (the expected largest and second-largest of four uniform draws are 0.8
     and 0.6), so "70/100" on a human document read as suspicious.
   * The same statistic *calibrated* fixed that, but it punishes one extreme family paired
     with a weak one: documents that fire a family — including ones containing "As an AI
     language model" — scored ~22, below a typical clean document. In a triage queue
     sorted by score they would never be seen. Tier and score contradicted each other.

   Fisher's method has neither problem: one extreme family and several moderate ones both
   raise X, and families pointing toward "human" contribute almost nothing. The families
   are correlated, so X's null is not the textbook chi-squared — it is calibrated
   empirically like everything else.

``Document.label`` never enters this module's computations. It is carried through
to the output for evaluation only; ``test_scoring.py`` asserts that permuting labels
changes nothing.
"""

from __future__ import annotations

import bisect
import json
import math
from array import array
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .metrics import (
    FAMILIES,
    FAMILY_ARTIFACT,
    FAMILY_DISTRIBUTIONAL,
    FAMILY_REASONING,
    FAMILY_SURPRISAL,
)
from .stats import mad, median

TIER_UNSCORED = "UNSCORED"
TIER_CLEAN = "CLEAN"
TIER_SUSPECT = "SUSPECT"
TIER_LIKELY = "LIKELY_SYNTHETIC"
TIERS = (TIER_LIKELY, TIER_SUSPECT, TIER_CLEAN, TIER_UNSCORED)

#: Normalised z is clipped here so that one extreme signal cannot dominate its family.
Z_CAP = 6.0

#: Decimal places the calibration stores thresholds and null quantiles at. Family z is
#: rounded to the same precision before it is compared with either: the artifact null
#: is one enormous tie, and a z computed at full precision sat a hair above the rounded
#: tie value, landing after all 177 tied copies (the 89.5th percentile) instead of in
#: the middle of them.
CALIBRATION_PRECISION = 6

#: Length strata: equal-frequency bins of at least this many documents, at most MAX_BINS.
#: Measured: 10 bins brought every signal to |rho(z, log length)| <= 0.06 on the baseline.
MIN_BIN = 50
MAX_BINS = 20

BASIS_CORPUS = "corpus"
BASIS_HUMAN = "human-baseline"

#: How each family is normalised (R5) and aggregated (R1).
NORMALISATION = {
    FAMILY_DISTRIBUTIONAL: BASIS_CORPUS,
    FAMILY_SURPRISAL: BASIS_CORPUS,
    FAMILY_REASONING: BASIS_CORPUS,
    FAMILY_ARTIFACT: BASIS_HUMAN,
}
AGGREGATION = {
    FAMILY_DISTRIBUTIONAL: "mean",
    FAMILY_SURPRISAL: "mean",
    FAMILY_REASONING: "mean",
    FAMILY_ARTIFACT: "any",
}

#: Which tails of a family's null count as evidence. Artifact signals are tells, which
#: only mean something in one direction. The other families measure how typical a
#: document's distribution is, and phase 6 found synthetic text deviating both ways
#: depending on the generator (see the module docstring).
SIDE_ONE = "one-sided"
SIDE_TWO = "two-sided"
SIDEDNESS = {
    FAMILY_DISTRIBUTIONAL: SIDE_TWO,
    FAMILY_SURPRISAL: SIDE_TWO,
    FAMILY_REASONING: SIDE_TWO,
    FAMILY_ARTIFACT: SIDE_ONE,
}

_NORMAL = NormalDist()
_P_FLOOR = 1e-12


def _z_from_tail(p: float) -> float:
    """Upper-tail probability -> z, clipped to [-Z_CAP, Z_CAP]."""
    p = min(max(p, _P_FLOOR), 1.0 - _P_FLOOR)
    return max(-Z_CAP, min(Z_CAP, _NORMAL.inv_cdf(1.0 - p)))


def _tail_from_z(z: float) -> float:
    return 1.0 - _NORMAL.cdf(z)


def _clip(z: float) -> float:
    return max(-Z_CAP, min(Z_CAP, z))


# ------------------------------------------------------------------ inputs

@dataclass
class SignalInfo:
    """What fusion needs to know about a signal: collected from emitted ``Signal``s."""

    family: str
    direction: int
    description: str = ""


@dataclass
class Row:
    """One document's measurements, as spilled to disk by pass 2 (R3)."""

    doc_id: str
    n_words: int
    analysable: bool
    signals: Dict[str, float]
    label: Optional[str] = None
    #: Stratification group (``--stratify-by``): documents are normalised against
    #: others in the same group. ``None`` when not stratifying.
    group: Optional[str] = None
    #: Why a long-enough document was not analysed (the language guard), if it was not.
    reason: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "Row":
        return cls(**json.loads(line))


# ------------------------------------------------------ continuous: strata

@dataclass
class SignalStats:
    """Length-stratified robust location and scale for one continuous signal."""

    edges: List[int]          # inclusive upper n_words bound of each bin, ascending
    medians: List[float]
    mads: List[float]
    global_median: float
    global_mad: float
    n: int

    def locate(self, n_words: int) -> Tuple[float, float]:
        """Median and MAD of the length stratum ``n_words`` falls in. Documents beyond
        the reference's range use the nearest stratum. A stratum whose MAD is zero
        falls back to the global MAD rather than dividing by zero."""
        i = min(bisect.bisect_left(self.edges, n_words), len(self.edges) - 1)
        scale = self.mads[i] if self.mads[i] > 0 else self.global_mad
        return self.medians[i], scale


def build_signal_stats(pairs: Sequence[Tuple[int, float]]) -> Optional[SignalStats]:
    """Stats from ``(n_words, value)`` pairs, or ``None`` if there are none."""
    if not pairs:
        return None
    lengths = array("i", (p[0] for p in pairs))
    values = array("d", (p[1] for p in pairs))
    return _stats_from_columns(lengths, values)


def _stats_from_columns(lengths: "array", values: "array") -> Optional[SignalStats]:
    """Stats from parallel typed columns.

    Columns rather than tuples because this is fusion's memory high-water mark: a
    ``(n_words, value)`` tuple costs ~100 bytes against ~12 in two typed arrays, which
    is the difference between ~2.4 GB and ~0.3 GB of reference stats per million
    documents at ~24 signals. The sort permutation is built for one signal at a time,
    so it never exists for all signals at once.
    """
    n = len(values)
    if n == 0:
        return None
    order = sorted(range(n), key=lengths.__getitem__)
    n_bins = max(1, min(MAX_BINS, n // MIN_BIN))
    edges, medians, mads = [], [], []
    for b in range(n_bins):
        idx = order[b * n // n_bins:(b + 1) * n // n_bins]
        chunk = [values[i] for i in idx]
        edges.append(lengths[idx[-1]])
        medians.append(median(chunk))
        mads.append(mad(chunk))
    return SignalStats(edges=edges, medians=medians, mads=mads,
                       global_median=median(values), global_mad=mad(values), n=n)


#: Key of the corpus-wide statistics in a stratified stats table.
ALL_GROUPS = None


def build_reference_stats(
    rows: Iterable[Row],
    catalog: Mapping[str, SignalInfo],
    stratify: bool = False,
) -> Dict[Any, SignalStats]:
    """Stats for every corpus-normalised signal, from a stream of reference rows.

    Unstratified, keys are signal names. Stratified (``--stratify-by``), keys are also
    ``(group, signal)`` pairs, alongside the corpus-wide entries — a group with fewer
    than :data:`MIN_BIN` documents for a signal gets no entry of its own and falls back
    to corpus-wide stats in :func:`stats_for`.

    Why stratify at all: corpus-relative normalisation makes the corpus majority the
    definition of normal. On the human baseline the distributional family fired on 6.1%
    of the minority genre against 1.8% of the majority. If documents carry a genre or
    source field, normalising within it removes that penalty.
    """
    columns: Dict[Any, Tuple["array", "array"]] = {}

    def push(key, n_words, value):
        if key not in columns:
            columns[key] = (array("i"), array("d"))
        columns[key][0].append(n_words)
        columns[key][1].append(value)

    for row in rows:
        if not row.analysable:
            continue
        for name, value in row.signals.items():
            info = catalog.get(name)
            if info and NORMALISATION.get(info.family) == BASIS_CORPUS:
                push(name, row.n_words, value)
                if stratify and row.group is not None:
                    push((row.group, name), row.n_words, value)
    out: Dict[Any, SignalStats] = {}
    for key in list(columns):
        lengths, values = columns.pop(key)   # release each column once summarised
        if isinstance(key, tuple) and len(values) < MIN_BIN:
            continue                          # too small to stand alone: fall back
        stats = _stats_from_columns(lengths, values)
        if stats is not None:
            out[key] = stats
    return out


def stats_for(stats: Mapping[Any, SignalStats], name: str, group: Optional[str]) -> Optional[SignalStats]:
    """The group's own stats for a signal when it has them, else corpus-wide."""
    if group is not None:
        own = stats.get((group, name))
        if own is not None:
            return own
    return stats.get(name)


# ------------------------------------------------ artifacts: human baseline

@dataclass
class HumanReference:
    """Empirical distribution of each artifact signal on guaranteed-human text, stored
    as sorted ``(value, count)`` pairs — most artifact rates are exactly zero."""

    counts: Dict[str, List[Tuple[float, int]]]

    @classmethod
    def from_values(cls, values: Mapping[str, Sequence[float]]) -> "HumanReference":
        counts = {}
        for name, vals in values.items():
            tally: Dict[float, int] = {}
            for v in vals:
                tally[round(v, 6)] = tally.get(round(v, 6), 0) + 1
            counts[name] = sorted(tally.items())
        return cls(counts=counts)

    def tail_p(self, name: str, value: float, direction: int) -> Optional[float]:
        """Mid-rank tail probability of ``value`` in the suspicious direction, with
        add-half smoothing so an unseen extreme is rare but not impossible."""
        pairs = self.counts.get(name)
        if not pairs:
            return None
        total = sum(c for _, c in pairs)
        value = round(value, 6)
        beyond = sum(c for v, c in pairs if (v > value if direction > 0 else v < value))
        equal = sum(c for v, c in pairs if v == value)
        return (beyond + 0.5 * equal + 0.5) / (total + 1.0)


# -------------------------------------------------------------- calibration

@dataclass
class Calibration:
    """Thresholds and null distributions learned on the human baseline (R7).

    Built by ``benchmarks/human_baseline.py calibrate`` and shipped as
    ``chaff/calibration.json``. Family thresholds are in standardised units, so they
    transfer to any corpus normalised the same way — subject to the caveat that a
    heterogeneous or heavily contaminated corpus has heavier tails than the baseline.
    """

    thresholds: Dict[str, float]
    family_null: Dict[str, List[float]]
    human: HumanReference
    signals: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    #: Lower thresholds, for two-sided families only: fire when z falls strictly below.
    thresholds_low: Dict[str, float] = field(default_factory=dict)
    #: Null distribution (quantiles) of the evidence statistic on clean documents.
    score_null: List[float] = field(default_factory=list)

    def percentile(self, family: str, z: float) -> Optional[float]:
        """Mid-rank percentile of ``z`` in the family's human null distribution.

        Mid-rank, not "share strictly below": the artifact null is massively tied at the
        no-evidence value, with a small tail *below* it (human typing noise pushes z
        down). "Strictly below" placed every no-evidence document at the top of that
        tie — the 89th percentile — and inflated the score of every clean document.
        Mid-rank puts a tie at its centre, so no evidence reads as neutral.
        """
        null = self.family_null.get(family)
        if not null:
            return None
        return _mid_rank(null, z)

    def score(self, statistic: float) -> Optional[float]:
        """0-100: mid-rank percentile of the evidence statistic among clean documents."""
        if not self.score_null:
            return None
        return 100.0 * _mid_rank(self.score_null, round(statistic, CALIBRATION_PRECISION))

    def statistic(self, percentiles: Mapping[str, Optional[float]]) -> float:
        return evidence_statistic(percentiles, {f: len(v) for f, v in self.family_null.items()})

    def fires(self, family: str, z: float) -> Optional[str]:
        """``"high"`` or ``"low"`` if ``z`` passes the family's threshold, else ``None``.
        Strict comparisons: see :data:`CALIBRATION_PRECISION` and PROJECT D27."""
        high = self.thresholds.get(family)
        if high is not None and z > high:
            return "high"
        low = self.thresholds_low.get(family)
        if low is not None and z < low:
            return "low"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"thresholds": self.thresholds, "family_null": self.family_null,
                "human": self.human.counts, "signals": self.signals, "meta": self.meta,
                "score_null": self.score_null, "thresholds_low": self.thresholds_low}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Calibration":
        human = HumanReference(counts={k: [tuple(p) for p in v] for k, v in data["human"].items()})
        return cls(thresholds=dict(data["thresholds"]), family_null=dict(data["family_null"]),
                   human=human, signals=list(data.get("signals", [])), meta=dict(data.get("meta", {})),
                   score_null=list(data.get("score_null", [])),
                   thresholds_low=dict(data.get("thresholds_low", {})))


def _mid_rank(sorted_values: Sequence[float], x: float) -> float:
    """Share of ``sorted_values`` below ``x``, counting ties as half: a value in the
    middle of a tie reads as the middle, not the top or bottom, of it."""
    lo = bisect.bisect_left(sorted_values, x)
    hi = bisect.bisect_right(sorted_values, x)
    return (lo + hi) / 2.0 / float(len(sorted_values))


#: Fisher contribution of a family with no evidence: E[-2 ln U] for U ~ Uniform(0, 1).
#: Imputing the null expectation keeps X comparable between documents with different
#: numbers of families (reasoning is withheld on short documents, for example).
_FISHER_MISSING = 2.0


def evidence_statistic(percentiles: Mapping[str, Optional[float]], null_sizes: Mapping[str, int]) -> float:
    """Fisher's combined evidence over the four families.

    Each family contributes ``-2 ln(tail)``, where ``tail`` is its probability under the
    human null: upper tail ``1 - p`` for one-sided families, ``2 * min(p, 1 - p)`` for
    two-sided ones. Floored at half a quantile step, so a family beyond every null value
    is very strong rather than infinite.
    """
    x = 0.0
    for family in FAMILIES:
        p = percentiles.get(family)
        if p is None:
            x += _FISHER_MISSING
            continue
        floor = 0.5 / max(1, null_sizes.get(family, 1))
        tail = 2.0 * min(p, 1.0 - p) if SIDEDNESS.get(family) == SIDE_TWO else 1.0 - p
        x += -2.0 * math.log(max(tail, floor))
    return x


def load_calibration(path: Optional[str] = None) -> Calibration:
    """Load a calibration file, defaulting to the one shipped with the package."""
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            return Calibration.from_dict(json.load(fh))
    import os
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
    with open(here, encoding="utf-8") as fh:
        return Calibration.from_dict(json.load(fh))


# ------------------------------------------------------------------ outputs

@dataclass
class Evidence:
    signal: str
    family: str
    value: float
    z: float
    basis: str
    description: str = ""
    #: "above" / "below" the typical value for comparable documents (raw, not oriented).
    deviation: str = ""

    @property
    def strength(self) -> float:
        """How much this signal contributes as evidence: ``|z|`` for two-sided families,
        where either direction counts; ``z`` for one-sided ones."""
        return abs(self.z) if SIDEDNESS.get(self.family) == SIDE_TWO else self.z


@dataclass
class FamilyResult:
    z: Optional[float]
    n_signals: int
    fired: bool
    threshold: Optional[float]
    percentile: Optional[float]
    #: Which tail fired ("high" = the originally hypothesised synthetic direction,
    #: "low" = the opposite extreme), or None.
    side: Optional[str] = None
    threshold_low: Optional[float] = None


@dataclass
class ScoredDoc:
    doc_id: str
    tier: str
    score: Optional[float]
    n_words: int
    families: Dict[str, FamilyResult]
    evidence: List[Evidence]
    label: Optional[str] = None
    #: Why the document is UNSCORED, when it is long enough to have been scored.
    reason: Optional[str] = None

    @property
    def fired(self) -> List[str]:
        return [f for f, r in self.families.items() if r.fired]

    def to_dict(self, max_evidence: int = 8) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "tier": self.tier,
            "score": None if self.score is None else round(self.score, 2),
            "n_words": self.n_words,
            "families_fired": self.fired,
            "families": {f: {"z": None if r.z is None else round(r.z, 4), "n_signals": r.n_signals,
                             "fired": r.fired, "side": r.side, "threshold": r.threshold,
                             "threshold_low": r.threshold_low,
                             "percentile": None if r.percentile is None else round(r.percentile, 4)}
                         for f, r in self.families.items()},
            "evidence": [{"signal": e.signal, "family": e.family, "value": round(e.value, 6),
                          "z": round(e.z, 3), "strength": round(e.strength, 3),
                          "deviation": e.deviation, "basis": e.basis, "description": e.description}
                         for e in self.evidence[:max_evidence]],
            **({"label": self.label} if self.label is not None else {}),
            **({"reason": self.reason} if self.reason is not None else {}),
        }


# ------------------------------------------------------------------ scoring

def normalise(
    signals: Mapping[str, float],
    n_words: int,
    catalog: Mapping[str, SignalInfo],
    stats: Mapping[Any, SignalStats],
    calibration: Calibration,
    dropped: Iterable[str] = (),
    group: Optional[str] = None,
) -> List[Evidence]:
    """Step 1: every present signal as an oriented, clipped z."""
    dropped = set(dropped)
    out: List[Evidence] = []
    for name, value in signals.items():
        info = catalog.get(name)
        if info is None or name in dropped:
            continue
        basis = NORMALISATION.get(info.family)
        if basis == BASIS_HUMAN:
            p = calibration.human.tail_p(name, value, info.direction)
            if p is None:
                continue
            z = _z_from_tail(p)
        else:
            s = stats_for(stats, name, group)
            if s is None:
                continue
            location, scale = s.locate(n_words)
            if scale <= 0:
                continue  # degenerate everywhere: no information, not a zero-evidence vote
            z = _clip(info.direction * (value - location) / scale)
        raw = z * info.direction
        out.append(Evidence(signal=name, family=info.family, value=value, z=z,
                            basis=basis, description=info.description,
                            deviation="above" if raw > 0 else "below" if raw < 0 else ""))
    return out


def aggregate(evidence: Sequence[Evidence]) -> Dict[str, Tuple[float, int]]:
    """Step 2: one z per family with evidence, and how many signals formed it."""
    by_family: Dict[str, List[float]] = {}
    for e in evidence:
        by_family.setdefault(e.family, []).append(e.z)
    out: Dict[str, Tuple[float, int]] = {}
    for family, zs in by_family.items():
        if AGGREGATION.get(family) == "any":
            # Sidak: the chance that the strongest of k null tells is at least this
            # strong is 1 - (1 - p_min)^k. Reporting the raw max would reward a family
            # simply for having more detectors.
            p_min = _tail_from_z(max(zs))
            out[family] = (_z_from_tail(1.0 - (1.0 - p_min) ** len(zs)), len(zs))
        else:
            out[family] = (sum(zs) / len(zs), len(zs))
    return out


def score_row(
    row: Row,
    catalog: Mapping[str, SignalInfo],
    stats: Mapping[str, SignalStats],
    calibration: Calibration,
    dropped: Iterable[str] = (),
) -> ScoredDoc:
    """Steps 1-4 for one document."""
    if not row.analysable or not row.signals:
        return ScoredDoc(doc_id=row.doc_id, tier=TIER_UNSCORED, score=None, n_words=row.n_words,
                         families={}, evidence=[], label=row.label, reason=row.reason)

    evidence = normalise(row.signals, row.n_words, catalog, stats, calibration, dropped, row.group)
    families: Dict[str, FamilyResult] = {}
    for family, (z, n) in aggregate(evidence).items():
        z = round(z, CALIBRATION_PRECISION)
        # Strict comparisons (Calibration.fires): artifact family scores are heavily tied,
        # and a threshold landing on a tie would otherwise fire on every tied document.
        side = calibration.fires(family, z)
        families[family] = FamilyResult(
            z=z, n_signals=n, fired=side is not None, side=side,
            threshold=calibration.thresholds.get(family),
            threshold_low=calibration.thresholds_low.get(family),
            percentile=calibration.percentile(family, z),
        )

    fired = sum(1 for r in families.values() if r.fired)
    tier = TIER_LIKELY if fired >= 2 else TIER_SUSPECT if fired == 1 else TIER_CLEAN

    score = calibration.score(calibration.statistic({f: r.percentile for f, r in families.items()}))

    evidence.sort(key=lambda e: e.strength, reverse=True)
    return ScoredDoc(doc_id=row.doc_id, tier=tier, score=score, n_words=row.n_words,
                     families=families, evidence=evidence, label=row.label)
