"""Build a calibration from a trusted human corpus (``chaff calibrate``).

The shipped calibration was learned on technical reference text (Python stdlib
docstrings and man pages). Phase 6 measured how badly that transfers: 0.92% of clean
technical documents reach ``LIKELY_SYNTHETIC``, but 0-5.1% across other genres, and
11-38% are flagged. A user auditing news, forum posts or scientific abstracts needs
thresholds learned on *their* genre. This module is that procedure, moved out of
``benchmarks/human_baseline.py`` so it is part of the tool rather than a script beside
it. The benchmark now calls it, and reproduces the shipped calibration exactly.

The procedure, unchanged from phases 5 and 6:

1. Profile the trusted corpus and split it deterministically into halves A and B by a
   hash of each document id. Half A fits everything; half B only evaluates.
2. Normalise continuous signals against the whole corpus (as ``chaff score`` does);
   build the artifact human reference from half A only.
3. Take each family's null distribution from half A. Search one per-family alpha — the
   largest on the grid whose half-A ``LIKELY_SYNTHETIC`` rate stays at or under the
   target — splitting it across both tails for two-sided families.
4. Calibrate the 0-100 score against its own half-A null.
5. Report the held-out half-B rates, the dependence inflation, and tier/score coherence.

The input must be text you trust to be human. Calibrating on a contaminated corpus
teaches chaff that contamination is normal, and every threshold will be too lax.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import __version__
from .metrics import FAMILIES
from .pipeline import collect_rows
from .scoring import (
    AGGREGATION,
    BASIS_HUMAN,
    CALIBRATION_PRECISION,
    NORMALISATION,
    SIDE_TWO,
    SIDEDNESS,
    Calibration,
    HumanReference,
    Row,
    SignalInfo,
    aggregate,
    build_reference_stats,
    normalise,
)

#: At most this share of clean documents may reach LIKELY_SYNTHETIC on half A.
DEFAULT_LIKELY_TARGET = 0.01
ALPHA_GRID = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)
NULL_QUANTILES = 201

#: Below this many analysable documents a calibration is refused: at alpha 0.02 a half
#: of 100 documents puts two documents in each family's tail.
MIN_DOCUMENTS = 200
#: Below this, a calibration is built but flagged as noisy.
RECOMMENDED_DOCUMENTS = 1000

RULE = ("one-sided families fire when z > the (1-alpha) quantile of half-A family z; "
        "two-sided families fire outside the alpha/2 and (1-alpha/2) quantiles (strict)")


class CalibrationError(RuntimeError):
    """The corpus cannot support a calibration (too small, or no alpha meets the target)."""


def half_of(doc_id: str) -> str:
    """Deterministic A/B split, so fitting and evaluation never share documents. The
    same rule ``benchmarks/human_baseline.py`` has always used."""
    return "A" if int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "B"


def _quantiles(sorted_values: Sequence[float], n: int) -> List[float]:
    last = len(sorted_values) - 1
    return [round(sorted_values[int(round(i * last / (n - 1)))], 6) for i in range(n)]


def _threshold(sorted_values: Sequence[float], alpha: float) -> float:
    """Value T with at most ``alpha`` of the null strictly above it."""
    k = max(0, int(math.ceil((1.0 - alpha) * len(sorted_values))) - 1)
    return sorted_values[k]


def _threshold_low(sorted_values: Sequence[float], alpha: float) -> float:
    """Value T with at most ``alpha`` of the null strictly below it."""
    k = min(len(sorted_values) - 1, int(math.floor(alpha * len(sorted_values))))
    return sorted_values[k]


def _fires(fam: str, z: float, high: Mapping[str, float], low: Mapping[str, float]) -> bool:
    return z > high[fam] or (fam in low and z < low[fam])


def _tier_rates(family_z: Sequence[Mapping[str, float]], high, low):
    fires: Dict[str, int] = defaultdict(int)
    suspect = likely = 0
    for fz in family_z:
        n = 0
        for fam, z in fz.items():
            if _fires(fam, z, high, low):
                fires[fam] += 1
                n += 1
        suspect += n == 1
        likely += n >= 2
    total = float(len(family_z)) or 1.0
    return {f: fires[f] / total for f in high}, suspect / total, likely / total


def _independent_likely(marginals: Mapping[str, float]) -> float:
    """P(at least two families fire) if they fired independently."""
    ps = list(marginals.values())
    none = 1.0
    for p in ps:
        none *= 1.0 - p
    exactly_one = 0.0
    for i, p in enumerate(ps):
        others = 1.0
        for j, q in enumerate(ps):
            if j != i:
                others *= 1.0 - q
        exactly_one += p * others
    return 1.0 - none - exactly_one


def _thresholds_for(null: Mapping[str, Sequence[float]], alpha: float):
    # Two-sided families split alpha between the tails, so every family fires on at
    # most alpha of clean documents whichever way it is sided.
    high, low = {}, {}
    for f, v in null.items():
        if SIDEDNESS.get(f) == SIDE_TWO:
            high[f], low[f] = _threshold(v, alpha / 2.0), _threshold_low(v, alpha / 2.0)
        else:
            high[f] = _threshold(v, alpha)
    return high, low


@dataclass
class CalibrationRun:
    """A calibration plus everything needed to explain how it was reached."""

    calibration: Calibration
    alpha_search: List[Tuple[float, float, float, bool]]   # alpha, SUSPECT, LIKELY, meets target
    family_z: Dict[str, Dict[str, float]]
    halves: Dict[str, str]
    notes: List[str] = field(default_factory=list)


def calibrate_rows(
    rows: Iterable[Row],
    catalog: Mapping[str, SignalInfo],
    *,
    likely_target: float = DEFAULT_LIKELY_TARGET,
    stratify: bool = False,
    source: str = "",
    stratify_by: Optional[str] = None,
) -> CalibrationRun:
    """Calibrate from already-profiled rows. ``Row.group`` drives both stratified
    normalisation (when ``stratify``) and the per-group held-out breakdown."""
    rows = [r for r in rows if r.analysable]
    notes: List[str] = []
    if len(rows) < MIN_DOCUMENTS:
        raise CalibrationError(
            "only {0} analysable documents; a calibration needs at least {1} (and {2}+ is "
            "recommended), or each family's tail rests on a handful of documents".format(
                len(rows), MIN_DOCUMENTS, RECOMMENDED_DOCUMENTS))
    if len(rows) < RECOMMENDED_DOCUMENTS:
        notes.append("{0} analysable documents: fewer than the recommended {1}, so thresholds "
                     "and held-out rates are noisy".format(len(rows), RECOMMENDED_DOCUMENTS))

    halves = {r.doc_id: half_of(r.doc_id) for r in rows}
    groups = {r.doc_id: r.group for r in rows}

    # Continuous signals are normalised against the whole trusted corpus, exactly as
    # `chaff score` normalises against a whole audited corpus. Artifact signals use half
    # A only, so half-B documents are scored out-of-sample against the human reference.
    stats = build_reference_stats(rows, catalog, stratify=stratify)
    artifact_values: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if halves[r.doc_id] != "A":
            continue
        for name, value in r.signals.items():
            info = catalog.get(name)
            if info and NORMALISATION.get(info.family) == BASIS_HUMAN:
                artifact_values[name].append(value)
    human = HumanReference.from_values(artifact_values)
    provisional = Calibration(thresholds={}, family_null={}, human=human)

    family_z: Dict[str, Dict[str, float]] = {}
    for r in rows:
        ev = normalise(r.signals, r.n_words, catalog, stats, provisional, group=r.group if stratify else None)
        # Rounded exactly as chaff.scoring.score_row rounds before comparing, so the
        # thresholds and null quantiles written here match what they are compared with.
        family_z[r.doc_id] = {f: round(z, CALIBRATION_PRECISION) for f, (z, _n) in aggregate(ev).items()}
    a_ids = [d for d in family_z if halves[d] == "A"]
    b_ids = [d for d in family_z if halves[d] == "B"]
    # The id hash splits any ordinary corpus roughly in half, but not every corpus: one
    # already filtered by this same rule lands entirely in one half. Refuse rather than
    # fit on everything and "evaluate" on nothing.
    smaller = min(len(a_ids), len(b_ids))
    if smaller < MIN_DOCUMENTS // 4:
        raise CalibrationError(
            "the document-id split put {0} documents in half A and {1} in half B; both halves "
            "need at least {2}. Was this corpus already filtered by sha1(id) % 2? Give the "
            "documents fresh ids.".format(len(a_ids), len(b_ids), MIN_DOCUMENTS // 4))
    null = {f: sorted(family_z[d][f] for d in a_ids if f in family_z[d]) for f in FAMILIES}
    null = {f: v for f, v in null.items() if v}

    search: List[Tuple[float, float, float, bool]] = []
    chosen = None
    for alpha in ALPHA_GRID:
        high, low = _thresholds_for(null, alpha)
        _, suspect, likely = _tier_rates([family_z[d] for d in a_ids], high, low)
        ok = likely <= likely_target
        search.append((alpha, suspect, likely, ok))
        if ok:
            chosen = (alpha, high, low)
    if chosen is None:
        raise CalibrationError(
            "no alpha on the grid keeps LIKELY_SYNTHETIC at or under {0:.1%} on half A: even "
            "alpha {1} gives {2:.1%}. The corpus may be too heterogeneous — try --stratify-by — "
            "or not entirely human.".format(likely_target, ALPHA_GRID[0], search[0][2]))
    alpha, high, low = chosen

    def evaluate(ids):
        marg, suspect, likely = _tier_rates([family_z[d] for d in ids], high, low)
        return {"documents": len(ids), "family_fire_rate": {k: round(v, 5) for k, v in marg.items()},
                "suspect_rate": round(suspect, 5), "likely_rate": round(likely, 5),
                "likely_rate_if_independent": round(_independent_likely(marg), 5)}

    evaluation = {"A": evaluate(a_ids), "B": evaluate(b_ids)}
    for g in sorted({groups[d] for d in b_ids if groups[d] is not None}):
        evaluation["B_" + g] = evaluate([d for d in b_ids if groups[d] == g])

    meta = {
        "created": datetime.date.today().isoformat(),
        "chaff_version": __version__,
        "python": "{0}.{1}.{2}".format(*sys.version_info[:3]),
        "baseline": source,
        "documents": {"A": len(a_ids), "B": len(b_ids)},
        "likely_target": likely_target,
        "alpha": alpha,
        "rule": RULE,
        "sidedness": SIDEDNESS,
        "normalisation": NORMALISATION,
        "aggregation": AGGREGATION,
        "evaluation": evaluation,
    }
    if stratify_by:
        meta["stratify_by"] = stratify_by
    cal = Calibration(thresholds={f: round(t, 6) for f, t in high.items()},
                      family_null={f: _quantiles(v, NULL_QUANTILES) for f, v in null.items()},
                      human=human, signals=sorted(catalog), meta=meta,
                      thresholds_low={f: round(t, 6) for f, t in low.items()})

    # The score's own null: the evidence statistic over half A, so a clean document's
    # score is uniform on 0-100 rather than wherever the raw statistic happens to sit.
    def stat(doc_id: str) -> float:
        return cal.statistic({f: cal.percentile(f, z) for f, z in family_z[doc_id].items()})

    cal.score_null = _quantiles(sorted(round(stat(d), CALIBRATION_PRECISION) for d in a_ids), NULL_QUANTILES)
    b_score = {d: cal.score(stat(d)) for d in b_ids}
    ordered = sorted(b_score.values())
    evaluation["B"]["score_median"] = round(ordered[len(ordered) // 2], 2)
    evaluation["B"]["score_p95"] = round(ordered[int(0.95 * len(ordered))], 2)
    # Coherence: a document that fires a family should outrank a typical clean one.
    by_tier: Dict[str, List[float]] = defaultdict(list)
    for d in b_ids:
        n = sum(1 for f, z in family_z[d].items() if _fires(f, z, high, low))
        by_tier["LIKELY" if n >= 2 else "SUSPECT" if n == 1 else "CLEAN"].append(b_score[d])
    evaluation["B"]["score_by_tier_min_median"] = {
        t: [round(min(v), 1), round(sorted(v)[len(v) // 2], 1)] for t, v in by_tier.items()}
    if evaluation["B"]["likely_rate"] > 1.5 * likely_target:
        notes.append(
            "held-out half B reaches LIKELY_SYNTHETIC at {0:.2%}, above the {1:.1%} target: the "
            "thresholds overfit half A. More documents, or --stratify-by on a mixed-genre "
            "corpus, will bring the held-out rate closer.".format(evaluation["B"]["likely_rate"], likely_target))

    return CalibrationRun(calibration=cal, alpha_search=search, family_z=family_z,
                          halves=halves, notes=notes)


def calibrate_corpus(
    path: str,
    *,
    text_field: Optional[str] = None,
    limit: Optional[int] = None,
    min_chars: int = 0,
    likely_target: float = DEFAULT_LIKELY_TARGET,
    stratify_by: Optional[str] = None,
    breakdown_by: Optional[str] = None,
    source: Optional[str] = None,
    language_guard: bool = True,
) -> CalibrationRun:
    """Profile a trusted human corpus and calibrate on it.

    ``stratify_by`` normalises within groups of a record field — use it here exactly when
    you will also pass it to ``chaff score``. ``breakdown_by`` only reports held-out rates
    per group, without changing the normalisation.
    """
    field_name = stratify_by or breakdown_by
    spill = io.StringIO()
    _, catalog = collect_rows(path, spill, text_field=text_field, limit=limit,
                              min_chars=min_chars, group_field=field_name,
                              language_guard=language_guard)
    rows = [Row.from_json(line) for line in spill.getvalue().splitlines() if line.strip()]
    return calibrate_rows(rows, catalog, likely_target=likely_target, stratify=stratify_by is not None,
                          source=source if source is not None else path, stratify_by=stratify_by)
