"""Corpus profiling orchestration.

Stream a corpus, build the shared text view for each document, run every registered
extractor, and report corpus shape. Contamination *scoring* arrives in phase 5.

Two passes, only when needed
----------------------------
Since phase 3 some extractors need corpus-level state (the language model), which is
built by an extra pass over the corpus **before** documents are profiled:

    pass 1   build_context()   stream words only -> NgramLM -> CorpusContext
    pass 2   profile_corpus()  stream documents  -> signals, LM bound in

The corpus is re-read rather than cached (ARCHITECTURE.md §5): holding a crawl dump
in memory would reintroduce the ceiling the streaming reader exists to avoid. If no
contextual extractor is active, pass 1 is skipped and the corpus is read once.

Both passes must see the identical document stream, because leave-one-out scoring
subtracts each document's counts from a model that must contain them. The same
``text_field`` / ``limit`` / ``min_chars`` are passed to both, and the document count
is checked afterwards; the LM raises on any impossible count before that.
"""

from __future__ import annotations

import json
import time
from array import array
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .context import SOURCE_REFERENCE, SOURCE_SELF, CorpusContext
from .corpus_io import iter_documents
from .document import Document
from .language import non_english_reason
from .lm import NgramLM, StreamMismatchError
from .metrics import Bound, Signal, bind_contextual, contextual_registered, extract_signals, registered
from .stats import describe
from .tokenization import MIN_ANALYSABLE_WORDS, TextView, build_view, tokenize_words

#: Per-document rows retained in memory when the caller is not streaming to a file.
#: Profiling a crawl dump must not be bounded by RAM, so the rows are capped and the
#: aggregate statistics are accumulated incrementally instead.
DEFAULT_ROW_CAP = 10_000


@dataclass
class DocProfile:
    """Shape and signals for one document."""

    doc_id: str
    source: str
    n_chars: int
    n_words: int
    n_types: int
    n_sentences: int
    n_lines: int
    analysable: bool
    label: Optional[str] = None
    signals: List[Signal] = field(default_factory=list)
    group: Optional[str] = None
    #: Why an otherwise long-enough document was not analysed (the language guard).
    unscored_reason: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "doc_id": self.doc_id,
            "source": self.source,
            "n_chars": self.n_chars,
            "n_words": self.n_words,
            "n_types": self.n_types,
            "n_sentences": self.n_sentences,
            "n_lines": self.n_lines,
            "analysable": self.analysable,
        }
        if self.label is not None:
            row["label"] = self.label
        if self.unscored_reason is not None:
            row["unscored_reason"] = self.unscored_reason
        for signal in self.signals:
            row[signal.name] = signal.value
        return row


@dataclass
class CorpusProfile:
    """Corpus-level result of a profiling pass."""

    path: str
    n_documents: int = 0
    n_analysable: int = 0
    #: Long enough to analyse but held out by the language guard.
    n_non_english: int = 0
    n_words: int = 0
    vocabulary_size: int = 0
    hapax_count: int = 0
    elapsed_seconds: float = 0.0
    families_active: List[str] = field(default_factory=list)
    metrics_active: List[str] = field(default_factory=list)
    length_summary: Dict[str, float] = field(default_factory=dict)
    type_summary: Dict[str, float] = field(default_factory=dict)
    rows: List[DocProfile] = field(default_factory=list)
    rows_truncated: bool = False
    context_source: Optional[str] = None
    lm_summary: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    #: Reference mode only: share of audited word positions whose bigram the reference
    #: model has seen (R4). ``None`` when the model was built from this corpus.
    audited_coverage: Optional[float] = None

    @property
    def corpus_ttr(self) -> float:
        """Corpus-wide type/token ratio. Length-dependent and therefore only
        comparable between corpora of similar size — reported for orientation,
        never used for scoring (phase 2 uses MTLD and Yule's K instead)."""
        return self.vocabulary_size / self.n_words if self.n_words else 0.0

    @property
    def hapax_ratio(self) -> float:
        """Share of the vocabulary occurring exactly once. A depressed hapax ratio
        is the most direct symptom of the truncated distribution tail described in
        CORE_PROBLEM §3.1."""
        return self.hapax_count / self.vocabulary_size if self.vocabulary_size else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "n_documents": self.n_documents,
            "n_analysable": self.n_analysable,
            "n_non_english": self.n_non_english,
            "n_words": self.n_words,
            "vocabulary_size": self.vocabulary_size,
            "hapax_count": self.hapax_count,
            "hapax_ratio": round(self.hapax_ratio, 6),
            "corpus_ttr": round(self.corpus_ttr, 6),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "families_active": self.families_active,
            "metrics_active": self.metrics_active,
            "length_summary": {k: round(v, 3) for k, v in self.length_summary.items()},
            "type_summary": {k: round(v, 3) for k, v in self.type_summary.items()},
            "context_source": self.context_source,
            "language_model": {k: (round(v, 6) if isinstance(v, float) else v)
                               for k, v in self.lm_summary.items()},
            "notes": self.notes,
            "audited_coverage": None if self.audited_coverage is None else round(self.audited_coverage, 6),
        }


def build_context(
    path: str,
    *,
    text_field: Optional[str] = None,
    limit: Optional[int] = None,
    min_chars: int = 0,
    source: str = SOURCE_SELF,
) -> CorpusContext:
    """Pass 1: stream the corpus once and build the corpus-level context.

    Only words are needed, so documents are tokenized with ``tokenize_words`` directly
    rather than through a full ``TextView`` — the same function ``build_view`` uses,
    which is what guarantees pass 1 counts exactly the words pass 2 will score.

    A self-built model is pruned (lossless under leave-one-out). A reference model is
    not, since the documents it scores were never part of it.
    """
    lm = NgramLM()
    n_documents = 0
    for doc in iter_documents(path, text_field=text_field, limit=limit, min_chars=min_chars):
        lm.observe(tokenize_words(doc.text))
        n_documents += 1
    lm.finalize(prune=(source == SOURCE_SELF))
    return CorpusContext(
        source=source,
        n_documents=n_documents,
        lm=lm,
        reference_path=path if source == SOURCE_REFERENCE else None,
    )


def profile_document(
    doc: Document,
    *,
    families: Optional[Iterable[str]] = None,
    view: Optional[TextView] = None,
    bound: Optional[Sequence[Bound]] = None,
    language_guard: bool = True,
) -> DocProfile:
    """Build the text view for one document and run the registered extractors,
    plus any contextual extractors already bound to a corpus context.

    With ``language_guard`` (the default), a document with positive evidence of being in
    another language is left unanalysed, with the reason recorded: every reference
    distribution chaff ships is English (see :mod:`chaff.language`)."""
    view = view if view is not None else build_view(doc.text)
    reason = non_english_reason(view.words) if language_guard and view.is_analysable else None
    analysable = view.is_analysable and reason is None
    signals = extract_signals(doc, view, families=families, bound=bound) if analysable else []
    return DocProfile(
        doc_id=doc.doc_id,
        source=doc.source,
        n_chars=view.n_chars,
        n_words=view.n_words,
        n_types=view.n_types,
        n_sentences=len(view.sentences),
        n_lines=len(view.lines),
        analysable=analysable,
        label=doc.label,
        signals=list(signals),
        unscored_reason=reason,
    )


def profile_corpus(
    path: str,
    *,
    text_field: Optional[str] = None,
    limit: Optional[int] = None,
    min_chars: int = 0,
    families: Optional[Iterable[str]] = None,
    row_cap: int = DEFAULT_ROW_CAP,
    on_row: Optional[Any] = None,
    reference: Optional[str] = None,
    context: Optional[CorpusContext] = None,
    group_field: Optional[str] = None,
    language_guard: bool = True,
) -> CorpusProfile:
    """Stream a corpus and profile every document in it.

    Parameters
    ----------
    on_row:
        Optional callable invoked with each :class:`DocProfile` as it is produced.
        The CLI uses it to stream per-document rows straight to disk so that a
        large corpus never has to be held in memory.
    reference:
        Path to a trusted corpus. The language model is built from it instead of from
        the audited corpus, and documents are scored against it directly. Strictly
        better when one is available, and it also removes pass 1 over ``path``.
    context:
        A prebuilt context, bypassing pass 1 entirely. Mostly for tests and for
        callers profiling the same corpus repeatedly.
    language_guard:
        Leave documents that are evidently not English unanalysed (the default).
    """
    started = time.time()
    profile = CorpusProfile(path=path)

    wants_context = bool(contextual_registered(families))
    if context is None and wants_context:
        if reference is not None:
            context = build_context(reference, text_field=text_field, source=SOURCE_REFERENCE)
        elif path == "-":
            profile.notes.append(
                "surprisal family skipped: stdin cannot be read twice, and the corpus "
                "language model needs its own pass. Pass a file, or supply --reference."
            )
        else:
            context = build_context(path, text_field=text_field, limit=limit, min_chars=min_chars)

    bound: List[Bound] = bind_contextual(context, families) if context is not None else []
    if context is not None and context.lm is not None:
        profile.context_source = context.source
        profile.lm_summary = context.lm.summary()

    bound_names = {name for name, _, _ in bound}
    contextual_names = {name for name, _ in contextual_registered(families)}
    active = [(name, family) for name, family in registered(families)
              if name not in contextual_names or name in bound_names]
    profile.families_active = sorted({family for _, family in active})
    profile.metrics_active = [name for name, _ in active]

    vocabulary: "Counter[str]" = Counter()
    # Typed arrays, not lists of floats: 8 bytes per document instead of ~32.
    lengths = array("d")
    type_counts = array("d")
    measure_coverage = (context is not None and context.lm is not None
                        and context.source == SOURCE_REFERENCE)
    hits = positions = 0

    for doc in iter_documents(
        path, text_field=text_field, limit=limit, min_chars=min_chars
    ):
        view = build_view(doc.text)
        row = profile_document(doc, families=families, view=view, bound=bound,
                               language_guard=language_guard)
        if group_field is not None:
            value = doc.meta.get(group_field)
            row.group = None if value is None else str(value)

        profile.n_documents += 1
        profile.n_words += view.n_words
        if row.analysable:
            profile.n_analysable += 1
        elif row.unscored_reason is not None:
            profile.n_non_english += 1
        vocabulary.update(view.words)
        if measure_coverage:
            h, t = context.lm.bigram_hits(view.words)
            hits += h
            positions += t
        lengths.append(float(view.n_words))
        type_counts.append(float(view.n_types))

        if on_row is not None:
            on_row(row)
        if len(profile.rows) < row_cap:
            profile.rows.append(row)
        else:
            profile.rows_truncated = True

    if context is not None and context.leave_one_out and context.n_documents != profile.n_documents:
        raise StreamMismatchError(
            "pass 1 saw {0} documents but pass 2 saw {1}; the corpus changed between "
            "passes, so leave-one-out scores are invalid".format(
                context.n_documents, profile.n_documents)
        )

    if measure_coverage and positions:
        profile.audited_coverage = hits / float(positions)

    profile.vocabulary_size = len(vocabulary)
    profile.hapax_count = sum(1 for count in vocabulary.values() if count == 1)
    profile.length_summary = describe(lengths)
    profile.type_summary = describe(type_counts)
    profile.elapsed_seconds = time.time() - started
    return profile


def non_english_note(profile: CorpusProfile) -> Optional[str]:
    """Say how many documents the language guard held out."""
    if not profile.n_non_english:
        return None
    return ("{0} of {1} documents are evidently not English and were left unscored: chaff's "
            "reference distributions are English, so their scores would be meaningless "
            "(--allow-non-english scores them anyway).").format(profile.n_non_english, profile.n_documents)


def short_document_note(profile: CorpusProfile) -> Optional[str]:
    """Warn when most of a corpus is too short to analyse distributionally."""
    skipped = profile.n_documents - profile.n_analysable - profile.n_non_english
    if profile.n_documents and skipped / profile.n_documents > 0.25:
        return (
            "{0} of {1} documents are under {2} words and will not be scored: "
            "distributional metrics are dominated by sampling noise at that length."
        ).format(skipped, profile.n_documents, MIN_ANALYSABLE_WORDS)
    return None


# --------------------------------------------------------------------------- fusion
#
# Fusion runs over per-document signal *rows* spilled to disk during pass 2, never over
# the corpus (R3, D16): contextual signals do not exist until pass 2, and a row is ~20
# floats against thousands of words. The corpus is therefore still read exactly twice
# (three times over a reference corpus in reference mode, which also profiles itself).

import heapq
import os
import tempfile

from .scoring import (
    TIER_LIKELY,
    TIERS,
    Calibration,
    Row,
    ScoredDoc,
    SignalInfo,
    build_reference_stats,
    load_calibration,
    score_row,
)

#: Documents kept in memory for the report; the full set is streamed to scores.jsonl.
DEFAULT_TOP_N = 25


def collect_rows(
    path: str,
    spill,
    *,
    text_field: Optional[str] = None,
    limit: Optional[int] = None,
    min_chars: int = 0,
    context: Optional[CorpusContext] = None,
    reference: Optional[str] = None,
    group_field: Optional[str] = None,
    language_guard: bool = True,
) -> "tuple":
    """Profile a corpus, streaming one :class:`Row` per document to ``spill``.

    Returns ``(CorpusProfile, catalog)``, where the catalog maps every signal name that
    was emitted to its family, direction and description — read off the ``Signal``
    objects themselves, so fusion never keeps a second copy of that metadata.
    """
    catalog: Dict[str, SignalInfo] = {}

    def keep(doc: DocProfile) -> None:
        spill.write(Row(doc_id=doc.doc_id, n_words=doc.n_words, analysable=doc.analysable,
                        signals={s.name: s.value for s in doc.signals},
                        label=doc.label, group=doc.group,
                        reason=doc.unscored_reason).to_json() + "\n")
        for s in doc.signals:
            if s.name not in catalog:
                catalog[s.name] = SignalInfo(s.family, s.direction, s.description)

    profile = profile_corpus(path, text_field=text_field, limit=limit, min_chars=min_chars,
                             on_row=keep, row_cap=0, context=context, reference=reference,
                             group_field=group_field, language_guard=language_guard)
    return profile, catalog


def _read_rows(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield Row.from_json(line)


def coverage_drops(profile: CorpusProfile) -> "tuple":
    """R4: in reference mode, withhold surprisal spread signals corpus-wide when the
    reference model covers the *audited* text too thinly — measured on the audited
    corpus, since a reference model can be well estimated and still out of domain."""
    from .metrics.surprisal import MIN_COVERAGE_SENTENCE_SPREAD, MIN_COVERAGE_TOKEN_SPREAD
    c = profile.audited_coverage
    if c is None:
        return set(), None
    if c < MIN_COVERAGE_TOKEN_SPREAD:
        return ({"surprisal_std", "sentence_surprisal_std"},
                "reference model covers only {0:.0%} of the audited text's bigrams (needs {1:.0%}); "
                "surprisal spread signals were withheld from fusion".format(c, MIN_COVERAGE_TOKEN_SPREAD))
    if c < MIN_COVERAGE_SENTENCE_SPREAD:
        return ({"sentence_surprisal_std"},
                "reference model covers {0:.0%} of the audited text's bigrams (needs {1:.0%} for "
                "sentence-level spread); that signal was withheld".format(c, MIN_COVERAGE_SENTENCE_SPREAD))
    return set(), None


@dataclass
class ScoreRun:
    """Corpus-level result of ``chaff score``. Per-document results are streamed to
    ``scores_path``; this holds only what a report needs."""

    meta: Dict[str, Any]
    top: List[ScoredDoc]


def score_corpus(
    path: str,
    *,
    text_field: Optional[str] = None,
    limit: Optional[int] = None,
    min_chars: int = 0,
    reference: Optional[str] = None,
    calibration: Optional[Calibration] = None,
    scores_path: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N,
    stratify_by: Optional[str] = None,
    language_guard: bool = True,
) -> ScoreRun:
    """Profile, fuse and tier every document in a corpus."""
    started = time.time()
    calibration = calibration or load_calibration()
    notes: List[str] = []

    with tempfile.TemporaryDirectory(prefix="chaff-") as tmp:
        audited_rows = os.path.join(tmp, "rows.jsonl")
        if reference is not None:
            ref_context = build_context(reference, text_field=text_field, source=SOURCE_REFERENCE)
            reference_rows = os.path.join(tmp, "reference_rows.jsonl")
            # The reference corpus is profiled leave-one-out against its own (unpruned)
            # model, so its rows describe how *reference* documents score — the null that
            # audited documents, scored against the same model, are compared with.
            self_context = CorpusContext(source=SOURCE_SELF, n_documents=ref_context.n_documents,
                                         lm=ref_context.lm)
            with open(reference_rows, "w", encoding="utf-8") as fh:
                _, ref_catalog = collect_rows(reference, fh, text_field=text_field, context=self_context,
                                              group_field=stratify_by, language_guard=language_guard)
            with open(audited_rows, "w", encoding="utf-8") as fh:
                profile, catalog = collect_rows(path, fh, text_field=text_field, limit=limit,
                                                min_chars=min_chars, context=ref_context,
                                                group_field=stratify_by, language_guard=language_guard)
            catalog = dict(ref_catalog, **catalog)
            stats_source = reference_rows
        else:
            with open(audited_rows, "w", encoding="utf-8") as fh:
                profile, catalog = collect_rows(path, fh, text_field=text_field, limit=limit,
                                                min_chars=min_chars, group_field=stratify_by,
                                                language_guard=language_guard)
            stats_source = audited_rows

        stats = build_reference_stats(_read_rows(stats_source), catalog, stratify=stratify_by is not None)
        if stratify_by is not None:
            groups = sorted({k[0] for k in stats if isinstance(k, tuple)})
            notes.append("normalised within groups of '{0}': {1}{2}".format(
                stratify_by, ", ".join(groups[:12]) or "none large enough",
                " (+{0} more)".format(len(groups) - 12) if len(groups) > 12 else "")
                + "; groups under {0} documents use corpus-wide stats".format(50))
        dropped, coverage_note = coverage_drops(profile)
        notes.extend(profile.notes)
        language_note = non_english_note(profile)
        if language_note:
            notes.append(language_note)
        if coverage_note:
            notes.append(coverage_note)
        reference_n = max((s.n for k, s in stats.items() if not isinstance(k, tuple)), default=0)
        if reference_n < 50:
            notes.append(
                "only {0} scoreable documents in the normalisation reference: corpus-relative "
                "z-scores rest on a tiny sample. Treat tiers as illustrative, or pass --reference "
                "with a larger trusted corpus.".format(reference_n))
        cal_strat = calibration.meta.get("stratify_by")
        if cal_strat != stratify_by:
            notes.append(
                "the calibration was built {0} but this run is {1}: its thresholds assume the "
                "same normalisation, so tier rates will not match its held-out figures".format(
                    "with --stratify-by {0}".format(cal_strat) if cal_strat else "without --stratify-by",
                    "stratified by {0}".format(stratify_by) if stratify_by else "unstratified"))
        unknown = sorted(n for n, info in catalog.items()
                         if info.family == "artifact" and n not in calibration.human.counts)
        if unknown:
            notes.append("artifact signals missing from the calibration were ignored: " + ", ".join(unknown))

        tier_counts = {t: 0 for t in TIERS}
        fired_counts: Dict[str, int] = {}
        histogram = [0] * 20
        labels: Dict[str, Dict[str, int]] = {}
        top: List = []
        out = open(scores_path, "w", encoding="utf-8") if scores_path else None
        try:
            for i, row in enumerate(_read_rows(audited_rows)):
                scored = score_row(row, catalog, stats, calibration, dropped)
                tier_counts[scored.tier] += 1
                for family in scored.fired:
                    fired_counts[family] = fired_counts.get(family, 0) + 1
                if scored.score is not None:
                    histogram[min(19, int(scored.score // 5))] += 1
                    item = (scored.score, -i, scored)
                    if len(top) < top_n:
                        heapq.heappush(top, item)
                    else:
                        heapq.heappushpop(top, item)
                if scored.label is not None:
                    bucket = labels.setdefault(scored.label, {t: 0 for t in TIERS})
                    bucket[scored.tier] += 1
                if out is not None:
                    out.write(json.dumps(scored.to_dict(), ensure_ascii=False) + "\n")
        finally:
            if out is not None:
                out.close()

    scored_total = sum(v for k, v in tier_counts.items() if k != "UNSCORED")
    meta: Dict[str, Any] = {
        "chaff_version": _version(),
        "corpus": path,
        "reference": reference,
        "normalisation": ("reference corpus" if reference else "audited corpus (corpus-relative)")
                         + (", stratified by '{0}'".format(stratify_by) if stratify_by else ""),
        "documents": profile.n_documents,
        "scored": scored_total,
        "tiers": tier_counts,
        "families_fired": fired_counts,
        "family_thresholds": calibration.thresholds,
        "score_histogram": histogram,
        "labels": labels,
        "withheld_signals": sorted(dropped),
        "profile": profile.to_dict(),
        "calibration": calibration.meta,
        "notes": notes,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    ranked = [item[2] for item in sorted(top, reverse=True)]
    if scores_path:
        with open(scores_path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(dict(meta, top=[d.to_dict() for d in ranked]), fh, indent=2, ensure_ascii=False)
    return ScoreRun(meta=meta, top=ranked)


def _version() -> str:
    from . import __version__
    return __version__
