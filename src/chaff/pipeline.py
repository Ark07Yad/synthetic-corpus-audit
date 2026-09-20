"""Corpus profiling orchestration.

Phase 1 scope: stream a corpus, build the shared text view for each document, run
whatever metric extractors are registered, and report corpus shape. Contamination
*scoring* arrives in phase 5 and plugs in here without changing this structure —
the per-document signal list is already threaded through.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .corpus_io import iter_documents
from .document import Document
from .metrics import Signal, extract_signals, registered
from .stats import describe
from .tokenization import MIN_ANALYSABLE_WORDS, TextView, build_view

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
        for signal in self.signals:
            row[signal.name] = signal.value
        return row


@dataclass
class CorpusProfile:
    """Corpus-level result of a profiling pass."""

    path: str
    n_documents: int = 0
    n_analysable: int = 0
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
        }


def profile_document(
    doc: Document,
    *,
    families: Optional[Iterable[str]] = None,
    view: Optional[TextView] = None,
) -> DocProfile:
    """Build the text view for one document and run the registered extractors."""
    view = view if view is not None else build_view(doc.text)
    signals = extract_signals(doc, view, families=families) if view.is_analysable else []
    return DocProfile(
        doc_id=doc.doc_id,
        source=doc.source,
        n_chars=view.n_chars,
        n_words=view.n_words,
        n_types=view.n_types,
        n_sentences=len(view.sentences),
        n_lines=len(view.lines),
        analysable=view.is_analysable,
        label=doc.label,
        signals=list(signals),
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
) -> CorpusProfile:
    """Stream a corpus and profile every document in it.

    Parameters
    ----------
    on_row:
        Optional callable invoked with each :class:`DocProfile` as it is produced.
        The CLI uses it to stream per-document rows straight to disk so that a
        large corpus never has to be held in memory.
    """
    started = time.time()
    active = registered(families)
    profile = CorpusProfile(
        path=path,
        families_active=sorted({family for _, family in active}),
        metrics_active=[name for name, _ in active],
    )

    vocabulary: "Counter[str]" = Counter()
    lengths: List[float] = []
    type_counts: List[float] = []

    for doc in iter_documents(
        path, text_field=text_field, limit=limit, min_chars=min_chars
    ):
        view = build_view(doc.text)
        row = profile_document(doc, families=families, view=view)

        profile.n_documents += 1
        profile.n_words += view.n_words
        if row.analysable:
            profile.n_analysable += 1
        vocabulary.update(view.words)
        lengths.append(float(view.n_words))
        type_counts.append(float(view.n_types))

        if on_row is not None:
            on_row(row)
        if len(profile.rows) < row_cap:
            profile.rows.append(row)
        else:
            profile.rows_truncated = True

    profile.vocabulary_size = len(vocabulary)
    profile.hapax_count = sum(1 for count in vocabulary.values() if count == 1)
    profile.length_summary = describe(lengths)
    profile.type_summary = describe(type_counts)
    profile.elapsed_seconds = time.time() - started
    return profile


def short_document_note(profile: CorpusProfile) -> Optional[str]:
    """Warn when most of a corpus is too short to analyse distributionally."""
    skipped = profile.n_documents - profile.n_analysable
    if profile.n_documents and skipped / profile.n_documents > 0.25:
        return (
            "{0} of {1} documents are under {2} words and will not be scored: "
            "distributional metrics are dominated by sampling noise at that length."
        ).format(skipped, profile.n_documents, MIN_ANALYSABLE_WORDS)
    return None
