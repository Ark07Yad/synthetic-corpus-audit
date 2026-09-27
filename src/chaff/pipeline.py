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

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .context import SOURCE_REFERENCE, SOURCE_SELF, CorpusContext
from .corpus_io import iter_documents
from .document import Document
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
    context_source: Optional[str] = None
    lm_summary: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

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
            "context_source": self.context_source,
            "language_model": {k: (round(v, 6) if isinstance(v, float) else v)
                               for k, v in self.lm_summary.items()},
            "notes": self.notes,
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
) -> DocProfile:
    """Build the text view for one document and run the registered extractors,
    plus any contextual extractors already bound to a corpus context."""
    view = view if view is not None else build_view(doc.text)
    signals = (
        extract_signals(doc, view, families=families, bound=bound)
        if view.is_analysable else []
    )
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
    reference: Optional[str] = None,
    context: Optional[CorpusContext] = None,
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
    lengths: List[float] = []
    type_counts: List[float] = []

    for doc in iter_documents(
        path, text_field=text_field, limit=limit, min_chars=min_chars
    ):
        view = build_view(doc.text)
        row = profile_document(doc, families=families, view=view, bound=bound)

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

    if context is not None and context.leave_one_out and context.n_documents != profile.n_documents:
        raise StreamMismatchError(
            "pass 1 saw {0} documents but pass 2 saw {1}; the corpus changed between "
            "passes, so leave-one-out scores are invalid".format(
                context.n_documents, profile.n_documents)
        )

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
