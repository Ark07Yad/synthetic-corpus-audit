"""Metric family contract and registry.

Every contamination signal in chaff is a :class:`Signal`. The scorer (phase 5) never
inspects a metric's internals — it only reads ``value``, ``family`` and ``direction``.
That indirection is what lets the four families be developed and tested independently
and then fused without special cases.

Families are *independent by construction*, which is what makes the corroboration rule
in OBJECTIVE §4.3 meaningful: two signals firing from the same family is one piece of
evidence seen twice, not two pieces of evidence.

Registered families
-------------------
``distributional``  entropy, Zipf, Heaps, lexical diversity, n-gram repetition  (phase 2)
``surprisal``       corpus-internal language-model perplexity variation          (phase 3)
``artifact``        formatting watermarks, hedging, system-prompt echoes         (phase 4)
``reasoning``       redundant reasoning-step loops / state gain                  (phase 4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..context import CorpusContext
from ..document import Document
from ..tokenization import TextView

FAMILY_DISTRIBUTIONAL = "distributional"
FAMILY_SURPRISAL = "surprisal"
FAMILY_ARTIFACT = "artifact"
FAMILY_REASONING = "reasoning"

FAMILIES: Tuple[str, ...] = (
    FAMILY_DISTRIBUTIONAL,
    FAMILY_SURPRISAL,
    FAMILY_ARTIFACT,
    FAMILY_REASONING,
)


@dataclass(frozen=True)
class Signal:
    """One measurement of one document.

    Parameters
    ----------
    name:
        Unique metric identifier, e.g. ``"zipf_tail_slope"``. Stable across versions;
        it appears in report output and in downstream joins.
    value:
        The raw measurement, in the metric's own units. Never pre-normalised — fusion
        owns normalisation so that raw values stay interpretable in ``chaff explain``.
    family:
        One of :data:`FAMILIES`.
    direction:
        ``+1`` if a *higher* value indicates more synthetic text, ``-1`` if a *lower*
        value does. Declared per signal so fusion never has to guess the polarity of
        a metric (OBJECTIVE §4.4).
    description:
        One line, written for a data engineer reading a report — not for a
        statistician. This string is user-facing.
    """

    name: str
    value: float
    family: str
    direction: int
    description: str = ""

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError("unknown family: {0}".format(self.family))
        if self.direction not in (1, -1):
            raise ValueError("direction must be +1 or -1, got {0!r}".format(self.direction))


#: An extractor turns one document into zero or more signals. It must be pure:
#: no corpus-level state, no mutation of the shared ``TextView``. Corpus-level
#: context (phase 3's language model) is injected via a factory instead, so that
#: extractors stay unit-testable on a single document.
Extractor = Callable[[Document, TextView], Sequence[Signal]]

#: A factory receives the corpus context after pass 1 and returns an ordinary pure
#: extractor with that context closed over — or ``None`` when the context lacks what
#: it needs, in which case the extractor is skipped rather than run on bad inputs.
ExtractorFactory = Callable[[CorpusContext], Optional[Extractor]]

#: ``(name, family, extractor)`` — a contextual extractor after binding.
Bound = Tuple[str, str, Extractor]

_REGISTRY: "Dict[str, Tuple[str, Extractor]]" = {}
_CONTEXTUAL: "Dict[str, Tuple[str, ExtractorFactory]]" = {}


def register(name: str, family: str) -> Callable[[Extractor], Extractor]:
    """Decorator registering an extractor under ``name`` within ``family``."""
    if family not in FAMILIES:
        raise ValueError("unknown family: {0}".format(family))

    def decorator(fn: Extractor) -> Extractor:
        _check_unique(name)
        _REGISTRY[name] = (family, fn)
        return fn

    return decorator


def register_contextual(name: str, family: str) -> Callable[[ExtractorFactory], ExtractorFactory]:
    """Decorator registering an extractor *factory* that needs corpus context.

    Its presence is what makes the pipeline run a first pass: if no contextual
    extractor is active, the corpus is read exactly once, as in phases 1 and 2.
    """
    if family not in FAMILIES:
        raise ValueError("unknown family: {0}".format(family))

    def decorator(factory: ExtractorFactory) -> ExtractorFactory:
        _check_unique(name)
        _CONTEXTUAL[name] = (family, factory)
        return factory

    return decorator


def _check_unique(name: str) -> None:
    # One namespace across both tables: a name identifies an extractor in reports
    # regardless of whether it needed a corpus pass.
    if name in _REGISTRY or name in _CONTEXTUAL:
        raise ValueError("extractor already registered: {0}".format(name))


def registered(families: Optional[Iterable[str]] = None) -> List[Tuple[str, str]]:
    """Return ``(name, family)`` for every registered extractor, sorted by name."""
    wanted = set(families) if families else None
    both = [(n, f) for n, (f, _) in _REGISTRY.items()] + [(n, f) for n, (f, _) in _CONTEXTUAL.items()]
    return sorted((n, f) for n, f in both if wanted is None or f in wanted)


def contextual_registered(families: Optional[Iterable[str]] = None) -> List[Tuple[str, str]]:
    """Return ``(name, family)`` for extractors that need a corpus pass."""
    wanted = set(families) if families else None
    return sorted(
        (n, f) for n, (f, _) in _CONTEXTUAL.items() if wanted is None or f in wanted
    )


def bind_contextual(
    context: CorpusContext,
    families: Optional[Iterable[str]] = None,
) -> List[Bound]:
    """Instantiate every active contextual factory against ``context``."""
    wanted = set(families) if families else None
    bound: List[Bound] = []
    for name in sorted(_CONTEXTUAL):
        family, factory = _CONTEXTUAL[name]
        if wanted is not None and family not in wanted:
            continue
        extractor = factory(context)
        if extractor is not None:
            bound.append((name, family, extractor))
    return bound


def extract_signals(
    doc: Document,
    view: TextView,
    families: Optional[Iterable[str]] = None,
    bound: Optional[Sequence[Bound]] = None,
) -> List[Signal]:
    """Run every registered extractor, plus any bound contextual ones, over one
    document. Order is by extractor name across both kinds, so output is stable."""
    wanted = set(families) if families else None
    runnable = [(n, f, fn) for n, (f, fn) in _REGISTRY.items()] + list(bound or [])
    out: List[Signal] = []
    for name, family, fn in sorted(runnable, key=lambda item: item[0]):
        if wanted is not None and family not in wanted:
            continue
        out.extend(fn(doc, view))
    return out


def clear_registry() -> None:
    """Drop all registrations. Test-support only.

    Extractors register at import time, and a module is only imported once per
    process, so a bare clear is **not** recoverable by re-importing. Tests that clear
    the registry must restore it — see :func:`registry_snapshot` and
    :func:`restore_registry` — or every later test in the session runs against an
    empty registry and silently passes for the wrong reason.
    """
    _REGISTRY.clear()
    _CONTEXTUAL.clear()


def registry_snapshot() -> Tuple[Dict[str, Tuple[str, Extractor]], Dict[str, Tuple[str, ExtractorFactory]]]:
    """A copy of both registration tables. Test-support only."""
    return dict(_REGISTRY), dict(_CONTEXTUAL)


def restore_registry(snapshot) -> None:
    """Replace both registration tables from ``snapshot``. Test-support only."""
    plain, contextual = snapshot
    _REGISTRY.clear()
    _REGISTRY.update(plain)
    _CONTEXTUAL.clear()
    _CONTEXTUAL.update(contextual)


# Importing the metric modules is what registers their extractors. Discovery is
# explicit and greppable by design (see ARCHITECTURE.md section 8): there is no plugin
# scanning, so which metrics ran is always answerable by reading this list. The import
# sits at the bottom because each module imports ``Signal`` and ``register`` from this
# one, which must therefore already be defined.
from . import entropy as _entropy  # noqa: E402,F401
from . import ngram as _ngram      # noqa: E402,F401
from . import zipf as _zipf        # noqa: E402,F401
from . import surprisal as _surprisal  # noqa: E402,F401
from . import artifacts as _artifacts  # noqa: E402,F401
from . import reasoning as _reasoning  # noqa: E402,F401
