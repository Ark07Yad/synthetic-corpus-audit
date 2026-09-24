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

_REGISTRY: "Dict[str, Tuple[str, Extractor]]" = {}


def register(name: str, family: str) -> Callable[[Extractor], Extractor]:
    """Decorator registering an extractor under ``name`` within ``family``."""
    if family not in FAMILIES:
        raise ValueError("unknown family: {0}".format(family))

    def decorator(fn: Extractor) -> Extractor:
        if name in _REGISTRY:
            raise ValueError("extractor already registered: {0}".format(name))
        _REGISTRY[name] = (family, fn)
        return fn

    return decorator


def registered(families: Optional[Iterable[str]] = None) -> List[Tuple[str, str]]:
    """Return ``(name, family)`` for registered extractors, sorted by name."""
    wanted = set(families) if families else None
    return sorted(
        (name, family)
        for name, (family, _) in _REGISTRY.items()
        if wanted is None or family in wanted
    )


def extract_signals(
    doc: Document,
    view: TextView,
    families: Optional[Iterable[str]] = None,
) -> List[Signal]:
    """Run every registered extractor over one document and collect its signals."""
    wanted = set(families) if families else None
    out: List[Signal] = []
    for name in sorted(_REGISTRY):
        family, fn = _REGISTRY[name]
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


def registry_snapshot() -> "Dict[str, Tuple[str, Extractor]]":
    """A copy of the current registrations. Test-support only."""
    return dict(_REGISTRY)


def restore_registry(snapshot: "Dict[str, Tuple[str, Extractor]]") -> None:
    """Replace the registry with ``snapshot``. Test-support only."""
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


# Importing the metric modules is what registers their extractors. Discovery is
# explicit and greppable by design (see ARCHITECTURE.md section 8): there is no plugin
# scanning, so which metrics ran is always answerable by reading this list. The import
# sits at the bottom because each module imports ``Signal`` and ``register`` from this
# one, which must therefore already be defined.
from . import entropy as _entropy  # noqa: E402,F401
from . import ngram as _ngram      # noqa: E402,F401
from . import zipf as _zipf        # noqa: E402,F401
