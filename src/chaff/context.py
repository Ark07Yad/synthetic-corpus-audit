"""Corpus-level state shared with contextual extractors.

Pass 1 builds a :class:`CorpusContext`; pass 2 hands it to every contextual extractor
factory, which binds it into an ordinary pure ``Extractor``. This module deliberately
knows nothing about *how* the context is built — that needs the corpus reader, which
lives a layer above (``pipeline.py``). Keeping the dataclass here, importing only the
language model, is what lets ``metrics/`` depend on it without reaching sideways into
corpus I/O (see ARCHITECTURE.md §1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .lm import NgramLM

#: The context was built from the corpus being audited; documents are scored
#: leave-one-out against it.
SOURCE_SELF = "self"

#: The context was built from a separate trusted corpus (``--reference``); documents
#: are scored against it directly, since they were never part of it.
SOURCE_REFERENCE = "reference"


@dataclass
class CorpusContext:
    """Everything a contextual extractor may know about the corpus as a whole."""

    source: str
    n_documents: int
    lm: Optional[NgramLM] = None
    reference_path: Optional[str] = None

    @property
    def leave_one_out(self) -> bool:
        """Whether each scored document is itself inside the model's training data."""
        return self.source == SOURCE_SELF
