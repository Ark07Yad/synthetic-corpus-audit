"""The unit of analysis: one document from a corpus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """A single corpus document.

    Attributes
    ----------
    doc_id:
        Stable identifier. Joins chaff output back to the source corpus, so it must
        survive a round trip: prefer an explicit ``id``/``url`` field from the source
        record, falling back to ``<relative-path>#<line-number>``.
    text:
        Raw document text, unmodified. Artifact detection in phase 4 reads the
        original surface form (casing, unicode punctuation, whitespace), so nothing
        upstream of the metrics is allowed to normalise this.
    source:
        Where it came from, for provenance in reports.
    meta:
        Passthrough of any remaining fields on the source record.
    label:
        Optional ground truth (``"human"`` / ``"synthetic"``), present only in
        labelled evaluation corpora. Never read by scoring — only by calibration
        in phase 6. Keeping it off the scoring path prevents accidental leakage.
    """

    doc_id: str
    text: str
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None

    def __len__(self) -> int:
        return len(self.text)

    @property
    def preview(self) -> str:
        flat = " ".join(self.text.split())
        return flat[:117] + "..." if len(flat) > 120 else flat
