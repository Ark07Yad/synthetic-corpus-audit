"""Streaming corpus readers and writers.

chaff is pointed at pre-training dumps, which are routinely larger than memory.
Every reader here is a generator and nothing accumulates the corpus in a list.
Phase 3 needs a second pass over the corpus to build its language model; that is
implemented as a genuine second read of the source rather than by caching documents.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .document import Document

#: Field names checked, in order, when the record's text field is not given
#: explicitly. Covers Common Crawl / C4 / RedPajama / The Pile conventions.
TEXT_FIELD_CANDIDATES: Sequence[str] = (
    "text",
    "content",
    "raw_content",
    "body",
    "document",
    "page_content",
)

#: Field names checked, in order, for a stable document id.
ID_FIELD_CANDIDATES: Sequence[str] = ("doc_id", "id", "_id", "url", "uri", "path")

#: Field names that may carry ground-truth labels in evaluation corpora.
LABEL_FIELD_CANDIDATES: Sequence[str] = ("label", "origin", "source_type", "class")

JSONL_SUFFIXES = (".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz", ".json.gz")
TEXT_SUFFIXES = (".txt", ".md", ".rst", ".text")


class CorpusError(RuntimeError):
    """Raised when a corpus cannot be read or contains no usable documents."""


def _open_text(path: str) -> io.TextIOBase:
    """Open a file as UTF-8 text, transparently decompressing ``.gz``.

    ``errors="replace"`` is intentional: crawl dumps contain genuinely broken
    encodings, and aborting an eight-hour profiling run over one bad byte is worse
    than analysing that document with a few replacement characters in it.
    """
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _first_present(record: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _record_to_document(
    record: Dict[str, Any],
    *,
    source: str,
    line_no: int,
    text_field: Optional[str],
) -> Optional[Document]:
    if text_field is not None:
        text = record.get(text_field)
        if not isinstance(text, str):
            return None
        used_field = text_field
    else:
        text = _first_present(record, TEXT_FIELD_CANDIDATES)
        if text is None:
            return None
        used_field = next(
            k for k in TEXT_FIELD_CANDIDATES
            if isinstance(record.get(k), str) and record[k].strip()
        )

    doc_id = _first_present(record, ID_FIELD_CANDIDATES) or "{0}#{1}".format(source, line_no)
    label = _first_present(record, LABEL_FIELD_CANDIDATES)
    meta = {k: v for k, v in record.items() if k != used_field}

    return Document(doc_id=doc_id, text=text, source=source, meta=meta, label=label)


def iter_jsonl(path: str, text_field: Optional[str] = None) -> Iterator[Document]:
    """Yield documents from a (optionally gzipped) JSON Lines file."""
    source = os.path.basename(path)
    with _open_text(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # One malformed line should not kill the pass; it is counted by
                # the caller through the skipped-document tally.
                continue
            if not isinstance(record, dict):
                continue
            doc = _record_to_document(
                record, source=source, line_no=line_no, text_field=text_field
            )
            if doc is not None:
                yield doc


def iter_textfile(path: str, root: Optional[str] = None) -> Iterator[Document]:
    """Yield a single document for a plain-text file."""
    rel = os.path.relpath(path, root) if root else os.path.basename(path)
    with _open_text(path) as handle:
        text = handle.read()
    if text.strip():
        yield Document(doc_id=rel, text=text, source=rel)


def iter_directory(root: str, text_field: Optional[str] = None) -> Iterator[Document]:
    """Walk a directory, reading every recognised file. Order is sorted for
    reproducibility (see OBJECTIVE §2 G5) — ``os.walk`` order is filesystem
    dependent and would make scores non-deterministic across machines."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            full = os.path.join(dirpath, name)
            if name.endswith(JSONL_SUFFIXES):
                for doc in iter_jsonl(full, text_field=text_field):
                    yield doc
            elif name.endswith(TEXT_SUFFIXES):
                for doc in iter_textfile(full, root=root):
                    yield doc


def iter_stdin(text_field: Optional[str] = None) -> Iterator[Document]:
    """Read JSONL from stdin, falling back to treating input as one plain document."""
    data = sys.stdin.read()
    stripped = data.lstrip()
    if stripped.startswith("{"):
        for line_no, line in enumerate(data.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                doc = _record_to_document(
                    record, source="<stdin>", line_no=line_no, text_field=text_field
                )
                if doc is not None:
                    yield doc
    elif data.strip():
        yield Document(doc_id="<stdin>", text=data, source="<stdin>")


def iter_documents(
    path: str,
    *,
    text_field: Optional[str] = None,
    limit: Optional[int] = None,
    min_chars: int = 0,
) -> Iterator[Document]:
    """Yield documents from a file, directory, or ``-`` for stdin.

    Parameters
    ----------
    limit:
        Stop after this many documents have been *yielded* (post-filtering), which
        makes ``--limit`` a predictable sampling knob rather than a read budget.
    min_chars:
        Drop documents shorter than this before they reach the metrics.
    """
    if path == "-":
        stream: Iterable[Document] = iter_stdin(text_field=text_field)
    elif os.path.isdir(path):
        stream = iter_directory(path, text_field=text_field)
    elif os.path.isfile(path):
        if path.endswith(JSONL_SUFFIXES):
            stream = iter_jsonl(path, text_field=text_field)
        else:
            stream = iter_textfile(path)
    else:
        raise CorpusError("no such corpus path: {0}".format(path))

    yielded = 0
    for doc in stream:
        if min_chars and len(doc.text) < min_chars:
            continue
        yield doc
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    """Write rows as JSON Lines, creating parent directories. Returns the count."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
