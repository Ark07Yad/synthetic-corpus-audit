"""Corpus reader behaviour, including the malformed-input paths that a crawl
dump will absolutely exercise."""

from __future__ import annotations

import gzip
import json
import os

import pytest

from chaff.corpus_io import CorpusError, iter_documents, write_jsonl


def _write(tmp_path, name, records):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


def test_reads_jsonl_and_autodetects_text_field(tmp_path):
    path = _write(tmp_path, "c.jsonl", [
        {"id": "a", "text": "first"},
        {"id": "b", "content": "second"},
        {"id": "c", "raw_content": "third"},
    ])
    assert [d.text for d in iter_documents(path)] == ["first", "second", "third"]


def test_explicit_text_field_skips_records_without_it(tmp_path):
    path = _write(tmp_path, "c.jsonl", [{"body": "kept"}, {"text": "dropped"}])
    docs = list(iter_documents(path, text_field="body"))
    assert [d.text for d in docs] == ["kept"]


def test_doc_id_falls_back_to_source_and_line(tmp_path):
    path = _write(tmp_path, "c.jsonl", [{"text": "no id here"}])
    assert list(iter_documents(path))[0].doc_id == "c.jsonl#1"


def test_id_precedence_prefers_explicit_id_over_url(tmp_path):
    path = _write(tmp_path, "c.jsonl", [{"id": "real", "url": "http://x", "text": "t"}])
    assert list(iter_documents(path))[0].doc_id == "real"


def test_label_is_captured_for_evaluation_corpora(tmp_path):
    path = _write(tmp_path, "c.jsonl", [{"text": "t", "label": "synthetic"}])
    assert list(iter_documents(path))[0].label == "synthetic"


def test_metadata_passthrough_excludes_the_text_field(tmp_path):
    path = _write(tmp_path, "c.jsonl", [{"text": "t", "genre": "forum"}])
    meta = list(iter_documents(path))[0].meta
    assert meta["genre"] == "forum"
    assert "text" not in meta


def test_malformed_lines_are_skipped_not_fatal(tmp_path):
    path = os.path.join(str(tmp_path), "c.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"text": "good"}\n')
        fh.write("{ not json at all\n")
        fh.write("\n")
        fh.write('["a list, not an object"]\n')
        fh.write('{"text": "also good"}\n')
    assert [d.text for d in iter_documents(path)] == ["good", "also good"]


def test_gzip_is_transparent(tmp_path):
    path = os.path.join(str(tmp_path), "c.jsonl.gz")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": "compressed"}) + "\n")
    assert [d.text for d in iter_documents(path)] == ["compressed"]


def test_limit_counts_documents_after_filtering(tmp_path):
    path = _write(tmp_path, "c.jsonl", [
        {"text": "x"}, {"text": "long enough"}, {"text": "also long enough"},
    ])
    docs = list(iter_documents(path, min_chars=5, limit=1))
    assert [d.text for d in docs] == ["long enough"]


def test_directory_walk_is_sorted_for_reproducibility(tmp_path):
    root = str(tmp_path)
    for name in ("b.txt", "a.txt", "c.txt"):
        with open(os.path.join(root, name), "w", encoding="utf-8") as fh:
            fh.write("content of " + name)
    assert [d.doc_id for d in iter_documents(root)] == ["a.txt", "b.txt", "c.txt"]


def test_hidden_files_are_ignored(tmp_path):
    root = str(tmp_path)
    for name in ("visible.txt", ".hidden.txt"):
        with open(os.path.join(root, name), "w", encoding="utf-8") as fh:
            fh.write("text")
    assert [d.doc_id for d in iter_documents(root)] == ["visible.txt"]


def test_missing_path_raises_corpus_error():
    with pytest.raises(CorpusError):
        list(iter_documents("/definitely/not/a/real/path.jsonl"))


def test_write_jsonl_creates_parents_and_roundtrips(tmp_path):
    path = os.path.join(str(tmp_path), "nested", "out.jsonl")
    count = write_jsonl(path, [{"doc_id": "a", "n": 1}, {"doc_id": "b", "n": 2}])
    assert count == 2
    with open(path, encoding="utf-8") as fh:
        assert [json.loads(line)["doc_id"] for line in fh] == ["a", "b"]
