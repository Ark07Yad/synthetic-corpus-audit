"""`chaff dedup`: near-duplicate clusters by one-permutation MinHash and LSH banding."""

from __future__ import annotations

import json
import os
import random

import pytest

from chaff.cli import main
from chaff.dedup import (
    BANDS,
    BINS,
    DEFAULT_THRESHOLD,
    MIN_SHINGLES,
    ROWS,
    SHINGLE,
    find_near_duplicates,
    signature,
    similarity,
)
from chaff.tokenization import ngrams

VOCAB = ["w" + chr(97 + i % 26) + chr(97 + (i // 26) % 26) for i in range(2000)]


def _doc(rng, n=400):
    return [rng.choice(VOCAB) for _ in range(n)]


def _edit(rng, words, frac):
    out = list(words)
    for i in rng.sample(range(len(out)), int(frac * len(out))):
        out[i] = rng.choice(VOCAB)
    return out


def _jaccard(a, b):
    sa, sb = set(ngrams(a, SHINGLE)), set(ngrams(b, SHINGLE))
    return len(sa & sb) / len(sa | sb)


def test_banding_examines_nearly_every_pair_at_the_default_threshold():
    """The first version used 8 bands of 8 rows: a pair at exactly the threshold became a
    candidate only 77% of the time, so a quarter of true near-copies were never compared."""
    assert BANDS * ROWS == BINS
    assert 1 - (1 - DEFAULT_THRESHOLD ** ROWS) ** BANDS > 0.99


def test_the_estimate_tracks_exact_jaccard():
    rng = random.Random(0)
    errors = []
    for frac in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0):
        for _ in range(12):
            a = _doc(rng)
            b = _edit(rng, a, frac) if frac < 1 else _doc(rng)
            errors.append(abs(similarity(signature(a), signature(b)) - _jaccard(a, b)))
    assert sum(errors) / len(errors) < 0.05
    assert max(errors) < 0.2


def test_identical_documents_have_similarity_one():
    words = _doc(random.Random(1))
    assert similarity(signature(words), signature(list(words))) == 1.0


def test_too_short_documents_get_no_signature():
    words = _doc(random.Random(2), n=MIN_SHINGLES + SHINGLE - 2)    # MIN_SHINGLES - 1 shingles
    assert signature(words) is None
    assert signature(_doc(random.Random(2), n=MIN_SHINGLES + SHINGLE - 1)) is not None


def test_empty_bins_are_never_agreement():
    """A short document fills only some of the 64 bins. Two of them must not look similar
    because they share empty bins — so even a document against itself scores the share of
    bins it filled, not 1."""
    short = _doc(random.Random(3), n=20)                 # 16 shingles, so many bins empty
    sig = signature(short)
    filled = sum(1 for v in sig if v != (1 << 64) - 1)
    assert filled < BINS
    assert similarity(sig, sig) == filled / BINS
    other = signature(_doc(random.Random(4), n=20))
    assert similarity(sig, other) == 0.0



def test_shared_empty_bins_never_make_candidates():
    """Short documents leave whole bands empty; bucketing on those would pair every short
    document with every other one. Those keys are skipped, so nothing is even compared."""
    docs = [("s{0}".format(i), _doc(random.Random(100 + i), n=20), None) for i in range(30)]
    result = find_near_duplicates(docs)
    assert result.pairs_checked == 0 and not result.clusters

def _docs(seed=0, families=20, copies=4, frac=0.01, unique=200):
    rng = random.Random(seed)
    docs, truth = [], {}
    for f in range(families):
        base = _doc(rng)
        for k in range(copies):
            doc_id = "fam{0:02d}-{1}".format(f, k)
            docs.append((doc_id, _edit(rng, base, frac) if k else base, "synthetic"))
            truth[doc_id] = f
    for u in range(unique):
        docs.append(("uniq{0:03d}".format(u), _doc(rng), "human"))
    return docs, truth


def test_near_copies_cluster_and_unrelated_documents_do_not():
    docs, truth = _docs()
    result = find_near_duplicates(docs)
    clustered = {result.doc_ids[i] for c in result.clusters for i in c}
    assert clustered == set(truth)                               # every copy, no unique doc
    assert len(result.clusters) == 20
    for members in result.clusters:
        assert len({truth[result.doc_ids[i]] for i in members}) == 1   # every cluster pure


def test_a_flood_costs_linear_verification():
    """Thousands of copies of one template land in one bucket. Verifying every pair there
    would be quadratic; anchor-based verification is about one check per document."""
    rng = random.Random(5)
    base = _doc(rng)
    docs = [("t{0}".format(i), _edit(rng, base, 0.005), None) for i in range(400)]
    result = find_near_duplicates(docs)
    assert len(result.clusters) == 1 and len(result.clusters[0]) == 400
    assert result.pairs_checked < 2 * 400          # all pairs would be 79,800 per band


def test_rows_report_clusters_largest_first():
    rng = random.Random(6)
    big, small = _doc(rng), _doc(rng)
    docs = ([("small{0}".format(i), list(small), "x") for i in range(2)]
            + [("big{0}".format(i), list(big), "y") for i in range(5)]
            + [("alone", _doc(rng), None), ("tiny", ["too", "short"], None)])
    result = find_near_duplicates(docs)
    rows = {r["doc_id"]: r for r in result.rows()}
    assert result.skipped_short == 1 and "tiny" not in rows
    assert rows["big0"]["cluster"] == 0 and rows["big0"]["cluster_size"] == 5
    assert rows["small0"]["cluster"] == 1 and rows["small0"]["cluster_size"] == 2
    assert rows["alone"]["cluster"] is None and rows["alone"]["cluster_size"] == 1
    assert rows["big3"]["max_similarity"] == 1.0 and rows["big3"]["label"] == "y"
    assert "label" not in rows["alone"]
    assert result.documents_in_clusters == 7



def test_members_that_miss_the_anchor_are_still_compared_with_each_other(monkeypatch):
    """Verification is anchored on each bucket's first member. Two near-copies whose every
    shared bucket opens with an unrelated document must still be compared with each other."""
    import chaff.dedup as dedup
    copy = list(range(1000, 1000 + BINS))
    sigs = {"b1": list(copy), "b2": list(copy)}
    for band in range(BANDS - 3, BANDS):            # the copies differ on three bands
        for k in range(band * ROWS, (band + 1) * ROWS):
            sigs["b2"][k] = 9000 + k
    for band in range(BANDS - 3):                   # a decoy opens every band they share
        decoy = list(range(5000 + 100 * band, 5000 + 100 * band + BINS))
        decoy[band * ROWS:(band + 1) * ROWS] = copy[band * ROWS:(band + 1) * ROWS]
        sigs["decoy{0}".format(band)] = decoy
    monkeypatch.setattr(dedup, "signature", lambda words: dedup.array("Q", sigs[words[0]]))
    order = ["decoy{0}".format(b) for b in range(BANDS - 3)] + ["b1", "b2"]
    result = find_near_duplicates([(k, [k], None) for k in order])
    assert [sorted(result.doc_ids[i] for i in c) for c in result.clusters] == [["b1", "b2"]]

@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_threshold_must_be_a_similarity(threshold):
    with pytest.raises(ValueError):
        find_near_duplicates([], threshold=threshold)


def _write(tmp_path, docs):
    path = os.path.join(str(tmp_path), "c.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for doc_id, words, label in docs:
            rec = {"id": doc_id, "text": " ".join(words)}
            if label:
                rec["label"] = label
            fh.write(json.dumps(rec) + "\n")
    return path


def test_cli_dedup_writes_one_row_per_document(tmp_path, capsys):
    docs, truth = _docs(families=3, unique=10)
    corpus = _write(tmp_path, docs)
    out = os.path.join(str(tmp_path), "clusters.jsonl")
    assert main(["dedup", corpus, "--out", out]) == 0
    printed = capsys.readouterr().out
    assert "3 clusters holding 12 documents" in printed and "9 would be removed" in printed
    rows = [json.loads(line) for line in open(out)]
    assert len(rows) == len(docs)
    assert sum(1 for r in rows if r["cluster"] is not None) == len(truth)


def test_cli_dedup_rejects_a_bad_threshold(tmp_path, capsys):
    corpus = _write(tmp_path, _docs(families=1, unique=1)[0])
    assert main(["dedup", corpus, "--threshold", "2"]) == 2
    assert "--threshold" in capsys.readouterr().err
