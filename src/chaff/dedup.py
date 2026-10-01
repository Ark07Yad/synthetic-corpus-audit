"""Near-duplicate clustering (``chaff dedup``).

Per-document scores cannot see one kind of contamination: a flood. A pipeline that
generates thousands of articles from one template, or one prompt with light variation,
produces documents that are each individually unremarkable and collectively obvious —
they are near-copies of each other. This module finds those clusters.

It was planned in phase 2 as a separate command rather than an extractor
(ARCHITECTURE §9.1): deduplication is a corpus operation whose output is a cluster
assignment, not a per-document signal, and keeping it out of the extractor path keeps
extractors pure. It reads the corpus once and needs no language model.

Method
------
**Shingles** are word 5-grams, lowercased by the shared tokenizer. Five words is long
enough that unrelated documents rarely share many, short enough that a lightly edited
copy still shares most.

**One-permutation MinHash.** Classic MinHash hashes every shingle under k independent
functions — 64 hashes per shingle, which is the whole cost of the method in pure
Python. One-permutation hashing (Li, Owen & Zhang, 2012) hashes each shingle *once*,
uses part of the hash to choose one of k bins and keeps the minimum per bin. The
fraction of bins two documents agree on estimates their Jaccard similarity, at a k-th
of the hashing cost. Bins left empty (documents with fewer shingles than bins) never
count as agreement, which makes short documents harder to match — the conservative
direction.

**LSH banding.** The k bins are cut into b bands of r rows; documents that agree on
every row of any band become candidates, with probability ``1 - (1 - J^r)^b``. The
first version used b = 8, r = 8 and claimed near-copies were "rarely missed"; at the
default threshold J = 0.8 that probability is only 0.77, so a quarter of true pairs were
never even examined. b = 16, r = 4 puts it at 0.9998 for J = 0.8 (and 0.12 for unrelated
J = 0.3, which verification then rejects). Candidates are verified on the full
signature, and verified pairs are joined into clusters with union-find.

**Floods stay linear.** The case this command exists for — thousands of copies of one
template — puts thousands of documents in the same bucket, and verifying every pair in
it would be quadratic. Instead each bucket is verified against an anchor: members that
match it join its cluster, the rest are re-bucketed around a new anchor, and pairs
already in the same cluster are never re-verified in later bands. A flood of m copies
costs about m verifications, not m^2 / 2.

What remains is estimator noise at the threshold itself: a pair whose true Jaccard is
0.82 is estimated anywhere around 0.80 +/- 0.05, so pairs sitting right at the cutoff
are matched only some of the time. That is inherent to any similarity threshold.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .corpus_io import iter_documents
from .tokenization import ngrams, tokenize_words

SHINGLE = 5
BINS = 64
BANDS = 16
ROWS = BINS // BANDS
DEFAULT_THRESHOLD = 0.8
#: Documents with fewer shingles than this are too short to compare meaningfully.
MIN_SHINGLES = 10

_EMPTY = (1 << 64) - 1   # an unfilled bin; never treated as agreement


def _hash64(shingle: Tuple[str, ...]) -> int:
    digest = hashlib.blake2b(" ".join(shingle).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def signature(words: Sequence[str]) -> Optional["array"]:
    """One-permutation MinHash signature of a word sequence, or ``None`` when the
    document has fewer than :data:`MIN_SHINGLES` distinct shingles."""
    shingles = set(ngrams(words, SHINGLE))
    if len(shingles) < MIN_SHINGLES:
        return None
    bins = [_EMPTY] * BINS
    for s in shingles:
        h = _hash64(s)
        b = h % BINS             # low bits pick the bin...
        v = h // BINS            # ...the rest is the value compared within it
        if v < bins[b]:
            bins[b] = v
    return array("Q", bins)


def similarity(a: Sequence[int], b: Sequence[int]) -> float:
    """Estimated Jaccard similarity: the share of bins both documents filled and agree on."""
    return sum(1 for x, y in zip(a, b) if x == y and x != _EMPTY) / float(len(a))


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class DedupResult:
    doc_ids: List[str]
    labels: List[Optional[str]]
    #: cluster index per compared document (-1: not in any cluster of two or more)
    cluster_of: List[int]
    #: highest similarity verified to another document, per document. Verification is
    #: anchor-based, so this is a lower bound on the true maximum (exact for pairs).
    best: List[float]
    clusters: List[List[int]]
    skipped_short: int
    pairs_checked: int
    threshold: float

    @property
    def documents_in_clusters(self) -> int:
        return sum(len(c) for c in self.clusters)

    def rows(self) -> Iterator[Dict]:
        for i, doc_id in enumerate(self.doc_ids):
            c = self.cluster_of[i]
            row = {"doc_id": doc_id, "cluster": c if c >= 0 else None,
                   "cluster_size": len(self.clusters[c]) if c >= 0 else 1,
                   "max_similarity": round(self.best[i], 4)}
            if self.labels[i] is not None:
                row["label"] = self.labels[i]
            yield row


def find_near_duplicates(
    docs: Iterable[Tuple[str, Sequence[str], Optional[str]]],
    threshold: float = DEFAULT_THRESHOLD,
) -> DedupResult:
    """Cluster ``(doc_id, words, label)`` triples whose estimated Jaccard similarity on
    word 5-gram shingles is at least ``threshold``."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    doc_ids: List[str] = []
    labels: List[Optional[str]] = []
    sigs: List["array"] = []
    skipped = 0
    for doc_id, words, label in docs:
        sig = signature(words)
        if sig is None:
            skipped += 1
            continue
        doc_ids.append(doc_id)
        labels.append(label)
        sigs.append(sig)

    n = len(sigs)
    uf = _UnionFind(n)
    best = [0.0] * n
    verified = 0
    for band in range(BANDS):
        lo, hi = band * ROWS, (band + 1) * ROWS
        buckets: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for i, sig in enumerate(sigs):
            key = tuple(sig[lo:hi])
            if _EMPTY in key:
                continue              # an empty bin is not evidence of agreement
            buckets[key].append(i)
        for members in buckets.values():
            while len(members) > 1:
                anchor, rest, unmatched = members[0], members[1:], []
                for b in rest:
                    if uf.find(b) == uf.find(anchor):
                        continue          # already joined, through this band or an earlier one
                    verified += 1
                    s = similarity(sigs[anchor], sigs[b])
                    if s >= threshold:
                        uf.union(anchor, b)
                        best[anchor] = max(best[anchor], s)
                        best[b] = max(best[b], s)
                    else:
                        unmatched.append(b)
                members = unmatched

    roots = Counter(uf.find(i) for i in range(n))
    index: Dict[int, int] = {}
    clusters: List[List[int]] = []
    cluster_of = [-1] * n
    for i in range(n):
        r = uf.find(i)
        if roots[r] < 2:
            continue
        if r not in index:
            index[r] = len(clusters)
            clusters.append([])
        cluster_of[i] = index[r]
        clusters[index[r]].append(i)
    clusters_sorted = sorted(range(len(clusters)), key=lambda c: -len(clusters[c]))
    remap = {old: new for new, old in enumerate(clusters_sorted)}
    cluster_of = [remap[c] if c >= 0 else -1 for c in cluster_of]
    clusters = [clusters[old] for old in clusters_sorted]
    return DedupResult(doc_ids=doc_ids, labels=labels, cluster_of=cluster_of, best=best,
                       clusters=clusters, skipped_short=skipped, pairs_checked=verified,
                       threshold=threshold)


def dedup_corpus(
    path: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    text_field: Optional[str] = None,
    limit: Optional[int] = None,
    min_chars: int = 0,
) -> DedupResult:
    """Read a corpus once and cluster its near-duplicates."""
    return find_near_duplicates(
        ((doc.doc_id, tokenize_words(doc.text), doc.label)
         for doc in iter_documents(path, text_field=text_field, limit=limit, min_chars=min_chars)),
        threshold=threshold,
    )
