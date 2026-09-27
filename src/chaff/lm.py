"""Corpus-internal n-gram language model with exact leave-one-document-out scoring.

This is the "perplexity" in "perplexity variation" (CORE_PROBLEM §1, decision D2). The
usual approach runs a pretrained transformer over every document. chaff builds a small
n-gram model *from the corpus being audited* instead, which has three consequences:

1. **No download, no GPU, no dependency.** Counting n-grams is one streaming pass.
2. **The null hypothesis is the corpus itself.** A document scores as predictable when
   it is predictable *given everything else in this corpus* — which is what matters
   when the question is whether it will distort a model trained on this corpus.
3. **A document must never predict itself.** Scored against counts that include its
   own n-grams, every document is partly memorised by the model and long documents
   look artificially fluent. Every document is therefore scored against the corpus
   **minus itself**, by subtracting its own counts at query time. That is exact, not
   approximate: ``test_lm.py`` asserts the scores equal those from a model trained on
   the corpus with the document physically removed.

Smoothing
---------
Interpolated absolute discounting (Ney, Essen & Kneser 1994), order 3, D = 0.75::

    P_k(w | h) = max(c(h,w) - D, 0) / c(h)  +  D * T(h) / c(h) * P_{k-1}(w | h')
    P_1(w)     = (c(w) + 1) / (N + V + 1)          (+1 reserves an OOV bucket)

``T(h)`` is the number of distinct words seen after ``h``. Every level normalises
exactly, so ``-log2 P`` is a true surprisal in bits rather than a score. Stupid
backoff — named in the phase 1 plan — was rejected for that reason: its scores do not
sum to one, and "surprisal variance" computed from unnormalised scores measures the
backoff schedule as much as the text.

Kneser-Ney continuation counts would smooth slightly better, but their leave-one-out
correction requires tracking which *other* contexts each word appears in. Absolute
discounting keeps the correction local to the document being scored.

Lossless pruning
----------------
Most distinct n-grams in any corpus occur exactly once (Zipf again). Under
leave-one-document-out scoring those entries are provably dead weight:

* A corpus-singleton n-gram belongs to exactly one document.
* Scoring that document subtracts it, so its effective count is 0 either way.
* Scoring any *other* document never looks it up, because a document only queries
  n-grams that it contains — and if another document contained it, its corpus count
  would be at least 2.

So after pass 1 every table entry equal to 1 is dropped, and a missing key queried by a
document is inferred to have exactly the document's own count. The context aggregates
``c(h)``, ``T(h)``, ``N`` and ``V`` are computed *before* pruning, so they still carry
the pruned entries' mass. The result is bit-identical to the unpruned model under
leave-one-out (asserted in the tests).

The guarantee is precisely scoped, and both limits are enforced or tested:

* It covers **the n-grams of the document being scored** — which is every query
  :meth:`NgramLM.surprisals` makes. It does *not* cover arbitrary queries: a singleton
  belonging to some other document is gone, so the full conditional distribution over
  all continuations no longer sums to one. The public API never asks that question.
* It does **not** hold for documents outside the training data, so a pruned model
  refuses to score in that mode.

The saving is on the *resident* model for pass 2. Peak memory during pass 1 is still
``O(distinct n-grams)``, because singletons cannot be identified until counting ends.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

BOS = "<s>"
DEFAULT_ORDER = 3
DEFAULT_DISCOUNT = 0.75

Gram = Tuple[str, ...]


class StreamMismatchError(RuntimeError):
    """Leave-one-out subtraction produced an impossible count.

    Means the document being scored was not part of the stream the model was trained
    on — pass 1 and pass 2 saw different documents (a different ``--limit``, a corpus
    file that changed between passes, a filter applied to only one pass). Raised
    rather than clamped, because clamping would silently produce plausible nonsense.
    """


class _LocalCounts:
    """One document's own n-gram counts, for leave-one-out subtraction."""

    __slots__ = ("counts", "ctx", "types", "types_equal", "n_tokens", "v_drop")

    def __init__(self, order: int) -> None:
        self.counts: List[Dict[Gram, int]] = [dict() for _ in range(order + 1)]
        self.ctx: List[Dict[Gram, int]] = [dict() for _ in range(order + 1)]
        # Distinct continuations of each context within this document. Needed to
        # reconstruct a pruned corpus T(h), which can only have been 1.
        self.types: List[Dict[Gram, int]] = [dict() for _ in range(order + 1)]
        # For each context: how many of its continuations exist in the corpus *only*
        # because of this document. Removing the document removes them from T(h).
        self.types_equal: List[Dict[Gram, int]] = [dict() for _ in range(order + 1)]
        self.n_tokens = 0
        # Vocabulary types that exist in the corpus only because of this document.
        self.v_drop = 0


class NgramLM:
    """Interpolated absolute-discounting n-gram LM over word sequences.

    Usage::

        lm = NgramLM()
        for words in corpus: lm.observe(words)     # pass 1
        lm.finalize()
        bits = lm.surprisals(words, loo=True)      # pass 2, per document
    """

    def __init__(self, order: int = DEFAULT_ORDER, discount: float = DEFAULT_DISCOUNT) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        if not 0.0 < discount < 1.0:
            # D < 1 is what guarantees every observed count survives discounting,
            # which is what makes each interpolation level normalise exactly.
            raise ValueError("discount must lie strictly between 0 and 1")
        self.order = order
        self.discount = discount
        # Index k holds k-grams. Index 0 is unused so that k reads naturally.
        self.counts: List[Dict[Gram, int]] = [dict() for _ in range(order + 1)]
        self.ctx: List[Dict[Gram, int]] = [dict() for _ in range(order + 1)]
        self.types: List[Dict[Gram, int]] = [dict() for _ in range(order + 1)]
        self.n_tokens = 0
        self.vocab_size = 0
        self.n_documents = 0
        self.finalized = False
        self.pruned = False
        self.entries_before_prune = 0

    # ------------------------------------------------------------------ pass 1

    def _padded(self, words: Sequence[str]) -> List[str]:
        return [BOS] * (self.order - 1) + list(words)

    def _grams(self, words: Sequence[str]):
        """Yield ``(k, gram)`` for every k-gram ending at every real word."""
        seq = self._padded(words)
        start = self.order - 1
        for i in range(start, len(seq)):
            for k in range(1, self.order + 1):
                yield k, tuple(seq[i - k + 1 : i + 1])

    def observe(self, words: Sequence[str]) -> None:
        """Add one document's words to the model."""
        if self.finalized:
            raise RuntimeError("cannot observe after finalize()")
        self.n_documents += 1
        self.n_tokens += len(words)
        for k, gram in self._grams(words):
            table = self.counts[k]
            previous = table.get(gram, 0)
            table[gram] = previous + 1
            if k == 1:
                if previous == 0:
                    self.vocab_size += 1
                continue
            history = gram[:-1]
            self.ctx[k][history] = self.ctx[k].get(history, 0) + 1
            if previous == 0:
                self.types[k][history] = self.types[k].get(history, 0) + 1

    def finalize(self, prune: bool = True) -> None:
        """Freeze the model. With ``prune``, drop every table entry equal to 1.

        Pruning is lossless for leave-one-out scoring and forbidden otherwise — see
        the module docstring for the argument, and ``test_lm.py`` for the proof.
        """
        if self.finalized:
            return
        self.entries_before_prune = self.n_entries
        if prune:
            for k in range(1, self.order + 1):
                for table in (self.counts[k], self.ctx[k], self.types[k]):
                    dead = [key for key, value in table.items() if value == 1]
                    for key in dead:
                        del table[key]
            self.pruned = True
        self.finalized = True

    @property
    def bigram_coverage(self) -> float:
        """Share of word positions whose bigram occurs at least twice in the corpus.

        The model's *adequacy*: how often, when scoring a word, the model has seen
        that exact word pair somewhere other than (probably) the current document.
        When coverage is low the model is mostly guessing from unigram frequencies,
        and the per-word surprisal sequence is dominated by backoff noise rather than
        by the text — which is exactly the regime where surprisal *spread* measures
        stop meaning anything and, measured on controlled corpora, invert.

        On a pruned model this is free: the surviving bigram entries are precisely
        those seen at least twice. It is an estimate — a pair seen twice inside one
        document has leave-one-out count zero but still counts here — and it is
        measured over the training corpus, so in reference mode it describes the
        reference model rather than how well that model covers the audited text.
        """
        if self.order < 2 or self.n_tokens == 0:
            return 0.0
        repeated = sum(c for c in self.counts[2].values() if c >= 2)
        return repeated / float(self.n_tokens)

    @property
    def n_entries(self) -> int:
        return sum(len(t) for k in range(1, self.order + 1)
                   for t in (self.counts[k], self.ctx[k], self.types[k]))

    # ------------------------------------------------------------------ pass 2

    def _corpus_value(self, table: Dict[Gram, int], key: Gram, local_value: int) -> int:
        """Corpus value of ``key``, reconstructing it when pruning removed it.

        A missing key on a pruned model had corpus value 1 at most. The document
        querying it contains it (value >= 1), so the corpus value was exactly 1 and
        the document supplied all of it. Anything else is a stream mismatch.
        """
        stored = table.get(key)
        if stored is not None:
            return stored
        if local_value == 0:
            return 0
        if self.pruned and local_value == 1:
            return 1
        raise StreamMismatchError(
            "document n-gram {0!r} is absent from the trained model".format(key)
        )

    def _local_counts(self, words: Sequence[str]) -> _LocalCounts:
        local = _LocalCounts(self.order)
        local.n_tokens = len(words)
        for k, gram in self._grams(words):
            table = local.counts[k]
            previous = table.get(gram, 0)
            table[gram] = previous + 1
            if k > 1:
                history = gram[:-1]
                local.ctx[k][history] = local.ctx[k].get(history, 0) + 1
                if previous == 0:
                    local.types[k][history] = local.types[k].get(history, 0) + 1

        # A continuation or vocabulary type vanishes from the leave-one-out model
        # exactly when this document contributed all of its corpus occurrences.
        for k in range(1, self.order + 1):
            for gram, own in local.counts[k].items():
                if self._corpus_value(self.counts[k], gram, own) != own:
                    continue
                if k == 1:
                    local.v_drop += 1
                else:
                    history = gram[:-1]
                    local.types_equal[k][history] = local.types_equal[k].get(history, 0) + 1
        return local

    def _loo(self, corpus_value: int, own: int, what: str, key: Gram) -> int:
        value = corpus_value - own
        if value < 0:
            raise StreamMismatchError(
                "negative {0} for {1!r} after leave-one-out".format(what, key)
            )
        return value

    def _get(self, table: Dict[Gram, int], local_table: Optional[Dict[Gram, int]],
             key: Gram, what: str) -> int:
        """A table value, leave-one-out adjusted when ``local_table`` is given."""
        if local_table is None:
            return table.get(key, 0)
        own = local_table.get(key, 0)
        return self._loo(self._corpus_value(table, key, own), own, what, key)

    def _prob(self, gram: Gram, local: Optional[_LocalCounts]) -> float:
        k = len(gram)

        if k == 1:
            count = self._get(self.counts[1], local.counts[1] if local else None, gram, "count")
            n = self.n_tokens - (local.n_tokens if local else 0)
            v = self.vocab_size - (local.v_drop if local else 0)
            return (count + 1.0) / (n + v + 1.0)

        history = gram[:-1]
        lower = self._prob(gram[1:], local)
        ctx = self._get(self.ctx[k], local.ctx[k] if local else None, history, "context")
        if ctx <= 0:
            return lower

        count = self._get(self.counts[k], local.counts[k] if local else None, gram, "count")
        if local is None:
            types = self.types[k].get(history, 0)
        else:
            # T(h) loses one for every continuation this document alone supplied.
            corpus_types = self._corpus_value(self.types[k], history, local.types[k].get(history, 0))
            types = self._loo(corpus_types, local.types_equal[k].get(history, 0), "type count", history)

        d = self.discount
        return max(count - d, 0.0) / ctx + (d * types / ctx) * lower

    def surprisals(self, words: Sequence[str], loo: bool = True) -> List[float]:
        """Per-word surprisal in bits, ``-log2 P(w_i | w_{i-2}, w_{i-1})``.

        With ``loo=True`` the document is scored against the corpus minus itself,
        which requires that it was observed during pass 1. With ``loo=False`` it is
        scored against the model as-is — the reference-corpus mode, where the
        document was never part of training.
        """
        if not self.finalized:
            raise RuntimeError("call finalize() before scoring")
        if not loo and self.pruned:
            raise ValueError(
                "a pruned model is only exact under leave-one-out; build the "
                "reference model with finalize(prune=False)"
            )
        if not words:
            return []
        local = self._local_counts(words) if loo else None
        seq = self._padded(words)
        start = self.order - 1
        out: List[float] = []
        for i in range(start, len(seq)):
            p = self._prob(tuple(seq[i - self.order + 1 : i + 1]), local)
            out.append(-math.log(p, 2))
        return out

    def summary(self) -> Dict[str, float]:
        """Model statistics for reports."""
        before = self.entries_before_prune or self.n_entries
        return {
            "order": self.order,
            "discount": self.discount,
            "documents": self.n_documents,
            "tokens": self.n_tokens,
            "vocabulary": self.vocab_size,
            "entries": self.n_entries,
            "entries_before_prune": before,
            "pruned_share": (1.0 - self.n_entries / before) if before else 0.0,
            "bigram_coverage": self.bigram_coverage,
        }
