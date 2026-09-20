"""Tokenization behaviour that the metric families depend on."""

from __future__ import annotations

from chaff.tokenization import (
    MIN_ANALYSABLE_WORDS,
    build_view,
    ngrams,
    split_lines,
    split_paragraphs,
    split_sentences,
    tokenize,
    tokenize_words,
)


def test_words_exclude_digits_and_punctuation():
    words = tokenize_words("Version 2.4 shipped, finally!")
    assert words == ["version", "shipped", "finally"]


def test_internal_apostrophes_survive_both_forms():
    assert tokenize_words("don't") == ["don't"]
    assert tokenize_words("don’t") == ["don’t"]


def test_full_tokenize_keeps_punctuation_as_tokens():
    # Formatting-artifact detection in phase 4 reads punctuation rhythm, so
    # punctuation must survive tokenization as its own token.
    assert tokenize("hi, there!") == ["hi", ",", "there", "!"]


def test_numbers_hold_together():
    assert tokenize("released 2.4.0 on 2026-03-11") == [
        "released", "2.4.0", "on", "2026", "-", "03", "-", "11",
    ]


def test_sentence_split_respects_abbreviations():
    sentences = split_sentences("Dr. Smith left. He returned at 3.5 hours.")
    assert sentences == ["Dr. Smith left.", "He returned at 3.5 hours."]


def test_sentence_split_respects_initials():
    assert split_sentences("J. R. R. Tolkien wrote it. Then he stopped.") == [
        "J. R. R. Tolkien wrote it.",
        "Then he stopped.",
    ]


def test_sentence_split_on_empty_text():
    assert split_sentences("   ") == []


def test_lines_drop_blanks_but_keep_indentation():
    # Markdown indentation is a signal, so leading whitespace must be preserved.
    assert split_lines("a\n\n  b  \n") == ["a", "  b"]


def test_paragraphs_split_on_blank_lines():
    assert split_paragraphs("one\nstill one\n\ntwo") == ["one\nstill one", "two"]


def test_ngrams_yield_overlapping_tuples():
    assert list(ngrams(["a", "b", "c"], 2)) == [("a", "b"), ("b", "c")]


def test_ngrams_empty_when_sequence_too_short():
    assert list(ngrams(["a"], 3)) == []


def test_view_caches_consistent_counts():
    view = build_view("The cat sat. The cat sat again.")
    assert view.n_words == len(view.words) == 7
    assert view.n_types == 4  # the, cat, sat, again
    assert len(view.sentences) == 2


def test_short_documents_are_not_analysable():
    assert not build_view("too short").is_analysable
    assert build_view(" ".join(["word"] * MIN_ANALYSABLE_WORDS)).is_analysable
