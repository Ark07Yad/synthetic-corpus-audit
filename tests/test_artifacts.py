"""The artifact family (phase 4).

Most tests here are regressions for false positives found on the guaranteed-human
baseline (benchmarks/human_baseline.py: Python 3.9 stdlib docstrings + man pages). Each
one pins a case where a plausible detector fired on human text for the wrong reason.
"""

from __future__ import annotations

import pytest

from chaff.document import Document
from chaff.metrics.artifacts import (
    LLM_LEXICON,
    MIN_LINES_FOR_RATIO,
    artifact_profile,
    assistant_echo_count,
    bold_lead_share,
    hedging_count,
    human_noise_count,
    lexicon_count,
    transition_share,
)
from chaff.tokenization import build_view, tokenize_words


def _signals(text):
    return {s.name: s.value for s in artifact_profile(Document("d", text), build_view(text))}


# ------------------------------------------------------------ assistant echoes

@pytest.mark.parametrize("text", [
    "As an AI language model, I cannot browse the internet.",
    "As of my last knowledge update in 2023, the policy was different.",
    "I'm sorry, but I can't assist with that request.",
    "I hope this helps! Let me know if you have any other questions.",
    "Great question! The answer depends on context.",
    "Certainly! Here's a summary of the key points.",
])
def test_assistant_echoes_are_detected(text):
    assert assistant_echo_count(text) >= 1


def test_assistant_openers_only_count_at_the_start_of_a_line():
    assert assistant_echo_count("Certainly! Here's the plan.") == 1
    assert assistant_echo_count("He said certainly! here is the key.") == 0
    assert assistant_echo_count("Intro line.\nSure, here is the list.") == 1


def test_ordinary_technical_prose_has_no_assistant_echo():
    text = ("The ls utility lists directory contents. If no operands are given, "
            "the contents of the current directory are displayed.")
    assert assistant_echo_count(text) == 0


# --------------------------------------------------------------------- lexicon

@pytest.mark.parametrize("word", ["underscore", "underscores", "realm", "realms",
                                  "harness", "vibrant", "intricate"])
def test_domain_vocabulary_is_not_in_the_lexicon(word):
    """Each of these fired on human technical text at LLM-like rates: the "_"
    character, Kerberos realms, test harnesses. Removed on half A of the baseline."""
    assert word not in LLM_LEXICON


@pytest.mark.parametrize("word", ["robust", "comprehensive", "crucial", "enhanced",
                                  "notably", "landscape", "navigate", "leverage"])
def test_ordinary_technical_english_is_not_in_the_lexicon(word):
    assert word not in LLM_LEXICON


def test_a_kerberos_style_page_does_not_score_as_llm_vocabulary():
    text = ("The default realm is used when no realm is specified. Each realm has a "
            "key distribution centre, and cross-realm authentication links realms. ") * 5
    assert lexicon_count(tokenize_words(text)) == 0


def test_lexicon_words_are_counted():
    assert lexicon_count(tokenize_words("Let us delve into this rich tapestry, a testament to it.")) == 3


# ------------------------------------------------------------------- hedging

def test_hedging_phrases_are_counted():
    text = ("It's important to note that this plays a crucial role. "
            "That said, a wide range of options exist. In conclusion, it depends.")
    assert hedging_count(text) == 5


# --------------------------------------------------------------- transitions

def test_transitions_count_single_adverbs_only():
    """Multi-word connectives were dropped: "In addition" alone opened sentences in
    72 of 877 human baseline documents."""
    sentences = ["Moreover, it works.", "In addition, it scales.", "It is fast.",
                 "Furthermore, it is cheap."]
    assert transition_share(sentences) == pytest.approx(0.5)


# ------------------------------------------------------------------ bold lead

def test_bold_labels_are_detected_with_or_without_a_bullet():
    lines = ["- **Cost Savings:** less", "**1. Scalability:** more", "**Bold:**", "plain", "text"]
    assert bold_lead_share(lines) == pytest.approx(0.6)


def test_rest_emphasis_without_a_colon_is_not_a_bold_label():
    # The only human-baseline hits for a bare bold opener were reST "**DEPRECATED**".
    assert bold_lead_share(["**DEPRECATED** use importlib", "a", "b", "c", "d"]) == 0.0


def test_bold_lead_needs_enough_lines_to_be_a_ratio():
    assert "bold_lead_rate" not in _signals("- **Only:** one line")
    text = "\n".join(["- **Item {0}:** detail".format(i) for i in range(MIN_LINES_FOR_RATIO)])
    assert _signals(text)["bold_lead_rate"] == pytest.approx(1.0)


# ----------------------------------------------------------------- human noise

def test_casual_typing_registers_as_noise():
    assert human_noise_count(build_view("ok so i think its fine lol!! i'm done tbh")) == 5


@pytest.mark.parametrize("text", [
    "for i in range(3): print(i)",                 # loop variable
    "See (i) the header and (ii) the body.",       # list numbering
    "Returns a copy.  The original is kept.",      # typewriter double space
    "os.path.join() builds a path.",               # lowercase code opener
    "Use !! to repeat the last command.",          # csh history, not emphasis
    "i.e. the default is used.",                   # abbreviation
])
def test_house_style_and_code_are_not_noise(text):
    """Each of these fired the first version of the detector on formal human text.
    Doubled spaces alone fired on 94% of baseline documents."""
    assert human_noise_count(build_view(text)) == 0


def test_mixed_apostrophes_count_once():
    assert human_noise_count(build_view("don't won’t can't")) == 1


# -------------------------------------------------------------------- contract

def test_rates_are_per_thousand_words():
    text = "Let us delve into it. " + " ".join(["word"] * 995)
    assert _signals(text)["llm_lexicon_rate"] == pytest.approx(1000.0 / 1000, rel=0.01)


def test_empty_document_emits_nothing():
    assert artifact_profile(Document("d", ""), build_view("")) == []
