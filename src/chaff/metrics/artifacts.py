"""Formatting-watermark and phrasing-artifact signals (artifact family).

CORE_PROBLEM §3.2. Generation leaves residue that survives scraping: assistant framing
pasted into web pages, hedging scaffolds at predictable structural positions, a
recognisable vocabulary, markdown habits, and a typographic cleanliness that ordinary
web text does not have.

The central risk in this family is false positives, not missed detections. Every
artifact here also occurs in genuine human writing — contracts hedge, technical writers
say "note that", 19th-century novelists wrote "tapestry". So every detector is scored
by how often it fires on text that is **guaranteed** human, and nothing is kept on the
strength of how often it fires on synthetic text alone. See
``benchmarks/human_baseline.py``, which builds that reference from the Python 3.9
standard library (released June 2021, so every docstring predates ChatGPT) and the
system's man pages.

Rates are per 1,000 words so that a long document does not score higher merely by
being long. Most documents score exactly zero on most of these, which matters for
fusion: a median/MAD z-score is undefined when most of a column is zero. That is a
phase 5 problem, logged in PROJECT.md, not something to paper over here.
"""

from __future__ import annotations

import re
from typing import List, Sequence

from ..document import Document
from ..tokenization import TextView
from . import FAMILY_ARTIFACT, Signal, register

# MULTILINE so that "^" means start-of-line: an assistant opener like "Certainly!
# Here's" only counts at the start of a line, wherever that line sits in the document.
_FLAGS = re.IGNORECASE | re.MULTILINE

#: Chat-assistant framing: refusals, knowledge-cutoff disclaimers, the service-desk
#: openings and sign-offs. High precision by construction — these phrases describe the
#: writer as an AI or address a user in a chat turn, which a man page never does.
ASSISTANT_ECHO = [re.compile(p, _FLAGS) for p in (
    r"\bas an ai\b(?:\s+(?:language\s+)?(?:model|assistant))?",
    r"\bas a (?:large )?language model\b",
    r"\bas of my (?:last|latest|most recent) (?:knowledge )?(?:update|training)",
    r"\bmy (?:knowledge|training)(?:\s+data)? cut-?off\b",
    r"\bi(?:'m| am) (?:not able|unable) to (?:browse|access|provide|assist)",
    r"\bi (?:cannot|can't|can not) (?:assist|help) with (?:that|this)",
    r"\bi(?:'m| am) sorry,? but i (?:cannot|can't|am unable)",
    r"\bi hope (?:this|that) helps\b",
    r"\b(?:let me know|feel free to (?:ask|reach out)) if you (?:have|need) any (?:other|further|more|additional) (?:questions|help)",
    r"\bgreat question\b",
    r"^\s*(?:certainly|absolutely|sure)[!,]\s+here(?:'s| is)",
    r"\bi(?:'d| would) be (?:happy|glad) to help\b",
)]

#: Hedging and scaffolding phrases. Lower precision than the echoes, which is why this
#: is a separate signal: formal human prose hedges too, just less formulaically.
HEDGING = [re.compile(p, _FLAGS) for p in (
    r"\bit(?:'s| is) (?:important|worth|crucial|essential|vital) (?:to )?(?:note|noting|remember|mention|consider|understand|recogni[sz]e|highlight)",
    r"\bit(?:'s| is) worth (?:noting|mentioning|highlighting)",
    r"\bthat (?:being )?said,",
    r"\bkeep in mind that\b",
    r"\b(?:plays?|played|playing) an? (?:crucial|vital|pivotal|key|significant|essential|important) role\b",
    r"\bin today's (?:fast-paced|digital|rapidly (?:evolving|changing)|modern|ever-changing|interconnected)",
    r"\b(?:rapidly|ever-)(?:evolving|changing) (?:landscape|world|field)",
    r"\ban? (?:wide|diverse|broad|rich) (?:range|array|variety|spectrum) of\b",
    r"\b(?:in conclusion|in summary|to summari[sz]e|to sum up),",
    r"\bnavigat(?:e|ing) the (?:complexities|challenges|intricacies|landscape)",
    r"\bat the end of the day\b",
    r"\bwhen it comes to\b",
)]

#: Words whose frequency in published writing rose sharply after LLM assistants became
#: widely used, following the excess-vocabulary approach of Kobak et al. (2024),
#: "Delving into ChatGPT usage in academic writing through excess vocabulary".
#:
#: **Pruned against guaranteed-human text, on a held-out split.** A candidate list was
#: checked against half A of the human baseline (877 documents: Python 3.9 stdlib
#: docstrings and man pages); any word family with a form appearing in two or more
#: human documents was removed, and the false-positive rate was then measured only on
#: half B. What that removed is the reason the step exists — the words collide with
#: *domain vocabulary* in technical writing, which produces concentrated false positives
#: at LLM-like rates rather than occasional ones:
#:
#:   underscore (26 docs: the "_" character)   realm (20: Kerberos realms)
#:   harness (19: test harness)                vibrant (22)   intricate (2)
#:
#: Words excluded before measurement on the prior that they are ordinary technical
#: English were confirmed by half A — enhanced (29), notably (24), comprehensive (17),
#: robust (16), facilitate (11), utilize (9), seamless (5), landscape (3: page
#: orientation), navigate (2), crucial (2). "leverage" did not occur in the baseline and
#: stays excluded on genre grounds the baseline cannot test (business prose). The
#: baseline is technical text only; literary and journalistic false positives for
#: words like "testament" or "bustling" are unmeasured until phase 6.
LLM_LEXICON = frozenset("""
    delve delves delved delving
    tapestry tapestries
    multifaceted
    testament
    pivotal
    showcase showcases showcased showcasing
    meticulous meticulously
    embark embarks embarked embarking
    fostering
    holistic
    paramount
    bustling
    commendable
    unwavering
    groundbreaking
    transformative
    invaluable
    elevate elevates elevating
    resonate resonates resonating
    unleash unleashing
    synergy synergies
""".split())

#: Sentence-initial discourse adverbs that generated prose leans on to fake cohesion —
#: the brief's "synthetic transition tokens". Single words only: multi-word connectives
#: ("In addition", "In other words", "On the other hand", "As a result") were measured
#: on half A of the human baseline as ordinary technical English ("In addition" alone
#: opens sentences in 72 of 877 documents) and dropped. The adverbs that remain are also
#: used by human technical writers — "additionally" in 37 documents — so this is a
#: *rate*, and its human baseline is what phase 5 must calibrate against. Its evidential
#: weight is not yet established.
TRANSITIONS = frozenset("""
    moreover furthermore additionally consequently ultimately notably importantly
    overall interestingly essentially crucially indeed subsequently
""".split())

#: A line that opens with a bold *label* — "**Scalability:** ...", "- **Cost Savings:**"
#: — the single most recognisable assistant markdown habit. The colon is required: the
#: only human-baseline hits for a bare bold opener were reST emphasis ("**DEPRECATED**").
_BOLD_LEAD = re.compile(
    r"^\s*(?:[-*+•]\s+|\d+[.)]\s+)?\*\*[^*\n]{1,80}(?::\*\*|\*\*:)")

#: Minimum lines before a markdown-habit ratio is meaningful.
MIN_LINES_FOR_RATIO = 5

# ---------------------------------------------------------------- human noise
# Negative evidence: marks of a person typing quickly. Restricted to markers that are
# (a) about haste rather than house style and (b) survive HTML text extraction, which is
# how a crawl reaches chaff. Measured on half A of the human baseline, two plausible
# markers failed both tests and were removed:
#   * doubled spaces mid-line fired on 94% of human documents — man pages and PEP 8
#     docstrings put two spaces after a full stop, and HTML collapses whitespace anyway;
#   * lowercase sentence openings fired on 82% — code identifiers ("os.path ...").
# "Lowercase i" is restricted to pronoun use: the bare form matched loop variables
# ("for i in") and list numbering ("(i)") in 20% of documents.
_LOWER_I = re.compile(
    r"(?<![A-Za-z0-9_.])i(?:'m|'ve|'d|'ll|\u2019m|\u2019ve|\u2019d|\u2019ll)(?![A-Za-z])"
    r"|(?<![A-Za-z0-9_.(])i (?:am|was|think|have|had|dont|don't|can|cant|can't|will|"
    r"would|just|really|guess|know|want|need|got|feel|did|mean|said|tried|found|"
    r"wanted|thought|love|hate|like|use|used|see|saw)\b")
# "!!" as emphasis, not as csh history expansion: a letter must precede it.
_REPEATED_PUNCT = re.compile(r"(?<=[A-Za-z])(?:[!?]{2,})")
_INFORMAL = frozenset("""
    lol lmao tbh idk imo imho btw gonna wanna kinda sorta dunno gotta yeah yep nope
    haha hahaha ugh omg thx pls plz
""".split())
_APOSTROPHES = re.compile(r"[A-Za-z](['\u2019])[A-Za-z]")


def _count_patterns(patterns: Sequence["re.Pattern[str]"], text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def _per_thousand(count: int, n_words: int) -> float:
    return 1000.0 * count / n_words if n_words else 0.0


def assistant_echo_count(text: str) -> int:
    return _count_patterns(ASSISTANT_ECHO, text)


def hedging_count(text: str) -> int:
    return _count_patterns(HEDGING, text)


def lexicon_count(words: Sequence[str]) -> int:
    return sum(1 for w in words if w in LLM_LEXICON)


def transition_share(sentences: Sequence[str]) -> float:
    """Share of sentences that open with a discourse adverb."""
    if not sentences:
        return 0.0
    hits = 0
    for sentence in sentences:
        first = re.match(r"[A-Za-z]+", sentence.lstrip(" \t\"'*#>-•"))
        if first and first.group(0).lower() in TRANSITIONS:
            hits += 1
    return hits / float(len(sentences))


def bold_lead_share(lines: Sequence[str]) -> float:
    if not lines:
        return 0.0
    return sum(1 for line in lines if _BOLD_LEAD.match(line)) / float(len(lines))


def human_noise_count(view: TextView) -> int:
    """Marks of hasty human typing: lowercase pronoun "i", doubled "!"/"?" after a word,
    informal tokens, and straight/curly apostrophes mixed within one document."""
    count = len(_LOWER_I.findall(view.raw))
    count += len(_REPEATED_PUNCT.findall(view.raw))
    count += sum(1 for w in view.words if w in _INFORMAL)
    if len({m.group(1) for m in _APOSTROPHES.finditer(view.raw)}) > 1:
        count += 1
    return count


@register("artifact_profile", FAMILY_ARTIFACT)
def artifact_profile(doc: Document, view: TextView) -> List[Signal]:
    """Emit artifact rates for one document. Grouped because every rate shares the
    same denominators (words, sentences, lines)."""
    n_words = view.n_words
    if n_words == 0:
        return []
    text = view.raw

    signals = [
        Signal(
            name="assistant_echo_rate",
            value=_per_thousand(assistant_echo_count(text), n_words),
            family=FAMILY_ARTIFACT,
            direction=1,
            description="Chat-assistant phrasing per 1,000 words: refusals, cutoff disclaimers, 'I hope this helps'.",
        ),
        Signal(
            name="hedging_rate",
            value=_per_thousand(hedging_count(text), n_words),
            family=FAMILY_ARTIFACT,
            direction=1,
            description="Formulaic hedging and scaffolding phrases per 1,000 words: 'it's important to note', 'plays a crucial role'.",
        ),
        Signal(
            name="llm_lexicon_rate",
            value=_per_thousand(lexicon_count(view.words), n_words),
            family=FAMILY_ARTIFACT,
            direction=1,
            description="Words with documented excess use in LLM-era writing, per 1,000 words: 'delve', 'tapestry', 'pivotal'.",
        ),
        Signal(
            name="human_noise_rate",
            value=_per_thousand(human_noise_count(view), n_words),
            family=FAMILY_ARTIFACT,
            direction=-1,
            description="Marks of hasty human typing per 1,000 words. Lower means unusually clean text.",
        ),
    ]
    if view.sentences:
        signals.append(
            Signal(
                name="transition_rate",
                value=transition_share(view.sentences),
                family=FAMILY_ARTIFACT,
                direction=1,
                description="Share of sentences opening with a discourse marker: 'Moreover', 'Furthermore', 'Additionally'.",
            )
        )
    if len(view.lines) >= MIN_LINES_FOR_RATIO:
        signals.append(
            Signal(
                name="bold_lead_rate",
                value=bold_lead_share(view.lines),
                family=FAMILY_ARTIFACT,
                direction=1,
                description="Share of lines opening with a bold lead-in ('**Scalability:** ...'), the assistant markdown habit.",
            )
        )
    return signals
