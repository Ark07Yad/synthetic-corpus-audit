# Evaluation

How much synthetic text chaff catches, at what cost, and where it fails. Everything
here is reproducible with `benchmarks/evaluate.py`. The corpora are rebuilt locally and
nothing from them is committed; this file reports only numbers and document IDs.

## Summary

- **Older open-weight base models: caught well.** On the MAGE benchmark, 85–100% of
  BLOOM, GPT-J/NeoX, FLAN-T5 and OPT output is flagged (AUC 0.92–0.99), as is 65–68% of
  LLaMA-1 output.
- **Modern instruction-tuned models: caught poorly.** OpenAI GPT-3.5/davinci/GPT-4 output
  is 30–40% flagged (AUC 0.61–0.64). Claude-written technical documentation is 19%
  flagged against a 9% false-positive rate (AUC 0.78 for ranking, weak at the threshold).
- **Its founding premise was half right.** Older models do produce compressed,
  repetitive text. Modern ones produce text that is *more* diverse than human writing of
  the same genre. The one-sided design ranked Claude's documentation as more human than
  human documentation (AUC 0.04). Scoring atypicality in **both** directions fixed that
  (0.78), and the change was confirmed on held-out data (0.61 → 0.87).
- **False positives depend on genre.** Calibrated on technical reference text, 0.92% of
  clean documents reach `LIKELY_SYNTHETIC`. On other genres it's 0–5.1%, and the
  flagged share (`SUSPECT` or above) runs 11–38%. Changelogs and licence boilerplate are
  the characteristic human false positives.
- **Use `--reference` with a genre-matched human corpus.** In corpus-relative mode,
  detection degrades as contamination rises. Reference mode doesn't.
- **Calibrate on that corpus too.** `chaff calibrate` on genre-matched human text cut
  MAGE's human false-positive rate from 17.7% to 7.3% flagged, with recall rising
  (72.9% → 76.0%).

chaff is a triage instrument for **older and open-weight model contamination and heavy
templated contamination**. It is not a reliable detector of frontier-model text.

## Setup

| Source | Synthetic | Human | Notes |
|---|---|---|---|
| **tech** | 80 by blind Claude subagents: 40 man pages, 40 Python-module docstring sets | 871 held-out docs from the human baseline (708 man pages, 163 stdlib modules) | Genre-matched. The writers never saw chaff, its detectors, or the purpose of the task. Fictional tools and modules, so no output paraphrases a baseline document. |
| **literary** | 40 blind-Claude essays | 315 chunks of 10 Project Gutenberg books (pre-1928) | **Era-confounded**: modern essays against 19th-century prose. Reported, but not used for conclusions. |
| **MAGE** (Li et al., ACL 2024) | 1,500 from 24 generators in 7 families | 855 from 10 domains | Documents of ≥ 250 words only (MAGE's median is 115). |
| **MAGE GPT-4** | 345 GPT-4 | 144 | MAGE's unseen-domain set: CNN, IMDB, PubMed, DialogSum. |
| **MAGE confirm** | 999 | 441 | Fresh MAGE documents built *after* the two-sided hypothesis was formed, sharing no text with any analysed document. |

**Modes.** *Corpus-relative* is the default: each document is compared with the rest of
the corpus. *Reference* compares each document with a disjoint half of the same source's
human text (`--reference`). *Stratify* normalises within a genre or domain field
(`--stratify-by`). "Flagged" means `SUSPECT` or `LIKELY_SYNTHETIC`.

**Non-English documents.** Since phase 7, chaff leaves evidently non-English documents
`UNSCORED` (they are excluded from every rate here). That guard found two inside MAGE's
English human text — a French Yelp review and a Welsh XSum article — which earlier
versions scored. One MAGE human and one confirm-set human document are gone from the
counts as a result. All figures below are from the phase 7 re-run. Only the MAGE results
moved: the headline rows by at most half a point, except the confirm set's `LIKELY` FPR
(2.3% → 1.8%, i.e. 10 → 8 documents); small per-family rows such as FLAN-T5 (n = 17) by
one document's worth.

## Results

Recommended mode: reference with stratification.

| Experiment | Recall flagged | Recall LIKELY | FPR flagged | FPR LIKELY | AUC |
|---|---|---|---|---|---|
| tech: reference + stratify genre | 18.8% | 0.0% | 9.3% | 1.1% | 0.778 |
| MAGE: reference + stratify domain | 77.7% | 44.9% | 21.3% | 1.8% | 0.885 |
| MAGE GPT-4: reference + stratify domain | 67.5% | 28.1% | 17.4% | 2.8% | 0.774 |
| **MAGE confirm: reference + stratify domain** | **72.9%** | **42.4%** | **17.7%** | **1.8%** | **0.874** |
| literary: reference *(era-confounded)* | 100% | 27.5% | 11.7% | 0.6% | 0.921 |

Default corpus-relative mode:

| Experiment | Recall flagged | Recall LIKELY | FPR flagged | FPR LIKELY | AUC |
|---|---|---|---|---|---|
| tech | 3.8% | 0.0% | 9.2% | 0.5% | 0.406 |
| MAGE | 29.4% | 6.1% | 12.4% | 0.1% | 0.782 |
| MAGE GPT-4 | 63.5% | 5.5% | 12.5% | 1.4% | 0.698 |
| MAGE confirm | 30.5% | 9.0% | 12.0% | 0.2% | 0.760 |

**Precision depends on prevalence.** MAGE's evaluation set is 64% synthetic, so its
97.8% precision at `LIKELY` is inflated. Using the confirmation set's rates (42.4%
recall, 1.8% FPR at `LIKELY`), precision would be about **55% at 5% contamination** and
about **85% at 20%**. The FPR rests on 8 of 441 documents, so treat both as rough.

### By generator family (MAGE, reference + stratify domain)

| Family | Recall flagged | Recall LIKELY | AUC | Confirm set: recall flagged / AUC |
|---|---|---|---|---|
| BLOOM | 100% | 67.8% | 0.981 | 100% / 0.980 |
| EleutherAI (GPT-J, NeoX) | 99.1% | 79.3% | 0.986 | 100% / 0.984 |
| FLAN-T5 (n = 17) | 88.2% | 47.1% | 0.952 | 100% / 0.987 |
| OPT | 89.9% | 49.5% | 0.930 | 85.2% / 0.922 |
| GLM-130B | 86.6% | 59.0% | 0.940 | 83.8% / 0.936 |
| LLaMA-1 (7–65B) | 68.4% | 38.7% | 0.879 | 64.9% / 0.878 |
| **OpenAI** (GPT-3.5, davinci-002/003, GPT-4) | **39.9%** | **7.3%** | **0.643** | **30.7% / 0.614** |

The ordering follows how heavily instruction-tuned each generator is. The confirmation
set reproduces it family by family.

## The finding that changed the design

chaff was built on the premise that synthetic text has a **compressed, repetitive
vocabulary** (CORE_PROBLEM §3.1). Per-signal AUC on the raw values, with each signal's
declared direction applied, shows where that holds:

| Signal | tech (Claude) | MAGE (older) | MAGE GPT-4 |
|---|---|---|---|
| MTLD (lexical diversity) | **0.12** | 0.49 | 0.40 |
| Yule's K (repetition) | **0.23** | 0.51 | 0.64 |
| hapax ratio | 0.38 | 0.45 | **0.20** |
| 4-gram repetition | 0.33 | 0.59 | 0.34 |
| 8-gram repetition | 0.33 | 0.65 | 0.33 |
| step novelty | 0.28 | 0.54 | 0.37 |
| mean surprisal | 0.21 | 0.60 | 0.70 |

Values below 0.5 mean the signal points the wrong way. Modern models write with
**richer vocabulary and less repetition** than the human text of the same genre. That's
the opposite of the premise, and it's why one-sided scoring *inverted*.

So the distributional, surprisal and reasoning families now flag documents that are
atypical for their genre **in either direction**; the artifact family stays one-sided.
The change was pre-registered and tested on the MAGE confirmation set:

| AUC | One-sided (phase 5) | Two-sided (phase 6) |
|---|---|---|
| tech (reference + stratify) | 0.036 | **0.778** |
| MAGE (reference + stratify) | 0.583 | **0.885** |
| MAGE GPT-4 (reference + stratify) | 0.635 | **0.774** |
| **MAGE confirm, never analysed** (reference + stratify) | 0.605 | **0.872** |
| MAGE confirm (corpus-relative) | 0.604 | **0.757** |

(Both columns as measured in phase 6, before the language guard; the phase 7 re-run gives
0.874 and 0.760.)

Recalibrated on the human baseline, the two-sided design keeps the held-out
`LIKELY_SYNTHETIC` rate at 0.92%, lowers `SUSPECT` from 8.15% to 6.66%, and every
held-out `SUSPECT` document now scores ≥ 80.

The hypothesis was formed from the tech, MAGE and GPT-4 results. The confirmation set is
the only evidence here that it generalises, and a single model family can't settle it.

## Contamination sweep (tech)

The share of synthetic documents in a corpus was varied by subsampling:

| Synthetic share | Corpus-relative AUC | Corpus-relative FPR flagged | Reference AUC | Reference FPR flagged |
|---|---|---|---|---|
| 2% | 0.447 | 7.8% | 0.603 | 9.3% |
| 8% | 0.405 | 8.6% | 0.671 | 9.3% |
| 20% | 0.343 | 11.2% | 0.668 | 8.1% |
| 50% | 0.295 | 16.2% | 0.713 | 11.2% |

As contamination rises, corpus-relative scoring gets worse, because the contamination
becomes part of what "typical" means, and the false-positive rate on human documents
climbs with it. Reference mode stays stable. **When a trusted human corpus exists, use
it.**

## False positives by genre

| Human text | Docs | Flagged | LIKELY |
|---|---|---|---|
| Man pages (held-out baseline) | 708 | 8.8% | 1.1% |
| Python stdlib docstrings (held-out baseline) | 163 | 11.7% | 1.2% |
| MAGE squad / wp / xsum / yelp | 27–150 each | 11–18% | 0–1.3% |
| MAGE eli5 / cmv | 150 each | 22–30% | 0.7–3.3% |
| MAGE science abstracts | 79 | **38%** | **5.1%** |
| Licence texts | 9 | **7 of 9** SUSPECT, 0 LIKELY | 0 |

**Licences** (all nine, scored against the technical reference): the Python 3.9
`LICENSE`, and the `six`, `setuptools`, `wheel`, `future`, `macholib` and `pip` licences
are all `SUSPECT` through surprisal: boilerplate is highly predictable. `altgraph`'s
licence and vim's `editorconfig` licence are `CLEAN`. Nine documents can't support a
rate; they're listed individually for that reason. The corroboration rule is what keeps
every one of them out of `LIKELY_SYNTHETIC`.

## Failure cases (tech, reference + stratify)

Highest-scoring human documents:

| Document | Tier | Score | Families fired |
|---|---|---|---|
| `man1/perldocstyle.1` (Perl documentation style guide) | LIKELY | 100.0 | artifact, distributional, reasoning |
| `man7/Accelerate.7` (Apple framework overview) | LIKELY | 100.0 | artifact, surprisal |
| `man1/perl5123delta.1` (Perl release notes) | LIKELY | 99.5 | distributional, surprisal |
| `man1/perl5181delta.1` (Perl release notes) | SUSPECT | 99.5 | surprisal |

Release notes, a style guide and a marketing-flavoured framework overview are exactly
the templated or promotional human genres CORE_PROBLEM predicted.

Lowest-scoring synthetic documents: `moduledoc/b13_versionspec` (16.4),
`moduledoc/b06_xmlpath` (29.9), `man/b11_timeout2` (31.3), `man/b14_encwrap` (37.8).
All are `CLEAN`, with no family firing. They are simply typical of their genre.

## Calibrating on your own genre (phase 7)

The shipped calibration was fit on technical reference text. `chaff calibrate` fits the
same procedure to any trusted human corpus. To test it, the MAGE human reference (1,495
documents after the language guard, given fresh ids) was calibrated with
`--stratify-by domain`, then the never-analysed confirmation set was scored with that
calibration instead of the shipped one:

| Calibration | Recall flagged | Recall LIKELY | FPR flagged | FPR LIKELY | AUC |
|---|---|---|---|---|---|
| shipped (technical text) | 72.9% | 42.4% | 17.7% | 1.8% | 0.874 |
| **`chaff calibrate` on MAGE human text** | **76.0%** | 38.6% | **7.3%** | 1.8% | **0.899** |

A matched calibration cut false flags by more than half while flagging *more* synthetic
text. At `LIKELY` the two are about level.

**Small calibrations are knife-edge.** The calibration's own held-out half reached 1.67%
`LIKELY` against a 1% target, and `chaff calibrate` said so. The cause is granularity:
before the language guard removed one French document, half A sat at 8 of 778 `LIKELY`
at alpha 0.025 — just over 1% — so alpha 0.02 was chosen and half B came in at 1.11%.
One document moved the threshold a grid step. With fewer than ~1,000 documents per half,
expect held-out rates to differ from the target by this much.

## Near-duplicates (phase 7)

`chaff dedup` clusters documents whose word-5-gram Jaccard similarity is estimated at
≥ 0.8. On planted families (4 copies of each of 50 documents, among 800 unique ones), it
recovered every copy with 1% edits (true Jaccard ~0.91) and 78% of copies with 2% edits
(~0.82, right at the threshold, where the ±0.05 estimate decides). No unique document was
ever clustered and every cluster was pure. On the 1,748-document human baseline it found
7 genuine near-copy pairs, such as successive Perl release notes and `qmgr`/`oqmgr`.
100,000 documents take 16.7 s and 150 MB; a flood of 5,000 copies of one template costs
5,074 similarity checks.

## Scale

100,000 documents (~25M tokens), end to end with `chaff score`: **5.1 minutes, 2.2 GB
peak memory**, single-threaded, two passes over the corpus.

## Caveats

- **Single-family synthetic text for tech and literary.** The blind subagents are all
  Claude. MAGE supplies the cross-model evidence.
- **MAGE's generators are older and continuation-prompted.** Its GPT-4 set is small (345)
  and domain-shifted. Neither is a sample of the current web.
- **The literary comparison is confounded by era** and supports no conclusions.
- **The shipped calibration is on technical reference text.** The false-positive rates
  above show it doesn't transfer. Run `chaff calibrate` on a trusted human corpus of your
  own genre before relying on the tiers.
- **Legal text is nine documents.**
