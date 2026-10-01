# chaff

**Synthetic corpus contamination & token entropy auditing for LLM pre-training data.**

Find AI-generated text in a pre-training corpus *before* it poisons your tokenizer.

[![ci](https://github.com/Ark07Yad/synthetic-corpus-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/Ark07Yad/synthetic-corpus-audit/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![tests](https://img.shields.io/badge/tests-254-brightgreen)
![dependencies](https://img.shields.io/badge/runtime%20dependencies-0-brightgreen)
![status](https://img.shields.io/badge/status-evaluated-blue)

---

## The problem in three sentences

The open web is filling with LLM output, and web crawls do not label it. Models trained
on that text fit the previous model's narrowed distribution instead of the real one —
the rare words, unusual constructions and genuine reasoning get sampled away, generation
by generation. That is **model collapse**, and by the time you can measure it in your
weights, it has been sitting in your corpus for months.

chaff reads a corpus and ranks every document by how far its *distributional profile*
deviates from natural language — then tells you exactly which measurements drove the
score.

The damage is done **before** anyone trains anything — it is baked into the corpus. And
it is cheapest to remove while it is still a file: after tokenization it is opaque
integer IDs, and after pre-training it is model weights.

## Results, up front

Evaluated on three independent sources of model-written text, each against human text of
the same kind (full report: [`benchmarks/EVALUATION.md`](benchmarks/EVALUATION.md)):

| Model-written text | Flagged | False positives (human) | AUC |
|---|---|---|---|
| Older open-weight models: BLOOM, GPT-J/NeoX, FLAN-T5, OPT, GLM (MAGE) | **84–100%** | 18–21% flagged, ~2% `LIKELY` | 0.92–0.99 |
| LLaMA-1, 7–65B (MAGE) | 65–68% | same | 0.88 |
| OpenAI GPT-3.5 / davinci / GPT-4 (MAGE) | **30–40%** | same | 0.61–0.64 |
| Claude, technical documentation (blind-written) | **19%** | 9% | 0.78 |

These figures are for the recommended mode, `--reference` with a genre-matched human
corpus. On 1,440 MAGE documents held back from every analysis: **AUC 0.87**, 72.9%
flagged at a 17.7% false-positive rate, 42.4% `LIKELY_SYNTHETIC` at 1.8%. **Calibrated
on matched human text with `chaff calibrate`, the false-positive rate drops to 7.3% while
76.0% is flagged.**

**What that means.** chaff is a useful triage tool for contamination by older and
open-weight models and for heavily templated synthetic text. It is **not** a reliable
detector of frontier-model output. False-positive rates depend on genre (11–38%
flagged, 0–5% `LIKELY`), and changelogs and licence boilerplate are the characteristic
human false positives.

**The finding that changed the design.** chaff was built on the premise that synthetic
text has a compressed, repetitive vocabulary. That holds for older models. Modern
instruction-tuned models write with *richer* vocabulary than the human text of the same
genre. The original one-sided scoring ranked Claude's technical docs as **more human than
human docs** (AUC 0.04). chaff now flags documents that are atypical for their genre *in
either direction*. That change was pre-registered and confirmed on held-out data, where
it took AUC from 0.61 to 0.87.

## Why this is not just a quality filter

Standard pre-training filters catch **low-quality** text: gibberish, spam, boilerplate,
bad language ID. Synthetic text is not low-quality. It is grammatical, on-topic and
clean. It is **atypical**, in a direction that depends on the generator. Different
failure mode, different detector.

| | Low-quality text | Synthetic text (as measured) |
|---|---|---|
| Grammatical | often not | yes |
| On-topic | often not | yes |
| Caught by perplexity cutoffs | yes | no; it often scores *better* than human text |
| Lexical tail | intact | truncated for older base models, **richer than human** for modern instruction-tuned ones |

## Install

```bash
git clone https://github.com/Ark07Yad/synthetic-corpus-audit.git
cd synthetic-corpus-audit
pip install -e .
```

Zero runtime dependencies. No model download, no GPU, no build step — it is meant to be
droppable onto the data node that already holds your dump. CI enforces this.

## Usage

```bash
chaff score corpus.jsonl --out scores.jsonl --report report.html   # tier, score and evidence per document
chaff explain scores.jsonl <doc_id>                                  # why one document got its tier
chaff report scores.jsonl --format md                                # re-render the run as md / json / html
chaff score crawl.jsonl --reference trusted.jsonl --out scores.jsonl # normalise against a trusted corpus (recommended)
chaff score crawl.jsonl --stratify-by domain --out scores.jsonl      # normalise within each value of a record field

chaff calibrate trusted.jsonl --out mine.json   # thresholds from a trusted human corpus of your own genre
chaff score crawl.jsonl --reference trusted.jsonl --calibration mine.json
chaff dedup crawl.jsonl --out clusters.jsonl    # near-duplicate clusters: template floods

chaff profile corpus.jsonl --out rows.jsonl  # raw signals per document, no scoring
chaff inspect corpus.jsonl -n 5              # see individual docs as chaff sees them
chaff families                               # what is implemented
```

`scores.jsonl` has one line per document and joins back to your corpus on `doc_id`:

```json
{"doc_id": "s_reasoning_01", "tier": "LIKELY_SYNTHETIC", "score": 98.0,
 "families_fired": ["reasoning", "distributional"],
 "evidence": [{"signal": "stalled_step_ratio", "value": 0.3, "z": 4.65, "basis": "corpus", ...}]}
```

The surprisal family reads the corpus **twice**: once to build a language model, once to
score. Everything else needs one pass, and chaff only makes the second pass when a family
that needs it is active. Input from stdin can't be read twice, so the surprisal family
is skipped there, with a note saying so, unless `--reference` supplies the model.

Reads `.jsonl`, `.jsonl.gz`, `.ndjson`, plain text files, directories, or stdin.
The text field is autodetected across Common Crawl / C4 / RedPajama / Pile conventions,
or set it with `--text-field`.

## Using it on your own corpus

Three tools for turning a general instrument into one tuned to your data:

**`chaff calibrate`: thresholds from your genre.** The shipped calibration was fit on
technical reference text, and false-positive rates don't transfer between genres. Give
`calibrate` a trusted human corpus of your own genre (200 documents minimum; it warns
below 1,000, and more is better still). It fits on half of it, reports the held-out rate on the other half, and writes
a file for `chaff score --calibration`. On MAGE's held-back documents, calibrating on
matched human text cut false flags from 17.7% to 7.3% while flagging *more* synthetic
text (72.9% → 76.0%). It refuses a corpus too small to calibrate on, and warns when the
held-out rate overshoots the target.

**`chaff dedup`: template floods.** Per-document scores can't see a pipeline that produced
5,000 near-identical articles from one prompt: each one looks fine alone. `dedup`
clusters near-duplicates (estimated Jaccard ≥ 0.8 on word 5-grams, by MinHash and LSH)
and writes each document's cluster. 100k documents take 17 seconds; a flood of 5,000
copies costs ~5,000 comparisons, not 12 million.

**Language guard.** Every reference distribution chaff ships is English, so a German page
scored against them gets a confident, meaningless score. Documents with positive evidence
of another language (mostly non-Latin script, or another language's function words
outnumbering English ones) are now `UNSCORED` with the reason in `scores.jsonl` and
`chaff explain`. On 7,478 English documents it flagged none; on ~1,400 localized macOS
help and licence files in 40+ languages it caught 98.7%. It also found a French review and
a Welsh news story inside MAGE's "English" human text. `--allow-non-english` turns it off.

## What it measures

Four **independent** metric families. Independence is the point: a document is only
tiered `LIKELY_SYNTHETIC` when at least two families fire, which is what keeps templated
human writing out of the results.

| Family | Signals | Phase |
|---|---|---|
| **distributional** | Zipf slope, frequency-spectrum slope, Heaps' β, branching entropy, hapax ratio, MTLD, Yule's K, 4/8-gram repetition, zlib compressibility | **2 — done** |
| **surprisal** | mean surprisal, surprisal spread at word and sentence scale (gated on model adequacy), recycled-span runs | **3 — done** |
| **artifact** | assistant echoes, hedging scaffolds, bold-label markdown, LLM-era lexicon, discourse-adverb transitions, *absence* of human typing noise | **4 — done** |
| **reasoning** | windowed step novelty, stalled-step ratio | **4 — done** |

### How a score is made

Every number below was measured on held-out data that took no part in setting it.

1. **Normalise** each signal to a z-score. Continuous signals are compared with documents
   *of similar length* in the corpus: on human text, eight distributional signals
   correlate with length at |ρ| 0.48–0.95, so an unstratified "corpus-relative" z would
   mostly rank documents by length. Length strata bring that to |ρ| ≤ 0.06. Artifact
   signals are compared with **guaranteed-human text** instead, because they're zero for
   most documents, and because in a crawl that's 20% synthetic, hedging would otherwise
   look "normal".
2. **Combine within each family.** Signals measuring the same property are averaged; the
   artifact family takes its strongest tell, corrected for having looked at six.
3. **Tier.** Each family has thresholds calibrated on 877 human documents. The
   distributional, surprisal and reasoning families fire when a document is atypical for
   its genre **in either direction**; the artifact family only when a tell is present.
   **Two or more families firing → `LIKELY_SYNTHETIC`**; one → `SUSPECT`; none →
   `CLEAN`. No single family, however strong, can produce `LIKELY_SYNTHETIC`.
4. **Score 0–100** = the document's combined evidence (Fisher's method across families)
   as a **percentile among clean human documents**. 50 is typical human text; 99 means
   more evidence than 99% of it.

On 871 held-out human documents:

| | Result |
|---|---|
| tiered `LIKELY_SYNTHETIC` | **0.92%** (target ≤ 1%) |
| tiered `SUSPECT` | 6.66% |
| median score / 95th percentile | 53.2 / 96.5 |
| lowest score of any `SUSPECT` document | 80.1 (so tier and score agree) |

That calibration is on technical reference text. On other genres the false-positive rate
is higher (see Results above), so re-calibrate on a trusted corpus of your own genre
before relying on the tiers.

The corroboration rule rests on the families being independent, and they aren't fully:
on clean text, two families fire together **3.6× more often** than independence would
predict. Thresholds are calibrated on that *measured* joint rate, so the 0.92% already
includes it. A tool that assumed independence would have understated its false-positive
rate by that factor.

### The artifact and reasoning families, validated against guaranteed-human text

Artifact detectors can't be validated on text I wrote myself. If I write "delve" and then
detect "delve", I've only proven the regex works. Their real risk is false positives,
so they're measured against text that is **human by date**: Python 3.9 standard-library
docstrings (released June 2021, before ChatGPT) plus 1,439 man pages, 1,748 documents in
all. Detector lists were pruned on one half and evaluated on the other.

| Detector | Fires on held-out human docs |
|---|---|
| assistant echoes ("As an AI language model", "I hope this helps") | **0.0%** |
| bold-label markdown (`**Scalability:**`) | **0.0%** |
| LLM-era lexicon | **0.7%** (was 11% before pruning) |
| hedging scaffolds | 2.6% |
| transition adverbs | 9.8% |

The lexicon result is the instructive one. Words like "underscore", "realm" and "harness"
are LLM tells in prose and **domain vocabulary** in technical text (the `_` character,
Kerberos, test harnesses), and they fired at LLM-like rates on human documents. The same
benchmark checks the assumption the whole scoring design rests on: that the metric
families are independent. A reasoning signal turned out to be re-measuring lexical
repetition (|ρ| 0.74 with the distributional family) and was removed; the strongest
remaining cross-family correlation is 0.49. Details and caveats:
[`benchmarks/README.md`](benchmarks/README.md).

### The surprisal family, and a thesis this project had to correct

The surprisal family is built on a **corpus-internal trigram language model**, not a
downloaded transformer. Every document is scored against the corpus *minus itself*, by
subtracting its own counts at query time. The tests assert that this gives exactly the
scores of a model retrained without the document, to the last bit. It costs one extra
pass and keeps the tool dependency-free. Most of the model's entries occur exactly once,
and under leave-one-out those are provably dead weight, so they're pruned **without
changing a single score**: 74% of entries on Zipfian text.

This project started from a popular claim: human text is *bursty*, so the **spread** of
surprisal should separate it from generated text better than the **mean** does. That was
tested on controlled corpora (one random language; "human" documents sampled from its
full distribution, "synthetic" ones through top-p truncation) and it turned out to be
wrong in a specific way:

- **The mean is the robust signal.** It separated the two at every model quality tested,
  with AUC 0.83 at the worst and 1.00 at the best.
- **Spread only works on a well-estimated model**, and below that it doesn't just weaken,
  it **inverts** (AUC 0.20–0.38), because backoff noise dominates the spread of *both*
  classes. So chaff measures its own model's adequacy (bigram coverage) and only emits
  spread signals above thresholds read off that experiment. `chaff profile` reports which
  side of the line your corpus is on.

The toy language has no topic structure, so it couldn't settle whether *real* human text
is bursty because of topic shifts. Phase 6 measured that on real data, and burstiness is
**weak**: surprisal spread reached AUC 0.44 on Claude-written technical docs and 0.59 on
MAGE. The level of surprisal did better where synthetic and human text share a domain
(0.60 on MAGE, 0.70 on GPT-4).

## Status: complete, and evaluated

| Phase | Scope | State |
|---|---|---|
| 1 | Foundation: tokenization, streaming corpus I/O, metric contract, CLI, sample corpus, CI | **done** |
| 2 | Distributional family — 3 extractors, up to 9 signals per document | **done** |
| 3 | Surprisal family — two-pass pipeline, leave-one-out trigram LM, coverage-gated signals | **done** |
| 4 | Artifact + reasoning families, false-positive benchmark on guaranteed-human text | **done** |
| 5 | Fusion: length-stratified normalisation, calibrated tiers, Fisher score, `score` / `explain` / `report` | **done** |
| 6 | Evaluation on real model output (MAGE, 25 generators; blind Claude), two-sided scoring, `--stratify-by`, 100k-document scale run | **done** |
| 7 | Usable on your own corpus: `chaff calibrate`, `chaff dedup`, language guard | **done** |

**Scale:** 100,000 documents (~25M tokens) scored end to end in 5.1 minutes at 2.2 GB peak
memory, single-threaded, two passes over the corpus.

Every signal is unit-tested against controlled corpora with known properties, its
false-positive behaviour is measured on 1,748 guaranteed-human documents, and its
true-positive behaviour on MAGE and on blind Claude-written text
([`benchmarks/EVALUATION.md`](benchmarks/EVALUATION.md)).

## Limitations, stated up front

- **Weak on frontier models.** 30–40% of OpenAI GPT-3.5/GPT-4 output and 19% of
  Claude-written technical docs are flagged. Don't use chaff as your only filter against
  modern-model contamination.
- **False positives depend on genre.** From 11% flagged (reference prose) to 38%
  (science abstracts). Run `chaff calibrate` on a trusted human corpus of your own genre.
- **English only.** Documents that are evidently in another language are `UNSCORED`
  with a reason rather than scored against English baselines. The guard misses
  languages it has no word list for (Lithuanian, for one) and some short Polish,
  Slovenian and Finnish pages.
- **Small calibrations are noisy.** With ~1,500 trusted documents, one document can move
  the chosen threshold a step. `chaff calibrate` reports the held-out rate and warns when
  it overshoots the target. Several thousand trusted documents make it stable.
- **Two-sided scoring has a cost.** Unusually *rich* human writing, such as a casual post
  where every sentence brings new content, can be flagged `SUSPECT` when compared against
  formal text.
- **Corpus-relative scoring penalises minority genres and degrades under
  contamination.** A genre that's rare in the corpus looks unusual, and at 50%
  contamination the corpus-relative AUC falls to 0.30 while reference mode holds at 0.71.
  Use `--reference` with a genre-matched trusted corpus, or `--stratify-by` a genre field
  (it halved the minority-genre false positives on the baseline).
- **This is a triage instrument, not an AI detector.** It ranks documents for review. It
  does not prove any individual document was machine-generated, and it never claims which
  model wrote it.
- **It will flag templated human writing.** Changelogs, API references and legal
  boilerplate are genuinely low-entropy. The corroboration rule blunts this; it does not
  eliminate it. Two such documents are in the sample corpus specifically to keep the
  metrics honest.
- **Not for academic integrity.** Wrong granularity, and the false-positive cost falls on
  a person rather than on a row in a dataset. Please do not use it that way.
- Single-document scores are noisy below ~50 words and are reported, not scored.
- **Near-copies right at the dedup threshold are found only some of the time.** The
  similarity is an estimate (±0.05 around 0.8), so pairs that sit at the cutoff fall
  either side of it. Pairs comfortably above it are found reliably.
- **Changelogs and licence boilerplate are the characteristic human false positives.**
  7 of 9 licence texts are `SUSPECT`, and the top human false positives are release
  notes. The corroboration rule keeps them out of `LIKELY_SYNTHETIC`, not out of
  `SUSPECT`.
- **The families are not fully independent.** The strongest measured cross-family
  correlation on human text is 0.49, so two families firing together is weaker evidence
  than two truly independent tests would be.
- **Corpus-relative surprisal measures typicality within your corpus.** Boilerplate only
  looks predictable if the corpus contains lots of similar boilerplate. That's usually
  true of a web crawl full of contracts and cookie banners, and it's exactly where the
  templated-human false positive is expected. `--reference` with a trusted corpus is the
  better setting when one exists.
- **Signals withhold themselves rather than guess.** Several metrics need a minimum
  document length to mean anything — the frequency-spectrum slope needs ~2,000 words,
  measured rather than assumed — and return nothing below it. A short document will
  legitimately produce fewer signals than a long one.

## Repository map

| Path | What it is |
|---|---|
| `src/chaff/` | The package |
| `src/chaff/metrics/` | The `Signal` contract and the metric families |
| `tests/` | 254 tests |
| [`benchmarks/EVALUATION.md`](benchmarks/EVALUATION.md) | true-positive evaluation: MAGE, blind Claude, contamination sweep, failure cases |
| [`benchmarks/`](benchmarks/) | false-positive benchmark on guaranteed-human text |
| [`data/samples/`](data/samples/) | 10 labelled documents, including deliberate hard cases |

## Licence

MIT — see [LICENSE](LICENSE).
