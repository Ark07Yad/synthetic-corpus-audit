# chaff

**Synthetic corpus contamination & token entropy auditing for LLM pre-training data.**

Find AI-generated text in a pre-training corpus *before* it poisons your tokenizer.

[![ci](https://github.com/Ark07Yad/synthetic-corpus-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/Ark07Yad/synthetic-corpus-audit/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![tests](https://img.shields.io/badge/tests-108-brightgreen)
![dependencies](https://img.shields.io/badge/runtime%20dependencies-0-brightgreen)
![status](https://img.shields.io/badge/status-phase%203%20of%206-orange)

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

## Why this is not just a quality filter

Standard pre-training filters catch **low-quality** text: gibberish, spam, boilerplate,
bad language ID. Synthetic text is not low-quality. It is grammatical, on-topic and
clean. It is **low-entropy**. Different failure mode, different detector.

| | Low-quality text | Synthetic text |
|---|---|---|
| Grammatical | often not | yes |
| On-topic | often not | yes |
| Caught by perplexity cutoffs | yes | **no — it scores *better* than human text** |
| Lexical tail | intact | truncated by top-p/top-k sampling |
| Surprisal over tokens | noisy | **flat** |

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
chaff profile corpus.jsonl.gz              # corpus shape and signal summary
chaff profile corpus.jsonl --out rows.jsonl  # per-document rows, joins back on doc_id
chaff inspect corpus.jsonl -n 5            # see individual docs as chaff sees them
chaff profile crawl.jsonl --reference wiki.jsonl  # score against a trusted corpus instead
chaff families                             # what is implemented, and what is coming
```

The surprisal family reads the corpus **twice**: once to build a language model, once to
score. Everything else needs one pass, and chaff only makes the second pass when a family
that needs it is active. Input from stdin can't be read twice, so the surprisal family
is skipped there, with a note saying so, unless `--reference` supplies the model.

Reads `.jsonl`, `.jsonl.gz`, `.ndjson`, plain text files, directories, or stdin.
The text field is autodetected across Common Crawl / C4 / RedPajama / Pile conventions,
or set it with `--text-field`.

## What it measures

Four **independent** metric families. Independence is the point: a document is only
tiered `LIKELY_SYNTHETIC` when at least two families fire, which is what keeps templated
human writing out of the results.

| Family | Signals | Phase |
|---|---|---|
| **distributional** | Zipf slope, frequency-spectrum slope, Heaps' β, branching entropy, hapax ratio, MTLD, Yule's K, 4/8-gram repetition, zlib compressibility | **2 — done** |
| **surprisal** | mean surprisal, surprisal spread at word and sentence scale (gated on model adequacy), recycled-span runs | **3 — done** |
| **artifact** | system-prompt echoes, hedging scaffolds, markdown watermarks, overused lexicon, *absence* of human error | 4 |
| **reasoning** | reasoning-step **state gain**, restatement ratio, loop detection | 4 |

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

The toy language has no topic structure, so this doesn't settle whether *real* human text
is bursty because of topic shifts. That's a phase 6 measurement on real data, not
something assumed here.

## Status: phase 3 of 6

| Phase | Scope | State |
|---|---|---|
| 1 | Foundation: tokenization, streaming corpus I/O, metric contract, CLI, sample corpus, CI | **done** |
| 2 | Distributional family — 3 extractors, up to 9 signals per document | **done** |
| 3 | Surprisal family — two-pass pipeline, leave-one-out trigram LM, coverage-gated signals | **done** |
| 4 | Artifact + reasoning families | next |
| 5 | Robust score fusion, tiers, MD/JSON/HTML reports | planned |
| 6 | Calibration on a labelled corpus, published precision/recall, graphify graph | planned |

chaff computes and emits distributional and surprisal signals per document. It does
**not** yet fuse them into a contamination score. That's phase 5, and `chaff families`
will tell you so rather than pretending otherwise.

Signals are implemented and unit-tested against controlled corpora with known
properties: Zipfian distributions at known exponents, artificially truncated tails,
nucleus-truncated sampling from a known language. They are **not yet calibrated against
real labelled data**. The 10-document sample corpus averages 150 words, which is below
the length at which most of these metrics carry information, and at 1,475 tokens it's
far too small to train a language model. Building a long-document evaluation corpus is
phase 6, and no accuracy claim will be made before then.

Throughput, single-threaded pure Python: ~620k tokens/s building the model, ~170k
tokens/s for the scoring pass with every family active.

## Limitations, stated up front

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
| `tests/` | 108 tests |
| [`data/samples/`](data/samples/) | 10 labelled documents, including deliberate hard cases |

## Licence

MIT — see [LICENSE](LICENSE).
