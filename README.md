# chaff

**Synthetic corpus contamination & token entropy auditing for LLM pre-training data.**

Find AI-generated text in a pre-training corpus *before* it poisons your tokenizer.

[![ci](https://github.com/Ark07Yad/synthetic-corpus-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/Ark07Yad/synthetic-corpus-audit/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![tests](https://img.shields.io/badge/tests-66-brightgreen)
![dependencies](https://img.shields.io/badge/runtime%20dependencies-0-brightgreen)
![status](https://img.shields.io/badge/status-phase%202%20of%206-orange)

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
chaff families                             # what is implemented, and what is coming
```

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
| **surprisal** | mean surprisal, **variance and burstiness**, low-surprisal run length | 3 |
| **artifact** | system-prompt echoes, hedging scaffolds, markdown watermarks, overused lexicon, *absence* of human error | 4 |
| **reasoning** | reasoning-step **state gain**, restatement ratio, loop detection | 4 |

The most load-bearing idea is in the surprisal family: human text is **bursty** — a
predictable stretch, then a surprising word, then another predictable stretch. Decoded
text is flat. Measuring the *variance* of surprisal separates them far better than
measuring its mean, which is the mistake that makes naive perplexity filters delete
human writing and keep the synthetic text.

That family is built on a **corpus-internal n-gram language model** with leave-one-doc-out
counts, not a downloaded transformer. It costs one extra pass over the data and keeps
the whole tool dependency-free.

## Status: phase 2 of 6

| Phase | Scope | State |
|---|---|---|
| 1 | Foundation: tokenization, streaming corpus I/O, metric contract, CLI, sample corpus, CI | **done** |
| 2 | Distributional family — 3 extractors, up to 9 signals per document | **done** |
| 3 | Surprisal family + corpus-internal LM | next |
| 4 | Artifact + reasoning families | planned |
| 5 | Robust score fusion, tiers, MD/JSON/HTML reports | planned |
| 6 | Calibration on a labelled corpus, published precision/recall, graphify graph | planned |

Phase 2 computes and emits distributional signals per document. It does **not** yet
fuse them into a contamination score — that is phase 5, and `chaff families` will tell
you so rather than pretending otherwise.

Signals are implemented and unit-tested against controlled corpora with known
properties (Zipfian distributions at known exponents, artificially truncated tails).
They are **not yet calibrated against real labelled data** — the 10-document sample
corpus averages 150 words, which is below the length at which most of these metrics
carry information. Building a long-document evaluation corpus is phase 6, and no
accuracy claim will be made before then.

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
- **Signals withhold themselves rather than guess.** Several metrics need a minimum
  document length to mean anything — the frequency-spectrum slope needs ~2,000 words,
  measured rather than assumed — and return nothing below it. A short document will
  legitimately produce fewer signals than a long one.

## Repository map

| Path | What it is |
|---|---|
| `src/chaff/` | The package |
| `src/chaff/metrics/` | The `Signal` contract and the metric families |
| `tests/` | 66 tests |
| [`data/samples/`](data/samples/) | 10 labelled documents, including deliberate hard cases |

## Licence

MIT — see [LICENSE](LICENSE).
