# Benchmarks

## `human_baseline.py` — false positives on guaranteed-human text

The artifact and reasoning families can't be validated the way the distributional and
surprisal families were, on controlled corpora. If you write text containing "delve" and
then detect "delve", you've only proven the regex works. Their real risk is **false
positives**: every artifact they look for also appears in genuine human writing. So they
need text that is *known* to be human, and a measurement of how often each detector fires
on it.

```bash
python3 benchmarks/human_baseline.py build             # ~12 s: builds benchmarks/out/human_baseline.jsonl
python3 benchmarks/human_baseline.py report            # held-out half B (default)
python3 benchmarks/human_baseline.py report --half A   # the half that shaped the detector lists
python3 benchmarks/human_baseline.py report --family artifact
python3 benchmarks/human_baseline.py calibrate         # fusion thresholds -> src/chaff/calibration.json
```

### Sources

| Source | Documents | Why it's trusted |
|---|---|---|
| Python standard-library docstrings, one document per module | 309 | **Human by date.** Accepted only from Python ≤ 3.10 (3.9.6 shipped June 2021), before ChatGPT (Nov 2022). The script refuses newer interpreters. |
| Man pages, rendered with `mandoc`, deduplicated | 1,439 | Mostly decades-old BSD reference text. Overwhelmingly pre-LLM, but *not* date-guaranteed, so it's reported separately. |

Nothing from these sources is committed. The corpus is regenerated from the local system
on demand, and `benchmarks/out/` is gitignored.

### Held-out discipline

Documents are split deterministically into halves A and B by a hash of their id. Detector
lists (the LLM lexicon, the transition words, the human-noise markers) were pruned
using **half A only**. False-positive rates are reported on **half B**, which played no
part in shaping them. `--half A` exists for comparison; the gap between A and B is the
curation optimism the split is there to expose.

### Results (phase 4, half B, 871 documents)

Share of human documents on which each artifact detector fires at all, and the highest
rate any human document reached:

| Detector | Fires | Human max | Before pruning (all docs) |
|---|---|---|---|
| `assistant_echo_rate` | **0.0%** | 0 | 0.0% |
| `bold_lead_rate` | **0.0%** | 0 | 0.2% |
| `llm_lexicon_rate` | **0.7%** | 0.90 / 1k words | 11.0%, max 39 / 1k |
| `hedging_rate` | 2.6% | 0.78 / 1k words | 3.0% |
| `human_noise_rate` | 2.9% | 3.9 / 1k words | 97% |
| `transition_rate` | 9.8% | 8.7% of sentences | 16.5% |

What the pruning removed, and why it mattered:

- **The lexicon collided with domain vocabulary.** "underscore" (the `_` character),
  "realm" (Kerberos) and "harness" (test harness) fired at LLM-like *rates*, 39 per 1,000
  words on one Kerberos page, not occasionally. Word families appearing in two or more
  half-A documents were removed.
- **"Human noise" was measuring typesetting.** Man pages and PEP 8 docstrings put two
  spaces after a full stop; code identifiers open sentences in lowercase. The detector now
  counts only haste markers that survive HTML extraction.

### Independence of the metric families

The corroboration rule (OBJECTIVE §4.3) counts two firing families as two pieces of
evidence, which only holds if the families are actually independent. This benchmark
measures that as the strongest cross-family Spearman correlation on human text.

A reasoning signal, `restatement`, correlated with the distributional family at |ρ| up to
**0.74** and was removed: it was re-measuring lexical repetition. The strongest remaining
cross-family correlation is **0.49** (`stalled_step_ratio` × `repetition_4gram`). That's
weaker coupling, but it isn't independence, and fusion has to account for it.

### Calibrating fusion (phase 5, recalibrated for two-sided scoring in phase 6)

`calibrate` learns everything `chaff score` needs from this baseline and writes it to
`src/chaff/calibration.json` (numbers only, no text, ~17 KB). It learns:

- the human distribution of each artifact signal (from half A),
- each family's null distribution and threshold,
- the null distribution of the 0–100 score.

The thresholds come from one per-family α, the largest for which at most **1%** of half-A
documents reach `LIKELY_SYNTHETIC`. That's α = 0.02. Held-out half B then gives:

| | Half A (fit) | Half B (held out) |
|---|---|---|
| `LIKELY_SYNTHETIC` | 0.91% | **0.92%** |
| `SUSPECT` | 5.36% | 6.66% |
| `LIKELY` if the families were independent | 0.19% | 0.26% |
| dependence inflation | 4.8× | **3.6×** |

Because α was chosen on the *measured* joint rate, the calibration already includes the
families' real dependence. The inflation row shows how much an independence assumption
would have understated the false-positive rate.

The scoring path (`chaff score`) and the calibration path compute fusion through
different code. Scoring the whole baseline with `chaff score` reproduces both halves'
rates exactly, to the hundredth of a percent.

The minority genre pays for corpus-relative normalisation. On half B, the distributional
family fires on **4.3% of stdlib documents** against 1.4% of man pages, because man pages
are 81% of the baseline. That's the minority-genre penalty the main README warns about;
`--stratify-by` removes most of it.

Since phase 7 the procedure lives in the package (`src/chaff/calibrate.py`), and
`chaff calibrate <trusted-corpus> --out mine.json` runs it on your own corpus. This script
is a wrapper around the same code, and it still reproduces the shipped
`calibration.json` exactly, every field but the date.

### What this benchmark cannot tell you

- **True-positive rates.** It contains no synthetic text. Whether these detectors *catch*
  LLM output at a useful rate needs an independent corpus of real model output, which is phase 6.
- **Non-technical genres.** Both sources are technical reference text. False positives in
  literary, journalistic or casual human writing ("testament", "bustling") are unmeasured.
- **Markdown.** Neither source is markdown, so markdown false positives rest on the
  sample corpus's changelog hard case until phase 6.
