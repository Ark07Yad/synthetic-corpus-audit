# Sample corpus

`mixed_sample.jsonl` — 10 labelled documents, 5 human / 5 synthetic, hand-written for
this repo. Small enough to eyeball, structured enough to smoke-test the pipeline.

| Field | Meaning |
|---|---|
| `id` | stable doc id |
| `label` | `human` / `synthetic` — ground truth, **never read by scoring** |
| `genre` | forum, essay, technical, changelog, legal, blog, explainer, reasoning, article, qa |
| `difficulty` | `easy` = textbook case, `hard` = deliberately adversarial |

## The hard cases are the point

Three documents exist specifically to stop the metrics from looking better than they are:

- **`h_changelog_01`** (human, hard) — a real-shaped release changelog. Markdown-regular,
  bulleted, repetitive, low lexical diversity. Every formatting-artifact heuristic in
  phase 4 should want to flag this. It must not be tiered `LIKELY_SYNTHETIC`.
- **`h_legal_01`** (human, hard) — contract boilerplate. Extreme phrase repetition and
  near-zero surprisal burstiness by construction. Same test.
- **`s_subtle_01`** (synthetic, hard) — synthetic text with the tells filed off: no
  hedging lexicon, no markdown, no triads, plain declarative register. Artifact metrics
  should find nothing here. Only the distributional and surprisal families have a
  chance, which is exactly why the tool does not rely on artifact detection alone.

A scorer that gets 10/10 on this file is probably overfitted to it. The real
calibration corpus arrives in phase 6; this one is for wiring, not for claims.

## Provenance

All ten documents were written by hand for this repository. No text was scraped,
and no third-party corpus is redistributed here.
