"""Phase 6 evaluation: how much synthetic text chaff actually catches, and at what cost.

Everything before phase 6 measured false positives. This measures true positives, on
three independent sources of model-written text, each set against human text of the
same kind wherever that exists:

``tech``      Blind Claude subagents wrote 40 man pages and 40 Python-module docstring
              sets (fictional tools and modules), set against the held-out human
              baseline (half B: 708 man pages, 163 stdlib modules). The writers never
              saw chaff, its detectors or the purpose of the task. Genre-matched.
``literary``  40 blind-Claude essays against Project Gutenberg books (pre-1928, human by
              date), chunked. NOT era-matched — see the caveat in the output.
``mage``      The MAGE benchmark (Li et al., ACL 2024): human text from 10 domains
              against 25 generators from 7 model families (OpenAI, LLaMA, OPT,
              EleutherAI, GLM, BLOOM, FLAN-T5). Documents of at least 250 words only.
``mage-gpt4`` MAGE's out-of-distribution set: GPT-4 on four unseen domains.
``mage-confirm`` Fresh MAGE documents that no analysis had touched, built *after* the
              two-sided hypothesis was formed, as its pre-registered confirmation set.

Each is scored corpus-relative (the default) and against a **trusted human reference**
(``--reference``) drawn from a disjoint half of the same source's human text, which is
the recommended mode when such a corpus exists.

Nothing produced here is committed; ``benchmarks/EVALUATION.md`` reports the numbers.

    python3 benchmarks/evaluate.py build     # assembles benchmarks/out/eval/*.jsonl
    python3 benchmarks/evaluate.py run       # scores every experiment -> results.json + results.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
OUT = os.path.join(HERE, "out")
EVAL = os.path.join(OUT, "eval")

MAGE_MIN_WORDS = 250
MAGE_PER_DOMAIN = {"eval_machine": 150, "eval_human": 150, "reference": 300}
GUTENBERG_CHUNK_WORDS = 1200
SWEEP_SHARES = (0.02, 0.08, 0.20, 0.50)


def _half(key: str) -> str:
    return "A" if int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "B"


def _write(name: str, records: List[Dict]) -> None:
    with open(os.path.join(EVAL, name), "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("  {0:<28} {1:>6} docs  {2}".format(name, len(records),
          dict(Counter(r["label"] for r in records))))


def mage_family(model: str) -> str:
    m = model.lower()
    if re.fullmatch(r"\d+b", m):
        return "LLaMA"
    for key, family in (("gpt4", "OpenAI"), ("gpt-3.5", "OpenAI"), ("davinci", "OpenAI"),
                        ("glm", "GLM"), ("bloom", "BLOOM"), ("flan_t5", "FLAN-T5"),
                        ("gpt_j", "EleutherAI"), ("gpt_neox", "EleutherAI"), ("opt", "OPT")):
        if key in m:
            return family
    return "other"


def _mage_model(src: str) -> str:
    m = src.split("_machine_", 1)[1] if "_machine_" in src else src.split("_", 1)[1]
    for prefix in ("continuation_", "specified_", "topical_"):
        if m.startswith(prefix):
            m = m[len(prefix):]
    return m


# ------------------------------------------------------------------- build

def _gutenberg_chunks() -> List[Dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(OUT, "gutenberg", "*.txt"))):
        book = os.path.basename(path)[:-4]
        text = open(path, encoding="utf-8", errors="replace").read()
        start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I | re.S)
        end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text, re.I)
        body = text[start.end() if start else 0: end.start() if end else len(text)]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        chunk, words, k = [], 0, 0
        for p in paragraphs:
            chunk.append(p)
            words += len(p.split())
            if words >= GUTENBERG_CHUNK_WORDS:
                k += 1
                out.append({"id": "gutenberg:{0}:{1:03d}".format(book, k), "text": "\n\n".join(chunk),
                            "label": "human", "genre": "literary", "generator": "human", "book": book})
                chunk, words = [], 0
    return out


def cmd_build(args: argparse.Namespace) -> int:
    os.makedirs(EVAL, exist_ok=True)
    print("building evaluation corpora in", os.path.relpath(EVAL))

    baseline = [json.loads(l) for l in open(os.path.join(OUT, "human_baseline.jsonl"), encoding="utf-8")]
    genre_of = {"man": "man", "stdlib": "apidoc"}
    tech_human = [{"id": b["id"], "text": b["text"], "label": "human", "genre": genre_of[b["source"]],
                   "generator": "human"} for b in baseline]
    synthetic = []
    for genre, folder in (("man", "man"), ("apidoc", "moduledoc"), ("essay", "essay")):
        for path in sorted(glob.glob(os.path.join(OUT, "synthetic", folder, "*.txt"))):
            synthetic.append({"id": "claude:{0}:{1}".format(folder, os.path.basename(path)[:-4]),
                              "text": open(path, encoding="utf-8").read(), "label": "synthetic",
                              "genre": genre, "generator": "claude-blind"})
    halves = {b["id"]: b["half"] for b in baseline}
    _write("tech_reference.jsonl", [r for r in tech_human if halves[r["id"]] == "A"])
    _write("tech.jsonl", [r for r in tech_human if halves[r["id"]] == "B"]
           + [s for s in synthetic if s["genre"] in ("man", "apidoc")])

    chunks = _gutenberg_chunks()
    _write("literary_reference.jsonl", [c for c in chunks if _half(c["id"]) == "A"])
    _write("literary.jsonl", [c for c in chunks if _half(c["id"]) == "B"]
           + [s for s in synthetic if s["genre"] == "essay"])

    csv.field_size_limit(sys.maxsize)
    for name, path in (("mage", "test.csv"), ("mage_gpt4", "test_ood_set_gpt.csv")):
        rng = random.Random(6)
        rows = [r for r in csv.DictReader(open(os.path.join(OUT, "mage", path), encoding="utf-8-sig"))
                if len(r["text"].split()) >= MAGE_MIN_WORDS]
        rng.shuffle(rows)
        take = defaultdict(int)
        ev, ref = [], []
        for i, r in enumerate(rows):
            domain = r["src"].split("_")[0]
            human = r["label"] == "1"
            rec = {"id": "mage:{0}:{1}".format(name, i), "text": r["text"],
                   "label": "human" if human else "synthetic", "domain": domain,
                   "generator": "human" if human else _mage_model(r["src"])}
            if not human:
                rec["family"] = mage_family(rec["generator"])
                bucket = "eval_machine"
            else:
                bucket = "reference" if _half(rec["id"]) == "A" else "eval_human"
            cap = MAGE_PER_DOMAIN[bucket] if name == "mage" else 10 ** 6
            if take[(bucket, domain)] >= cap:
                continue
            take[(bucket, domain)] += 1
            (ref if bucket == "reference" else ev).append(rec)
        _write(name + "_reference.jsonl", ref)
        _write(name + ".jsonl", ev)

    # Confirmation set: MAGE rows that appear in no other evaluation file (by text hash),
    # so the two-sided hypothesis — formed on the files above — is tested on unseen text.
    used = set()
    for f in ("mage.jsonl", "mage_reference.jsonl"):
        for line in open(os.path.join(EVAL, f), encoding="utf-8"):
            used.add(hashlib.sha1(json.loads(line)["text"].encode("utf-8")).hexdigest())
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, "mage", "test.csv"), encoding="utf-8-sig"))
            if len(r["text"].split()) >= MAGE_MIN_WORDS
            and hashlib.sha1(r["text"].encode("utf-8")).hexdigest() not in used]
    rng = random.Random(2026)
    rng.shuffle(rows)
    take, fresh = defaultdict(int), []
    for i, r in enumerate(rows):
        domain, human = r["src"].split("_")[0], r["label"] == "1"
        if take[(domain, human)] >= 100:
            continue
        take[(domain, human)] += 1
        rec = {"id": "mage-confirm:{0}".format(i), "text": r["text"], "domain": domain,
               "label": "human" if human else "synthetic",
               "generator": "human" if human else _mage_model(r["src"])}
        if not human:
            rec["family"] = mage_family(rec["generator"])
        fresh.append(rec)
    _write("mage_confirm.jsonl", fresh)
    return 0


# --------------------------------------------------------------------- run

def auc(pos: List[float], neg: List[float]) -> Optional[float]:
    """Mann-Whitney AUC with mid-ranks for ties: P(random positive > random negative)."""
    if not pos or not neg:
        return None
    scored = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    rank_sum = 0.0
    while i < len(scored):
        j = i
        while j + 1 < len(scored) and scored[j + 1][0] == scored[i][0]:
            j += 1
        mid = (i + j) / 2.0 + 1
        rank_sum += mid * sum(1 for k in range(i, j + 1) if scored[k][1] == 1)
        i = j + 1
    return (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def metrics(rows: List[Dict]) -> Dict:
    rows = [r for r in rows if r["tier"] != "UNSCORED"]
    syn = [r for r in rows if r.get("label") == "synthetic"]
    hum = [r for r in rows if r.get("label") == "human"]

    def rate(sub, pred):
        return sum(1 for r in sub if pred(r)) / float(len(sub)) if sub else None

    likely = lambda r: r["tier"] == "LIKELY_SYNTHETIC"
    flagged = lambda r: r["tier"] in ("LIKELY_SYNTHETIC", "SUSPECT")

    def precision(pred):
        tp = sum(1 for r in syn if pred(r))
        fp = sum(1 for r in hum if pred(r))
        return tp / float(tp + fp) if tp + fp else None

    fam_fires = Counter(f for r in syn for f in r["families_fired"])
    return {
        "n_synthetic": len(syn), "n_human": len(hum),
        "prevalence": len(syn) / float(len(rows)) if rows else None,
        "recall_likely": rate(syn, likely), "recall_flagged": rate(syn, flagged),
        "fpr_likely": rate(hum, likely), "fpr_flagged": rate(hum, flagged),
        "precision_likely": precision(likely), "precision_flagged": precision(flagged),
        "auc": auc([r["score"] for r in syn], [r["score"] for r in hum]),
        "synthetic_family_fire_rate": {f: c / float(len(syn)) for f, c in fam_fires.items()} if syn else {},
    }


def _score(corpus: str, reference: Optional[str], stratify: Optional[str]) -> List[Dict]:
    from chaff.pipeline import score_corpus
    out = os.path.join(EVAL, "scores.tmp.jsonl")
    score_corpus(corpus, reference=reference, scores_path=out, stratify_by=stratify)
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(corpus, encoding="utf-8")}
    for r in rows:
        src = meta[r["doc_id"]]
        for key in ("genre", "domain", "generator", "family", "book"):
            if key in src:
                r[key] = src[key]
    return rows


def _failures(rows: List[Dict], k: int = 5) -> Dict:
    scored = [r for r in rows if r["tier"] != "UNSCORED"]
    fp = sorted((r for r in scored if r.get("label") == "human"), key=lambda r: -r["score"])[:k]
    fn = sorted((r for r in scored if r.get("label") == "synthetic"), key=lambda r: r["score"])[:k]
    brief = lambda r: {"doc_id": r["doc_id"], "tier": r["tier"], "score": r["score"],
                       "fired": r["families_fired"], "top_evidence": [e["signal"] for e in r["evidence"][:3]]}
    return {"highest_scoring_humans": [brief(r) for r in fp], "lowest_scoring_synthetic": [brief(r) for r in fn]}


EXPERIMENTS = [
    # name, corpus, reference, stratify
    ("tech / corpus-relative", "tech.jsonl", None, None),
    ("tech / reference", "tech.jsonl", "tech_reference.jsonl", None),
    ("tech / reference + stratify genre", "tech.jsonl", "tech_reference.jsonl", "genre"),
    ("literary / corpus-relative", "literary.jsonl", None, None),
    ("literary / reference", "literary.jsonl", "literary_reference.jsonl", None),
    ("mage / corpus-relative", "mage.jsonl", None, None),
    ("mage / reference + stratify domain", "mage.jsonl", "mage_reference.jsonl", "domain"),
    ("mage-gpt4 / corpus-relative", "mage_gpt4.jsonl", None, None),
    ("mage-gpt4 / reference + stratify domain", "mage_gpt4.jsonl", "mage_gpt4_reference.jsonl", "domain"),
    ("mage-confirm / corpus-relative", "mage_confirm.jsonl", None, None),
    ("mage-confirm / reference + stratify domain", "mage_confirm.jsonl", "mage_reference.jsonl", "domain"),
]


def _sweep() -> List[Dict]:
    """Recall and false-positive rate as the synthetic share of a corpus rises."""
    base = [json.loads(l) for l in open(os.path.join(EVAL, "tech.jsonl"), encoding="utf-8")]
    syn = [r for r in base if r["label"] == "synthetic"]
    hum = [r for r in base if r["label"] == "human"]
    out = []
    for share in SWEEP_SHARES:
        rng = random.Random(11)
        n_h = len(hum)
        n_s = int(round(share * n_h / (1 - share)))
        if n_s > len(syn):                   # not enough synthetic docs: shrink the human side
            n_s = len(syn)
            n_h = int(round(n_s * (1 - share) / share))
        docs = rng.sample(syn, n_s) + rng.sample(hum, n_h)
        path = os.path.join(EVAL, "sweep.tmp.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for d in docs:
                fh.write(json.dumps(d) + "\n")
        for mode, ref in (("corpus-relative", None), ("reference", os.path.join(EVAL, "tech_reference.jsonl"))):
            m = metrics(_score(path, ref, None))
            out.append({"share": share, "mode": mode, "n": len(docs), **{k: m[k] for k in (
                "recall_flagged", "recall_likely", "fpr_flagged", "fpr_likely", "auc")}})
            print("  sweep share {0:>4.0%}  {1:<16} recall(flag) {2:5.1%}  FPR(flag) {3:5.1%}  AUC {4:.3f}".format(
                share, mode, m["recall_flagged"], m["fpr_flagged"], m["auc"]))
    return out


def cmd_run(args: argparse.Namespace) -> int:
    results = {"experiments": {}, "sweep": []}
    for name, corpus, reference, stratify in EXPERIMENTS:
        if args.only and args.only not in name:
            continue
        rows = _score(os.path.join(EVAL, corpus), os.path.join(EVAL, reference) if reference else None, stratify)
        m = metrics(rows)
        entry = {"overall": m, "failures": _failures(rows)}
        for key in ("genre", "domain", "family"):
            if any(key in r for r in rows):
                # Per-group recall against the whole human side, so each group's number
                # is the chance chaff catches that group's documents in this corpus.
                humans = [r for r in rows if r.get("label") == "human"]
                groups = defaultdict(list)
                for r in rows:
                    if r.get("label") == "synthetic" and key in r:
                        groups[r[key]].append(r)
                    elif r.get("label") == "human" and key in r and key != "family":
                        groups["human:" + str(r[key])].append(r)
                entry["by_" + key] = {g: metrics(sub + (humans if not g.startswith("human:") else []))
                                      for g, sub in sorted(groups.items())}
        results["experiments"][name] = entry
        print("{0:<42} syn {1:>4} hum {2:>4}  recall flag {3:5.1%} likely {4:5.1%}  FPR flag {5:5.1%} "
              "likely {6:5.1%}  AUC {7:.3f}".format(name, m["n_synthetic"], m["n_human"], m["recall_flagged"],
                                                   m["recall_likely"], m["fpr_flagged"], m["fpr_likely"], m["auc"]))
    if not args.only:
        results["sweep"] = _sweep()
    for tmp in ("scores.tmp.jsonl", "scores.tmp.jsonl.meta.json", "sweep.tmp.jsonl"):
        path = os.path.join(EVAL, tmp)
        if os.path.exists(path):
            os.remove(path)
    with open(os.path.join(EVAL, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", os.path.relpath(os.path.join(EVAL, "results.json")))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--only", default=None, help="run only experiments whose name contains this")
    args = parser.parse_args(argv)
    return {"build": cmd_build, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
