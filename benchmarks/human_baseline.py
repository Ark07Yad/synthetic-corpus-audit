"""Build a guaranteed-human reference corpus from this machine, and measure chaff on it.

Why this exists
---------------
Artifact and reasoning detectors cannot be validated the way the distributional and
surprisal families were, on controlled corpora: if you write text containing "delve"
and then detect "delve", you have only proven the regex works. Their real risk is
false positives — CORE_PROBLEM §3.2: every artifact also appears in genuine human
writing. So what they need is text that is *known* to be human, and a measurement of
how often each detector fires on it.

Sources
-------
``stdlib``  Docstrings of the running interpreter's standard library, one document per
            module. **Only accepted from Python <= 3.10**: 3.10 was released October 2021
            and 3.9.6 in June 2021, both before ChatGPT (November 2022), so every docstring
            in them is human-written *by date* rather than by assumption. Formal, templated
            technical prose — the genre CORE_PROBLEM predicts will cause false positives.
``man``     Rendered man pages (``mandoc``), deduplicated by content. Mostly decades-old
            BSD reference text. Overwhelmingly pre-LLM but not date-guaranteed; reported
            separately so the two sources can be told apart.

Nothing produced here is committed: the corpus is regenerated from the local system on
demand (``benchmarks/out/`` is gitignored). Only aggregate numbers go into the docs.

Usage
-----
    python3 benchmarks/human_baseline.py build          # writes benchmarks/out/human_baseline.jsonl
    python3 benchmarks/human_baseline.py report         # firing rates + cross-family correlation
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import sysconfig
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

OUT_DIR = os.path.join(HERE, "out")
CORPUS = os.path.join(OUT_DIR, "human_baseline.jsonl")

#: Newest interpreter whose stdlib is date-guaranteed pre-LLM (3.10.0: Oct 2021).
MAX_PRE_LLM_PYTHON = (3, 10)
MIN_WORDS = 80
MAX_MAN_PAGES = 1500

_SKIP_DIRS = {"test", "tests", "idle_test", "site-packages", "__pycache__", "ensurepip"}


# ------------------------------------------------------------------- stdlib

def _module_docstrings(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except (SyntaxError, ValueError):
        return None
    parts: List[str] = []
    for node in [tree] + [n for n in ast.walk(tree)
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]:
        doc = ast.get_docstring(node, clean=True)
        if doc:
            parts.append(doc)
    return "\n\n".join(parts) if parts else None


def iter_stdlib() -> Iterator[Dict[str, str]]:
    if sys.version_info[:2] > MAX_PRE_LLM_PYTHON:
        raise SystemExit(
            "stdlib source needs Python <= {0}.{1} to guarantee pre-LLM authorship; "
            "this is {2}.{3}. Run with an older interpreter or use --sources man.".format(
                MAX_PRE_LLM_PYTHON[0], MAX_PRE_LLM_PYTHON[1], *sys.version_info[:2]))
    root = sysconfig.get_paths()["stdlib"]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            text = _module_docstrings(path)
            if text and len(text.split()) >= MIN_WORDS:
                rel = os.path.relpath(path, root)
                yield {"id": "stdlib:" + rel, "text": text, "source": "stdlib"}


# ---------------------------------------------------------------------- man

def iter_man(limit: int = MAX_MAN_PAGES) -> Iterator[Dict[str, str]]:
    root = "/usr/share/man"
    if not os.path.isdir(root):
        return
    seen = set()
    emitted = 0
    for section in ("man1", "man8", "man5", "man7"):
        folder = os.path.join(root, section)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if emitted >= limit:
                return
            path = os.path.realpath(os.path.join(folder, name))
            if path in seen or not os.path.isfile(path):
                continue
            seen.add(path)
            try:
                rendered = subprocess.run(["mandoc", "-T", "utf8", path], capture_output=True,
                                          timeout=10, check=False).stdout
                text = subprocess.run(["col", "-b"], input=rendered, capture_output=True,
                                      timeout=10, check=False).stdout.decode("utf-8", "replace")
            except (OSError, subprocess.TimeoutExpired):
                continue
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if digest in seen or len(text.split()) < MIN_WORDS:
                continue
            seen.add(digest)
            emitted += 1
            yield {"id": "man:{0}/{1}".format(section, name), "text": text, "source": "man"}


# ------------------------------------------------------------------- report

def half_of(doc_id: str) -> str:
    """Deterministic A/B split, so curation and evaluation never share documents."""
    return "A" if int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "B"


def _ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else None


def profile_rows(path: str) -> Tuple[List[Dict], Dict]:
    from chaff.pipeline import profile_corpus
    rows: List[Dict] = []
    profile = profile_corpus(path, on_row=lambda r: rows.append(
        {"doc_id": r.doc_id, "signals": {s.name: (s.value, s.family, s.direction) for s in r.signals}}))
    return rows, profile.lm_summary


def cmd_build(args: argparse.Namespace) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    counts: Dict[str, int] = defaultdict(int)
    with open(CORPUS, "w", encoding="utf-8") as fh:
        for source in args.sources:
            stream = iter_stdlib() if source == "stdlib" else iter_man()
            for record in stream:
                record["label"] = "human"
                record["half"] = half_of(record["id"])
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[source] += 1
    print("wrote {0} ({1})".format(CORPUS, ", ".join("{0}: {1}".format(k, v) for k, v in counts.items())))
    print("python {0}.{1}.{2}".format(*sys.version_info[:3]))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows, lm = profile_rows(CORPUS)
    sources, halves = {}, {}
    with open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            sources[rec["id"]] = rec["source"]
            halves[rec["id"]] = rec["half"]
    # The whole corpus is profiled — the surprisal family needs every document in its
    # language model — and rows are filtered afterwards. Half A shaped the detector
    # lists, so only half B gives an unbiased false-positive rate.
    if args.half != "all":
        rows = [r for r in rows if halves.get(r["doc_id"]) == args.half]

    by_signal: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    families: Dict[str, str] = {}
    for row in rows:
        src = sources.get(row["doc_id"], "?")
        for name, (value, family, _direction) in row["signals"].items():
            by_signal[name][src].append(value)
            by_signal[name]["all"].append(value)
            families[name] = family

    print("human baseline, half {0}: {1} documents  (LM coverage {2:.3f})\n".format(
        args.half, len(rows), lm.get("bigram_coverage", 0.0)))
    print("{0:<28} {1:<15} {2:>6} {3:>8} {4:>8} {5:>9} {6:>9}".format(
        "signal", "family", "docs", "fires%", "median", "p95", "max"))
    for name in sorted(by_signal, key=lambda n: (families[n], n)):
        if args.family and families[name] != args.family:
            continue
        for src in ("all", "stdlib", "man"):
            vals = sorted(by_signal[name].get(src, []))
            if not vals:
                continue
            fires = 100.0 * sum(1 for v in vals if v > 0) / len(vals)
            p95 = vals[min(len(vals) - 1, int(0.95 * len(vals)))]
            label = name if src == "all" else "  " + src
            print("{0:<28} {1:<15} {2:>6} {3:>7.1f}% {4:>8.3f} {5:>9.3f} {6:>9.3f}".format(
                label, families[name] if src == "all" else "", len(vals), fires,
                vals[len(vals) // 2], p95, vals[-1]))

    # Independence check for the corroboration rule (OBJECTIVE §4.3 / D5).
    print("\nstrongest cross-family rank correlations (|rho| >= 0.5 undermines 'independent'):")
    names = sorted(by_signal)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if families[a] == families[b]:
                continue
            shared = [(r["signals"][a][0], r["signals"][b][0]) for r in rows
                      if a in r["signals"] and b in r["signals"]]
            if len(shared) < 30:
                continue
            rho = spearman([x for x, _ in shared], [y for _, y in shared])
            if rho is not None:
                pairs.append((abs(rho), rho, a, b, len(shared)))
    for _, rho, a, b, n in sorted(pairs, reverse=True)[: args.top]:
        print("  {0:+.2f}  {1} [{2}]  x  {3} [{4}]   (n={5})".format(rho, a, families[a], b, families[b], n))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", nargs="+", default=["stdlib", "man"], choices=["stdlib", "man"])
    r = sub.add_parser("report")
    r.add_argument("--top", type=int, default=12)
    r.add_argument("--half", choices=["A", "B", "all"], default="B",
                   help="A shaped the detector lists; B (default) is the held-out evaluation")
    r.add_argument("--family", default=None, help="only print signals of this family")
    args = parser.parse_args(argv)
    return {"build": cmd_build, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
