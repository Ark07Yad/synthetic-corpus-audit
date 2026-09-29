"""Command line interface.

    chaff score <corpus>       fuse signals into calibrated tiers, scores and evidence
    chaff explain <scores> ID  why one document got its tier
    chaff report <scores>      render a scoring run as Markdown, JSON or HTML
    chaff profile <corpus>     corpus shape + raw signals, no scoring
    chaff inspect <corpus>     look at individual documents as chaff sees them
    chaff families             what is registered, and which phase ships what

"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from . import __version__
from .context import SOURCE_REFERENCE
from .corpus_io import CorpusError, iter_documents, write_jsonl
from .lm import StreamMismatchError
from .metrics import FAMILIES, bind_contextual, contextual_registered, registered
from .metrics.surprisal import MIN_COVERAGE_SENTENCE_SPREAD, MIN_COVERAGE_TOKEN_SPREAD
from .pipeline import build_context, profile_corpus, profile_document, score_corpus, short_document_note
from .report import RENDERERS, load_run
from .tokenization import build_view

#: Roadmap shown by ``chaff families``. Kept beside the registry so the CLI can
#: report honestly on what is implemented versus planned.
PHASE_PLAN = {
    "distributional": ("phase 2", "entropy, Zipf slope, frequency spectrum, Heaps' law, MTLD, n-gram repetition"),
    "surprisal": ("phase 3", "corpus-internal LM: mean surprisal, coverage-gated spread, recycled spans"),
    "artifact": ("phase 4", "assistant echoes, hedging, LLM lexicon, transitions, bold labels, human noise"),
    "reasoning": ("phase 4", "windowed step novelty, stalled-step ratio"),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chaff",
        description=(
            "Audit a pre-training corpus for synthetic contamination before "
            "tokenizer training."
        ),
    )
    parser.add_argument("--version", action="version", version="chaff {0}".format(__version__))
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("corpus", help="path to a .jsonl[.gz] file, a directory, or - for stdin")
    common.add_argument("--text-field", default=None,
                        help="record field holding the document text (default: autodetect)")
    common.add_argument("--limit", type=int, default=None, help="stop after N documents")
    common.add_argument("--min-chars", type=int, default=0,
                        help="skip documents shorter than N characters")
    common.add_argument("--reference", default=None, metavar="CORPUS",
                        help="build the language model from this trusted corpus instead of "
                             "the audited one (also skips the audited corpus's extra pass)")

    p_profile = sub.add_parser("profile", parents=[common],
                               help="profile corpus shape and registered signals")
    p_profile.add_argument("--out", default=None,
                           help="write per-document rows to this JSONL path")
    p_profile.add_argument("--json", action="store_true",
                           help="print the corpus summary as JSON instead of a table")
    p_profile.add_argument("--family", action="append", choices=FAMILIES, default=None,
                           help="restrict to a metric family (repeatable)")

    p_inspect = sub.add_parser("inspect", parents=[common],
                               help="show how chaff tokenizes individual documents")
    p_inspect.add_argument("-n", "--num", type=int, default=3,
                           help="how many documents to show (default: 3)")

    sub.add_parser("families", help="list metric families and their delivery phase")

    p_score = sub.add_parser("score", parents=[common],
                             help="fuse signals into calibrated tiers, scores and evidence")
    p_score.add_argument("--out", default=None, metavar="SCORES",
                         help="write one JSON line per document here (plus SCORES.meta.json)")
    p_score.add_argument("--report", default=None, metavar="PATH",
                         help="also write a report; format from the extension (.md, .html, .json)")
    p_score.add_argument("--top", type=int, default=10, help="documents to list (default: 10)")
    p_score.add_argument("--calibration", default=None, metavar="PATH",
                         help="use this calibration file instead of the shipped one")

    p_explain = sub.add_parser("explain", help="why one document got its tier")
    p_explain.add_argument("scores", help="scores file written by 'chaff score --out'")
    p_explain.add_argument("doc_id")

    p_report = sub.add_parser("report", help="render a scoring run as Markdown, JSON or HTML")
    p_report.add_argument("scores", help="scores file written by 'chaff score --out'")
    p_report.add_argument("--format", choices=sorted(RENDERERS), default="md")
    p_report.add_argument("--output", "-o", default=None, help="write here instead of stdout")
    return parser


def _fmt_int(value: float) -> str:
    return "{0:,}".format(int(value))


def cmd_profile(args: argparse.Namespace) -> int:
    rows: List[dict] = []
    collector = rows.append if args.out else None

    profile = profile_corpus(
        args.corpus,
        text_field=args.text_field,
        limit=args.limit,
        min_chars=args.min_chars,
        families=args.family,
        on_row=(lambda row: collector(row.to_row())) if collector else None,
        reference=args.reference,
    )

    if profile.n_documents == 0:
        sys.stderr.write("no documents found in {0}\n".format(args.corpus))
        return 1

    if args.out:
        written = write_jsonl(args.out, rows)
        sys.stderr.write("wrote {0} rows to {1}\n".format(_fmt_int(written), args.out))

    if args.json:
        print(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
        return 0

    summary = profile.length_summary
    print("corpus          {0}".format(profile.path))
    print("documents       {0}  ({1} analysable)".format(
        _fmt_int(profile.n_documents), _fmt_int(profile.n_analysable)))
    print("words           {0}".format(_fmt_int(profile.n_words)))
    print("vocabulary      {0} types".format(_fmt_int(profile.vocabulary_size)))
    print("hapax ratio     {0:.3f}  (share of vocabulary seen exactly once)".format(
        profile.hapax_ratio))
    print("corpus TTR      {0:.5f}".format(profile.corpus_ttr))
    print("doc length      median {0}  p25 {1}  p75 {2}  p95 {3}  max {4}  (words)".format(
        _fmt_int(summary["median"]), _fmt_int(summary["p25"]), _fmt_int(summary["p75"]),
        _fmt_int(summary["p95"]), _fmt_int(summary["max"])))
    print("elapsed         {0:.2f}s".format(profile.elapsed_seconds))

    if profile.lm_summary:
        _print_lm(profile.lm_summary, profile.context_source)

    if profile.metrics_active:
        print("\nsignals active  {0}".format(", ".join(profile.metrics_active)))
    else:
        print("\nsignals active  none — run 'chaff families'")

    for note in profile.notes:
        print("\nnote: {0}".format(note))

    note = short_document_note(profile)
    if note:
        print("\nnote: {0}".format(note))
    return 0


def _print_lm(summary: dict, source: Optional[str]) -> None:
    """Report language-model adequacy, since it decides which surprisal signals exist."""
    coverage = summary.get("bigram_coverage", 0.0)
    if source == SOURCE_REFERENCE:
        how = "reference corpus"
    else:
        how = "this corpus, leave-one-document-out"
    print("language model  {0}-gram from {1}; {2} tokens, {3} types".format(
        summary["order"], how, _fmt_int(summary["tokens"]), _fmt_int(summary["vocabulary"])))
    print("                {0} entries after lossless pruning ({1:.0%} removed)".format(
        _fmt_int(summary["entries"]), summary["pruned_share"]))
    if coverage >= MIN_COVERAGE_SENTENCE_SPREAD:
        verdict = "all surprisal signals active"
    elif coverage >= MIN_COVERAGE_TOKEN_SPREAD:
        verdict = "sentence-level spread withheld (needs {0:.2f})".format(MIN_COVERAGE_SENTENCE_SPREAD)
    else:
        verdict = "spread signals withheld: model too sparse (needs {0:.2f}); mean surprisal only".format(
            MIN_COVERAGE_TOKEN_SPREAD)
    print("bigram coverage {0:.3f}  -> {1}".format(coverage, verdict))


def cmd_inspect(args: argparse.Namespace) -> int:
    # Contextual signals only exist relative to the whole corpus, so inspecting even
    # one document needs the pass-1 model. Without it, inspect would silently show a
    # different signal set from profile for the same document.
    bound = []
    if contextual_registered():
        if args.reference:
            context = build_context(args.reference, text_field=args.text_field, source=SOURCE_REFERENCE)
        elif args.corpus != "-":
            context = build_context(args.corpus, text_field=args.text_field,
                                    limit=args.limit, min_chars=args.min_chars)
        else:
            context = None
            sys.stderr.write("note: surprisal family skipped for stdin (see 'chaff profile')\n")
        if context is not None:
            bound = bind_contextual(context)

    shown = 0
    for doc in iter_documents(
        args.corpus, text_field=args.text_field,
        limit=args.limit, min_chars=args.min_chars,
    ):
        view = build_view(doc.text)
        row = profile_document(doc, view=view, bound=bound)
        print("=" * 72)
        print("doc_id      {0}".format(doc.doc_id))
        if doc.label:
            print("label       {0}   (ground truth; never read by scoring)".format(doc.label))
        print("chars {0}  words {1}  types {2}  sentences {3}  lines {4}".format(
            row.n_chars, row.n_words, row.n_types, row.n_sentences, row.n_lines))
        print("analysable  {0}".format("yes" if row.analysable else "no (too short)"))
        print("preview     {0}".format(doc.preview))
        if view.words:
            print("first words {0}".format(" ".join(view.words[:14])))
        for signal in row.signals:
            print("  {0:<28} {1:>10.4f}  [{2}]".format(
                signal.name, signal.value, signal.family))
        shown += 1
        if shown >= args.num:
            break

    if shown == 0:
        sys.stderr.write("no documents found in {0}\n".format(args.corpus))
        return 1
    return 0


def cmd_families(_args: argparse.Namespace) -> int:
    active = dict()
    contextual = {name for name, _ in contextual_registered()}
    for name, family in registered():
        label = name + ("   (needs a corpus pass)" if name in contextual else "")
        active.setdefault(family, []).append(label)

    print("metric families\n")
    for family in FAMILIES:
        phase, blurb = PHASE_PLAN[family]
        names = active.get(family, [])
        # Extractors, not signals: one extractor emits several related signals when
        # they share a computation (lexical_profile emits six from one word count).
        status = ("{0} extractor{1}".format(len(names), "" if len(names) == 1 else "s")
                  if names else "not yet implemented")
        print("  {0:<16} {1:<9} {2:<20} {3}".format(family, phase, status, blurb))
        for name in sorted(names):
            print("      - {0}".format(name))

    print("\n'chaff score' tiers a document LIKELY_SYNTHETIC only when two or more families")
    print("exceed thresholds calibrated on guaranteed-human text (see 'chaff score --help').")
    return 0


def _format_from_path(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {"markdown": "md", "htm": "html"}.get(ext, ext) if ext in ("md", "markdown", "html", "htm", "json") else "md"


def cmd_score(args: argparse.Namespace) -> int:
    from .scoring import load_calibration
    run = score_corpus(
        args.corpus, text_field=args.text_field, limit=args.limit, min_chars=args.min_chars,
        reference=args.reference, calibration=load_calibration(args.calibration),
        scores_path=args.out, top_n=max(args.top, 25),
    )
    meta = run.meta
    if meta["documents"] == 0:
        sys.stderr.write("no documents found in {0}\n".format(args.corpus))
        return 1

    total = meta["scored"]
    tiers = meta["tiers"]
    print("corpus          {0}".format(meta["corpus"]))
    print("documents       {0}  ({1} scored)".format(_fmt_int(meta["documents"]), _fmt_int(total)))
    print("normalised vs   {0}".format(meta["normalisation"]))
    print("tiers           " + "  ".join("{0} {1}".format(t, tiers.get(t, 0)) for t in
                                         ("LIKELY_SYNTHETIC", "SUSPECT", "CLEAN", "UNSCORED")))
    fired = meta["families_fired"]
    print("families fired  " + ("  ".join("{0} {1}".format(f, n) for f, n in sorted(fired.items())) or "none"))
    held = ((meta.get("calibration") or {}).get("evaluation") or {}).get("B") or {}
    if held:
        print("calibration     on held-out human text: {0:.2%} LIKELY_SYNTHETIC, {1:.2%} SUSPECT".format(
            held["likely_rate"], held["suspect_rate"]))

    shown = [d for d in run.top if d.tier in ("LIKELY_SYNTHETIC", "SUSPECT")][: args.top]
    if shown:
        width = max(len(d.doc_id) for d in shown)
        print("\n  score  tier              {0:<{1}}  families fired".format("doc", width))
        for d in shown:
            print("  {0:5.1f}  {1:<17} {2:<{3}}  {4}".format(d.score, d.tier, d.doc_id, width, "+".join(d.fired)))
    for note in meta["notes"]:
        print("\nnote: {0}".format(note))

    sys.stdout.flush()
    if args.out:
        sys.stderr.write("\nwrote {0} and {0}.meta.json — try: chaff explain {0} <doc_id>\n".format(args.out))
    if args.report:
        fmt = _format_from_path(args.report)
        top = [d.to_dict() for d in run.top]
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(RENDERERS[fmt](meta, top))
        sys.stderr.write("wrote {0} report to {1}\n".format(fmt, args.report))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    found = None
    with open(args.scores, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("doc_id") == args.doc_id:
                found = record
                break
    if found is None:
        sys.stderr.write("no document {0!r} in {1}\n".format(args.doc_id, args.scores))
        return 1

    print("doc_id    {0}".format(found["doc_id"]))
    print("tier      {0}".format(found["tier"]))
    if found["score"] is None:
        print("          too short to analyse ({0} words); nothing was scored".format(found["n_words"]))
        return 0
    print("score     {0:.1f} / 100   (mean human-null percentile of its two strongest families)".format(found["score"]))
    if "label" in found:
        print("label     {0}   (ground truth; never read by scoring)".format(found["label"]))
    print("\nfamily           z        threshold  fired  human percentile  signals")
    for fam, r in sorted(found["families"].items(), key=lambda kv: -(kv[1]["z"] or 0)):
        print("  {0:<14} {1:+7.2f}   {2:>8}   {3:<5}  {4:>15}   {5}".format(
            fam, r["z"], "-" if r["threshold"] is None else "{0:+.2f}".format(r["threshold"]),
            "yes" if r["fired"] else "no",
            "-" if r["percentile"] is None else "{0:.1%}".format(r["percentile"]), r["n_signals"]))
    print("\nstrongest evidence (z > 0 points toward synthetic)")
    for e in found["evidence"]:
        print("  {0:+6.2f}  {1:<26} = {2:<10.4g} [{3}, vs {4}]".format(
            e["z"], e["signal"], e["value"], e["family"], e["basis"]))
        print("          {0}".format(e["description"]))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    meta, top = load_run(args.scores)
    text = RENDERERS[args.format](meta, top)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
        sys.stderr.write("wrote {0} report to {1}\n".format(args.format, args.output))
    else:
        print(text)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    handlers = {"profile": cmd_profile, "inspect": cmd_inspect, "families": cmd_families,
                "score": cmd_score, "explain": cmd_explain, "report": cmd_report}
    try:
        return handlers[args.command](args)
    except (CorpusError, StreamMismatchError) as exc:
        sys.stderr.write("error: {0}\n".format(exc))
        return 2
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
