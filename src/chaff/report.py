"""Render a scoring run as Markdown, JSON or self-contained HTML.

Every renderer reads the same two things — the run's ``meta`` and its top-ranked
documents — which ``chaff score`` writes to ``scores.jsonl.meta.json``. Reports can
therefore be re-rendered in any format without re-reading the corpus.

Each report carries its own caveats. The calibration's provenance and held-out error
rates are printed alongside the results, never in a separate document a reader might
not open (OBJECTIVE §4.5: evidence is emitted, always).
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .scoring import TIERS

#: Evidence below this z is not shown in reports: a signal at z 0.03 is not evidence of
#: anything, and listing it next to real evidence dilutes the explanation.
#: ``chaff explain`` still shows every signal, for debugging.
NOTABLE_Z = 1.0

_TIER_BLURB = {
    "LIKELY_SYNTHETIC": "two or more independent-ish families exceed their calibrated threshold",
    "SUSPECT": "exactly one family exceeds its threshold",
    "CLEAN": "no family exceeds its threshold",
    "UNSCORED": "too short to analyse (under 50 words)",
}


def load_run(scores_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read the ``.meta.json`` sidecar written next to a scores file."""
    with open(scores_path + ".meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    return meta, meta.pop("top", [])


def _pct(n: int, total: int) -> str:
    return "{0:.1%}".format(n / float(total)) if total else "–"


def _calibration_line(meta: Mapping[str, Any]) -> str:
    cal = meta.get("calibration") or {}
    held = (cal.get("evaluation") or {}).get("B") or {}
    if not held:
        return "No calibration metadata available."
    return ("Thresholds calibrated on {a} guaranteed-human documents (alpha {alpha} per family). "
            "On {b} held-out human documents they tier {likely:.2%} LIKELY_SYNTHETIC and "
            "{suspect:.2%} SUSPECT — the expected false-positive rates on text like the baseline "
            "(technical reference prose).").format(
        a=cal.get("documents", {}).get("A", "?"), alpha=cal.get("alpha"),
        b=held.get("documents", "?"), likely=held.get("likely_rate", 0.0),
        suspect=held.get("suspect_rate", 0.0))


CAVEATS = (
    "chaff is a triage instrument, not an AI detector. A tier is a reason to look at a "
    "document, not proof of how it was written.",
    "Corpus-relative scoring measures how unusual a document is *within this corpus*. A "
    "genre that is rare in the corpus looks unusual: on the calibration baseline, the "
    "distributional family fired on 6.1% of the minority genre against 1.8% of the majority. "
    "Use --reference with a genre-matched trusted corpus when you have one.",
    "False-positive rates were measured on technical reference text only. Rates on "
    "literary, journalistic or casual writing are not yet measured.",
    "True-positive rates are not yet measured: that needs an independent corpus of real "
    "model output (phase 6).",
)


# ------------------------------------------------------------------ markdown

def render_markdown(meta: Mapping[str, Any], top: Sequence[Mapping[str, Any]]) -> str:
    total = meta.get("scored", 0)
    tiers = meta.get("tiers", {})
    out: List[str] = []
    out.append("# chaff report")
    out.append("")
    out.append("`{0}` · {1} documents · {2} scored · normalised against the {3}".format(
        meta.get("corpus"), meta.get("documents"), total, meta.get("normalisation")))
    out.append("")
    out.append("## Tiers")
    out.append("")
    out.append("| Tier | Documents | Share | Meaning |")
    out.append("|---|---|---|---|")
    for tier in TIERS:
        n = tiers.get(tier, 0)
        share = _pct(n, total) if tier != "UNSCORED" else "–"
        out.append("| {0} | {1} | {2} | {3} |".format(tier, n, share, _TIER_BLURB[tier]))
    out.append("")
    out.append(_calibration_line(meta))
    out.append("")
    out.append("## Families")
    out.append("")
    out.append("| Family | Fired on | Threshold (z) |")
    out.append("|---|---|---|")
    fired = meta.get("families_fired", {})
    for fam, thr in sorted((meta.get("family_thresholds") or {}).items()):
        out.append("| {0} | {1} ({2}) | {3:.2f} |".format(fam, fired.get(fam, 0), _pct(fired.get(fam, 0), total), thr))
    if meta.get("labels"):
        out.append("")
        out.append("## Tier by ground-truth label")
        out.append("")
        out.append("Labels are carried through for evaluation only; scoring never reads them.")
        out.append("")
        out.append("| Label | " + " | ".join(TIERS) + " |")
        out.append("|---|" + "---|" * len(TIERS))
        for label, counts in sorted(meta["labels"].items()):
            out.append("| {0} | ".format(label) + " | ".join(str(counts.get(t, 0)) for t in TIERS) + " |")
    out.append("")
    out.append("## Highest-scoring documents")
    out.append("")
    for doc in top:
        out.append("### {0} — {1} · score {2:.1f}".format(doc["doc_id"], doc["tier"], doc["score"] or 0))
        out.append("")
        fams = ", ".join("{0} z={1:+.2f}{2}".format(f, r["z"], " ✔" if r["fired"] else "")
                         for f, r in sorted(doc["families"].items(), key=lambda kv: -(kv[1]["z"] or 0)))
        out.append("Families: " + fams)
        out.append("")
        for e in [x for x in doc["evidence"] if x["z"] >= NOTABLE_Z][:5]:
            out.append("- **{0}** = {1:.4g} (z {2:+.2f}, vs {3}) — {4}".format(
                e["signal"], e["value"], e["z"], e["basis"], e["description"]))
        out.append("")
    if meta.get("notes"):
        out.append("## Notes")
        out.append("")
        out.extend("- " + n for n in meta["notes"])
        out.append("")
    out.append("## Read this before acting on the results")
    out.append("")
    out.extend("- " + c for c in CAVEATS)
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------- json

def render_json(meta: Mapping[str, Any], top: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(dict(meta, top=list(top), caveats=list(CAVEATS)), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------- html

_CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--card:#ffffff;
--likely:#b3261e;--suspect:#b86e00;--clean:#2f6f4f;--unscored:#8a8983;--bar:#3d5a80}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--muted:#a3a29c;
--line:#2c2b28;--card:#1c1c1a;--likely:#f2867d;--suspect:#f0b35a;--clean:#7cc6a0;
--unscored:#8a8983;--bar:#8fb3e0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px}
.sub{color:var(--muted);margin:0 0 24px;word-break:break-all}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .n{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}
.card .k{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.t-LIKELY_SYNTHETIC{color:var(--likely)}.t-SUSPECT{color:var(--suspect)}
.t-CLEAN{color:var(--clean)}.t-UNSCORED{color:var(--unscored)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);font-weight:600}
.doc{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.doc h3{margin:0 0 6px;font-size:15px;word-break:break-all}
.pill{display:inline-block;font-size:12px;font-weight:600;padding:1px 8px;border-radius:99px;
border:1px solid currentColor;margin-right:6px}
.ev{margin:6px 0 0;padding-left:18px;color:var(--fg)}.ev li{margin:3px 0}
.ev code,.fam code{font-size:13px}.muted{color:var(--muted)}
.caveats li{margin:6px 0}.wrap{overflow-x:auto}
svg text{fill:var(--muted);font-size:11px}
"""


def _histogram_svg(counts: Sequence[int]) -> str:
    width, height, pad = 640, 150, 22
    peak = max(counts) or 1
    bw = (width - 2 * pad) / float(len(counts))
    bars = []
    for i, c in enumerate(counts):
        h = (height - 2 * pad) * c / float(peak)
        bars.append('<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="2" '
                    'fill="var(--bar)"><title>score {lo}-{hi}: {c} documents</title></rect>'.format(
                        x=pad + i * bw + 1, y=height - pad - h, w=bw - 2, h=h, lo=i * 5, hi=i * 5 + 5, c=c))
    ticks = "".join('<text x="{0:.1f}" y="{1}" text-anchor="middle">{2}</text>'.format(
        pad + i * bw * 4, height - 4, i * 20) for i in range(6))
    return ('<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Score distribution">'
            '{bars}{ticks}</svg>').format(w=width, h=height, bars="".join(bars), ticks=ticks)


def render_html(meta: Mapping[str, Any], top: Sequence[Mapping[str, Any]]) -> str:
    e = html.escape
    total = meta.get("scored", 0)
    tiers = meta.get("tiers", {})
    cards = "".join(
        '<div class="card"><div class="k">{t}</div><div class="n t-{t}">{n}</div>'
        '<div class="muted">{share}</div></div>'.format(
            t=t, n=tiers.get(t, 0), share=_pct(tiers.get(t, 0), total) if t != "UNSCORED" else "not scored")
        for t in TIERS)
    fired = meta.get("families_fired", {})
    fam_rows = "".join(
        "<tr><td><code>{0}</code></td><td>{1}</td><td>{2}</td><td>{3:.2f}</td></tr>".format(
            e(f), fired.get(f, 0), _pct(fired.get(f, 0), total), thr)
        for f, thr in sorted((meta.get("family_thresholds") or {}).items()))
    docs = []
    for doc in top:
        fams = " · ".join('<code>{0}</code> z {1:+.2f}{2}'.format(
            e(f), r["z"], " <b>fired</b>" if r["fired"] else "")
            for f, r in sorted(doc["families"].items(), key=lambda kv: -(kv[1]["z"] or 0)))
        ev = "".join('<li><code>{0}</code> = {1:.4g} <span class="muted">(z {2:+.2f} vs {3})</span> — {4}</li>'.format(
            e(x["signal"]), x["value"], x["z"], e(x["basis"]), e(x["description"]))
            for x in [y for y in doc["evidence"] if y["z"] >= NOTABLE_Z][:5])
        docs.append('<div class="doc"><h3><span class="pill t-{tier}">{tier}</span>{id}</h3>'
                    '<div class="muted">score {score:.1f} · {words} words</div>'
                    '<div class="fam">{fams}</div><ul class="ev">{ev}</ul></div>'.format(
                        tier=e(doc["tier"]), id=e(doc["doc_id"]), score=doc["score"] or 0,
                        words=doc.get("n_words", "?"), fams=fams, ev=ev))
    labels = ""
    if meta.get("labels"):
        head = "".join("<th>{0}</th>".format(t) for t in TIERS)
        body = "".join("<tr><td>{0}</td>{1}</tr>".format(
            e(label), "".join("<td>{0}</td>".format(c.get(t, 0)) for t in TIERS))
            for label, c in sorted(meta["labels"].items()))
        labels = ('<h2>Tier by ground-truth label</h2><p class="muted">Labels are carried through for '
                  'evaluation only; scoring never reads them.</p><div class="wrap"><table><tr><th>Label</th>'
                  '{0}</tr>{1}</table></div>').format(head, body)
    notes = "".join("<li>{0}</li>".format(e(n)) for n in meta.get("notes", []))
    caveats = "".join("<li>{0}</li>".format(e(c).replace("*", "")) for c in CAVEATS)
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>chaff report</title><style>{css}</style></head>
<body><main>
<h1>chaff report</h1>
<p class="sub">{corpus} · {ndocs} documents · {scored} scored · normalised against the {norm}</p>
<div class="grid">{cards}</div>
<p class="muted">{cal}</p>
<h2>Score distribution</h2>{hist}
<h2>Families</h2><div class="wrap"><table><tr><th>Family</th><th>Fired on</th><th>Share</th>
<th>Threshold (z)</th></tr>{fam_rows}</table></div>
{labels}
<h2>Highest-scoring documents</h2>{doclist}
{notes}
<h2>Read this before acting on the results</h2><ul class="caveats">{caveats}</ul>
<p class="muted">chaff {version} · {elapsed}s</p>
</main></body></html>
""".format(css=_CSS, corpus=e(str(meta.get("corpus"))), ndocs=meta.get("documents"), scored=total,
           norm=e(str(meta.get("normalisation"))), cards=cards, cal=e(_calibration_line(meta)),
           hist=_histogram_svg(meta.get("score_histogram") or [0] * 20), fam_rows=fam_rows,
           labels=labels, doclist="".join(docs) or '<p class="muted">No scored documents.</p>',
           notes=('<h2>Notes</h2><ul>{0}</ul>'.format(notes) if notes else ""),
           caveats=caveats, version=e(str(meta.get("chaff_version"))),
           elapsed=meta.get("elapsed_seconds"))


RENDERERS = {"md": render_markdown, "json": render_json, "html": render_html}
