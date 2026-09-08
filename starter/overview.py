"""The validation report as one page: every submission, every section.

`report.py` prints the same thing to a terminal. This renders it - the
scoreboard, then each submission's sections, findings and conditions - as a
single self-contained HTML file with no external assets and no JavaScript.

It is a VERDICT document, not a dashboard. Each section already ships its own
interactive page for the numbers behind a line:

    friction    friction_report.py  -> figures/execution_stress.html
    luck        luck/pipeline.py    -> reports/
    shelf_life  shelf_life.py       -> shelf_life_<submission>.html

    python overview.py                 # -> figures/validation_report.html
    python overview.py out.html        # explicit path
    python overview.py out.html --fragment    # no <html>/<head>, for publishing
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd

import execution as ex
import report as R
from repricer import load_market

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

RANK = {"FAIL": 0, "SUSPECT": 1, "PASS": 2, "INFO": 3, "·": 4}
CLASS = {"PASS": "pass", "SUSPECT": "suspect", "FAIL": "fail", "INFO": "info"}

# What each section asks, straight from the six-sections handout - so a
# reader who has not seen it knows what a column means.
ASKS = {
    "luck": "Is the edge distinguishable from chance?",
    "lineage": "Was the test honest?",
    "friction": "Does it survive contact with the market?",
    "shelf_life": "Will it keep working — and does it work everywhere?",
    "evidence": "How much can we trust the above, given the data we have?",
    "warranty": "Is the live strategy the one we approved?",
}

CSS = """
:root { --ink:#16191d; --muted:#5c6672; --rule:#e2e6ec; --bg:#fff;
        --accent:#2d5f8b; --soft:#f7f9fb;
        --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",
               Roboto,Helvetica,Arial,sans-serif;
        --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,
               monospace; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--sans); font-size:14px; line-height:1.55;
  -webkit-font-smoothing:antialiased; }
.wrap { max-width:1120px; margin:0 auto; padding:34px 22px 70px; }
h1 { font-size:25px; margin:0 0 4px; letter-spacing:-0.01em;
  text-wrap:balance; }
h2 { font-size:12px; text-transform:uppercase; letter-spacing:0.08em;
  color:var(--muted); font-weight:600; margin:38px 0 10px;
  padding-bottom:7px; border-bottom:1px solid var(--rule); }
.lede { color:var(--muted); margin:0 0 4px; max-width:64ch; }
.stamp { font-family:var(--mono); font-size:12px; color:var(--muted);
  margin-top:10px; }

.badge { display:inline-block; font:600 11px/1 var(--mono);
  letter-spacing:0.06em; padding:5px 9px; border-radius:4px;
  border:1px solid; white-space:nowrap; }
.badge.pass { color:#2f6b46; background:#eef7f1; border-color:#bcdfc9; }
.badge.suspect { color:#8a5a12; background:#fdf5e7; border-color:#eed9ae; }
.badge.fail { color:#93332f; background:#fdeeed; border-color:#eec3c0; }
.badge.info { color:#4a5560; background:#f2f4f7; border-color:#d8dee6; }
.badge.none { color:#9aa3ad; background:transparent; border-color:transparent;
  font-size:13px; }

/* the tally */
.tally { display:flex; flex-wrap:wrap; gap:9px; margin:16px 0 4px; }
.tally .cell { border:1px solid var(--rule); border-radius:7px;
  padding:9px 14px; display:flex; align-items:baseline; gap:9px; }
.tally .n { font:600 19px/1 var(--mono); font-variant-numeric:tabular-nums; }
.tally .k { font-size:12px; color:var(--muted); }

/* the scoreboard */
.scroll { overflow-x:auto; }
table.board { border-collapse:collapse; width:100%; margin-top:6px;
  font-size:13px; }
table.board th { text-align:center; font-weight:600; font-size:11px;
  text-transform:uppercase; letter-spacing:0.05em; color:var(--muted);
  padding:0 8px 9px; vertical-align:bottom; white-space:nowrap; }
table.board th.sub, table.board td.sub { text-align:left; }
table.board td { padding:7px 8px; text-align:center;
  border-top:1px solid var(--rule); }
table.board td.sub { font-weight:640; white-space:nowrap; }
table.board td.sub a { color:var(--ink); text-decoration:none;
  border-bottom:1px solid var(--rule); }
table.board td.sub a:hover { border-bottom-color:var(--accent); }
table.board td.overall { border-left:1px solid var(--rule); }
table.board th.overall { border-left:1px solid var(--rule); }
.asks { margin:14px 0 0; padding:0; list-style:none; column-gap:34px;
  columns:2; font-size:12.5px; color:var(--muted); }
.asks li { break-inside:avoid; margin:3px 0; }
.asks code { font-family:var(--mono); font-size:11.5px; color:var(--ink); }

/* one submission */
.sub { margin-top:40px; scroll-margin-top:14px; }
.subhead { display:flex; align-items:center; gap:13px; flex-wrap:wrap;
  padding:12px 15px; border:1px solid var(--rule); border-radius:8px; }
.subhead.pass { background:#f2f9f5; border-color:#cfe6d8; }
.subhead.suspect { background:#fdf8ef; border-color:#eee0c2; }
.subhead.fail { background:#fdf2f1; border-color:#eecfcd; }
.subhead .name { font-weight:660; font-size:17px; }
.subhead .why { color:var(--muted); font-size:12.5px; margin-left:auto;
  text-align:right; }

.sec { margin:16px 0 0; }
.sechead { display:flex; align-items:baseline; gap:10px; margin:0 0 2px;
  padding:9px 0 6px; border-bottom:1px solid var(--rule); }
.sechead .sname { font-family:var(--mono); font-size:12.5px; font-weight:600;
  color:var(--accent); }
.sechead .ask { font-size:12.5px; color:var(--muted); }

.find { display:grid; grid-template-columns:82px 1fr; gap:11px;
  align-items:baseline; padding:6px 0 6px 2px;
  border-bottom:1px solid #f0f3f7; }
.find:last-child { border-bottom:0; }
.find .fname { font-family:var(--mono); font-size:12px; color:var(--ink);
  word-break:break-word; }
.find .fnote { color:#39414a; font-size:13px; }
.find .fwrap { display:grid; grid-template-columns:180px 1fr; gap:11px;
  align-items:baseline; }
@media (max-width:720px) {
  .find { grid-template-columns:74px 1fr; }
  .find .fwrap { grid-template-columns:1fr; gap:2px; }
  .asks { columns:1; }
}

.conds { margin:12px 0 0; padding:11px 14px; border-radius:7px;
  background:var(--soft); border:1px solid var(--rule); }
.conds .ch { font-size:11px; text-transform:uppercase; letter-spacing:0.07em;
  color:var(--muted); font-weight:600; margin-bottom:5px; }
.conds ul { margin:0; padding-left:18px; font-size:13px; }
.conds li { margin:3px 0; }
.nocond { color:var(--muted); font-size:12.5px; margin-top:11px; }

footer { margin-top:52px; padding-top:14px; border-top:1px solid var(--rule);
  color:var(--muted); font-size:12px; }
footer code { font-family:var(--mono); font-size:11.5px; }
"""


TITLE_TAG = "<title>Signal Gauntlet Validation</title>"

DOCTYPE = '<!doctype html><html lang="en">'

HEAD = (
    '<head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    f"{TITLE_TAG}"
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=IBM+Plex+Mono:wght@400;500;600&"
    'family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">')


def _e(x) -> str:
    return html.escape(str(x))


def badge(v: str) -> str:
    if v == "·":
        return '<span class="badge none">·</span>'
    return f'<span class="badge {CLASS.get(v, "info")}">{_e(v)}</span>'


def tally(reports: dict) -> str:
    counts = {v: 0 for v in ("PASS", "SUSPECT", "FAIL")}
    for r in reports.values():
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    cells = "".join(
        f'<div class="cell">{badge(v)}<span class="n">{n}</span>'
        f'<span class="k">of {len(reports)}</span></div>'
        for v, n in counts.items())
    return f'<div class="tally">{cells}</div>'


def scoreboard(reports: dict) -> str:
    grid = R.grid(reports)
    grid = grid.sort_values("OVERALL", key=lambda s: s.map(RANK), kind="stable")

    head = "".join(f'<th{" class=\'overall\'" if c == "OVERALL" else ""}>'
                   f"{_e(c.replace('_', ' '))}</th>"
                   for c in grid.columns)
    rows = []
    for sub, row in grid.iterrows():
        cells = "".join(
            f'<td{" class=\'overall\'" if c == "OVERALL" else ""}>'
            f"{badge(row[c])}</td>" for c in grid.columns)
        rows.append(f'<tr><td class="sub"><a href="#{_e(sub)}">{_e(sub)}</a>'
                    f"</td>{cells}</tr>")

    asks = "".join(f"<li><code>{_e(k)}</code> — {_e(v)}</li>"
                   for k, v in ASKS.items())
    return (f'<div class="scroll"><table class="board"><thead><tr>'
            f'<th class="sub">submission</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            f'<ul class="asks">{asks}</ul>')


def findings(section) -> str:
    out = []
    for f in section.findings:
        out.append(
            f'<div class="find">{badge(f.verdict)}'
            f'<div class="fwrap"><span class="fname">{_e(f.name)}</span>'
            f'<span class="fnote">{_e(f.note)}</span></div></div>')
    return "".join(out)


def submission(key: str, rep) -> str:
    cls = CLASS.get(rep.verdict, "info")
    worst = [s.section for s in rep.sections if s.verdict == rep.verdict]
    why = (f"{rep.verdict.lower()} on {', '.join(worst)}" if worst
           else "no section built")

    secs = []
    for s in rep.sections:
        secs.append(
            f'<div class="sec"><div class="sechead">{badge(s.verdict)}'
            f'<span class="sname">{_e(s.section)}</span>'
            f'<span class="ask">{_e(ASKS.get(s.section, ""))}</span></div>'
            f"{findings(s)}</div>")

    if rep.conditions:
        items = "".join(f"<li>{_e(c)}</li>" for c in rep.conditions)
        conds = ('<div class="conds"><div class="ch">Conditions — mandatory '
                 f"before funding</div><ul>{items}</ul></div>")
    elif rep.verdict == "SUSPECT":
        conds = '<p class="nocond">SUSPECT with no conditions recorded.</p>'
    else:
        conds = ""

    return (f'<section class="sub" id="{_e(key)}">'
            f'<div class="subhead {cls}"><span class="name">{_e(key)}</span>'
            f'{badge(rep.verdict)}<span class="why">{_e(why)}</span></div>'
            f'{"".join(secs)}{conds}</section>')


def render(reports: dict, fragment: bool = False) -> str:
    built = sorted({s.section for r in reports.values() for s in r.sections},
                   key=R.SECTIONS.index)
    unbuilt = [s for s in R.SECTIONS if s not in built]
    stamp = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")

    body = [
        '<div class="wrap">',
        "<h1>Signal Gauntlet · validation report</h1>",
        '<p class="lede">Seven strategies proposed for capital, each run '
        "through the same pipeline. Every verdict below points at the number "
        "that produced it.</p>",
        f'<p class="stamp">{len(built)} of {len(R.SECTIONS)} sections built · '
        f"{_e(', '.join(built))} · {stamp}</p>",
        tally(reports),
        "<h2>Scoreboard</h2>",
        scoreboard(reports),
    ]
    if unbuilt:
        body.append(
            f'<p class="nocond">{_e(", ".join(unbuilt))} '
            f"{'are' if len(unbuilt) > 1 else 'is'} not built yet and "
            "score nothing — a verdict here is the verdict of the sections "
            "that ran, not of all six.</p>")

    body.append("<h2>Submission by submission</h2>")
    order = sorted(reports, key=lambda k: (RANK[reports[k].verdict], k))
    body += [submission(k, reports[k]) for k in order]

    body.append(
        "<footer>Built by <code>overview.py</code> from "
        "<code>report.run_all</code>. The numbers behind each line live in "
        "each section's own page: <code>friction_report.py</code>, "
        "<code>luck/pipeline.py</code>, <code>shelf_life.py</code>.</footer>"
        "</div>")

    page = f"<style>{CSS}</style>" + "".join(body)
    if fragment:
        # for publishing: the host supplies <html>/<head>/<body>
        return f"{TITLE_TAG}{page}"
    return (f"{DOCTYPE}{HEAD}</head><body>{page}</body></html>")


def build(out_path: str | Path | None = None, fragment: bool = False) -> Path:
    sys.path.insert(0, str(ROOT / "registry"))
    from loader import load_registry                          # noqa: E402

    reports = R.run_all(load_registry(ROOT / "registry"),
                        load_market(ROOT / "data"),
                        ROOT / "blotters",
                        ex.Tape.load(ROOT / "data"))
    out = Path(out_path) if out_path else HERE / "figures" / "validation_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(reports, fragment=fragment), encoding="utf-8")
    print(f"wrote {out}  ({sum(r.verdict == 'PASS' for r in reports.values())}"
          f" PASS, {sum(r.verdict == 'SUSPECT' for r in reports.values())}"
          f" SUSPECT, {sum(r.verdict == 'FAIL' for r in reports.values())} FAIL)")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    build(args[0] if args else None, fragment="--fragment" in sys.argv)
