"""The build target: an automated validation report.

One pipeline, run unchanged over all seven submissions:

    report = run(registry_entry, blotter, market)
    report.verdict  ->  "PASS" | "SUSPECT" | "FAIL"

This file defines the contract only. The six sections — what each one should
compute — are the six-sections handout; the evidence behind every verdict
line is the point of the exercise. Nothing below does any validation yet.
"""
from __future__ import annotations

import html
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SECTIONS = ["luck", "lineage", "friction", "shelf_life", "evidence", "warranty"]

# The question each section answers, from the six-sections handout. Used as
# the heading on the assembled report, including for sections not yet built.
SECTION_TITLES = {
    "luck":       ("Luck", "Is the edge distinguishable from chance?"),
    "lineage":    ("Lineage", "Was the test honest?"),
    "friction":   ("Friction", "Does it survive contact with the market?"),
    "shelf_life": ("Shelf-life", "Will it keep working — and does it work "
                                 "everywhere?"),
    "evidence":   ("Evidence", "How much can we trust all of the above, given "
                               "the data we have?"),
    "warranty":   ("The Warranty", "Is the live strategy the one we approved?"),
}

# A section module may expose:
#
#     contribute(registry_entry, blotter, market, out_dir) -> (SectionResult, str)
#
# where the str is body HTML (no <html>/<head>/<style>) using the classes this
# file's stylesheet defines. It may carry its own <script>: the assembled page
# loads Plotly in <head>, so it is defined by the time body scripts run.
# Returning "" means "render me with the generic findings table".
#
# One call returns both so a section computes once — several of these do
# thousands of bootstrap resamples and must not be run twice per report.
# A section without `contribute` is called through an adapter in `run()` and
# rendered by `_generic_fragment`, which is why the four unbuilt sections can
# land without this file changing.
PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/4.0.0/plotly.min.js"

BADGE = {"PASS": "#0f8a3c", "SUSPECT": "#b8860b", "FAIL": "#c22f2f",
         "INFO": "#8a8a85", "·": "#c8c8c2"}

# Sections whose verdicts are declared rather than computed. Marked on the
# scoreboard so nobody reads a hand-entered cell as measured evidence.
# Empty since Friction started measuring fills against the executed tape;
# put a section back in here the moment its verdict is entered by hand again.
PROVISIONAL_SECTIONS: set[str] = set()

VERDICTS = ("PASS", "SUSPECT", "FAIL")


@dataclass
class Finding:
    """One test: a name, a number (or figure path), a verdict, a sentence."""
    name: str
    value: object
    verdict: str = "INFO"          # PASS | SUSPECT | FAIL | INFO
    note: str = ""


@dataclass
class SectionResult:
    section: str                   # one of SECTIONS
    verdict: str                   # PASS | SUSPECT | FAIL | INFO
    findings: list[Finding] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    """What this section requires if it is not a clean PASS. Sections state
    their own; `Report.conditions` collects them."""


@dataclass
class Report:
    submission: str
    sections: list[SectionResult] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)   # mandatory if SUSPECT
    fragments: dict[str, str] = field(default_factory=dict)
    """Body HTML each section contributed, kept so the combined document can
    be assembled without recomputing seven reports' worth of bootstraps."""

    @property
    def verdict(self) -> str:
        """Aggregation rule: any section FAIL fails the submission; any
        SUSPECT (and no FAIL) makes it SUSPECT — and a SUSPECT report with an
        empty conditions list is an unfinished report.

        INFO aggregates as SUSPECT, never as PASS. A section that could not
        run has found nothing, which is not the same as having found nothing
        wrong; letting it through as a pass is exactly the mistake the
        Lineage pages refuse to make one level down.
        """
        vs = [s.verdict for s in self.sections]
        if "FAIL" in vs:
            return "FAIL"
        if "SUSPECT" in vs or "INFO" in vs:
            return "SUSPECT"
        return "PASS"


def grid(reports: dict[str, Report],
         submissions: list[str] | None = None) -> pd.DataFrame:
    """The scoreboard: one row per submission, one column per section, plus
    the overall verdict. Unfilled cells show '·'. The day fills this in;
    the reveal judges it."""
    subs = submissions if submissions is not None else sorted(reports)
    rows = {}
    for sub in subs:
        row = {s: "·" for s in SECTIONS} | {"OVERALL": "·"}
        if sub in reports:
            r = reports[sub]
            row.update({s.section: s.verdict for s in r.sections})
            row["OVERALL"] = r.verdict
        rows[sub] = row
    return pd.DataFrame(rows).T[SECTIONS + ["OVERALL"]]


def _fallback_conditions(s: SectionResult) -> list[str]:
    """A condition for a non-PASS section that did not state its own.

    Never silently empty: a SUSPECT report with no conditions is, by this
    file's own rule, an unfinished report.
    """
    if s.verdict == "PASS":
        return []
    if s.verdict == "INFO":
        return [f"The {s.section} section could not run on this submission, "
                f"so its question is unruled-out rather than answered. Do not "
                f"fund on this section's silence: clear it with a test that "
                f"does run."]
    flagged = [f for f in s.findings if f.verdict in ("SUSPECT", "FAIL")]
    if flagged:
        return [f"{s.section}: {f.name} = {f.value} — {f.note}"
                for f in flagged]
    return [f"The {s.section} section returned {s.verdict}."]


_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#fff;color:#1a1a1a;
font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:34px 24px 72px}
h1{font-size:21px;font-weight:660;margin:0 0 3px;letter-spacing:-.01em}
.badge{display:inline-block;font-size:11px;font-weight:650;color:#fff;
padding:2px 9px;border-radius:3px;vertical-align:3px;margin-left:8px}
.sub{color:#8a8a85;font-size:12.5px;margin:0 0 8px}
.conds{margin:18px 0 0;padding:14px 16px;border:1px solid #ececE7;
border-left:3px solid #b8860b;background:#fcfbf7}
.conds h3{font-size:12px;font-weight:650;margin:0 0 7px;
text-transform:uppercase;letter-spacing:.05em;color:#52514e}
.conds ol{margin:0;padding-left:18px;color:#52514e;font-size:12.5px}
.conds li{margin:0 0 6px}
.sec{margin:34px 0 0;padding:22px 0 0;border-top:1px solid #ececE7}
.sec.todo{opacity:.55}
.sechead{display:flex;align-items:baseline;gap:9px;margin:0 0 2px}
.sechead h2{font-size:15px;font-weight:650;margin:0}
.sechead .n{color:#b6b6b0;font-size:11px;font-weight:650}
.q{color:#8a8a85;font-size:12.5px;margin:0 0 14px}
.cap{color:#8a8a85;font-size:12px;margin:0 0 8px}
.todo .note{color:#8a8a85;font-size:12.5px;font-style:italic;margin:0}
table{border-collapse:collapse;width:100%;font-size:12.5px}
td,th{padding:6px 8px;border-bottom:1px solid #ececE7;text-align:right}
td:first-child,th:first-child{text-align:left;color:#52514e}
th{color:#8a8a85;font-weight:600;font-size:11px}
td.num{font-variant-numeric:tabular-nums;font-weight:600;color:#1a1a1a}
td.hot{color:#c22f2f}
.v{font-weight:650;font-size:11px}
.foot{color:#8a8a85;font-size:11.5px;margin:10px 0 0}
a{color:#2a78d6}
.panel{margin:12px 0 0;border:1px solid #ececE7;border-radius:2px;
overflow:hidden;background:#fff}
.panel iframe{display:block;width:100%;border:0;overflow:hidden}
"""


def display_verdict(v: str) -> str:
    """A SECTION's verdict as the report shows it: INFO reads as SUSPECT.

    INFO means the section could not run on this submission. It has always
    aggregated as SUSPECT — a test that found nothing is not a test that
    found nothing wrong — and showing it as SUSPECT makes the page agree
    with the rule instead of asking the reader to know that INFO is not a
    pass. Why it could not run stays in the section's findings and in its
    conditions, so nothing is lost but the ambiguous label.

    FINDING-level INFO is left alone: those are informational lines (which
    execution basis was used, where a page was written), not failed tests.
    """
    return "SUSPECT" if v == "INFO" else v


def _badge(v: str) -> str:
    return f'<span class="badge" style="background:{BADGE.get(v, "#8a8a85")}">{v}</span>'


def panel(page: "Path | str", caption: str = "", height: int = 600) -> str:
    """Embed a section's own page as an isolated frame, sized to its content.

    Sections were authored independently and carry their own stylesheets,
    their own Plotly versions and their own element ids. An iframe gives each
    one its own document, so its buttons and sliders keep working and no
    section can restyle or break another.

    The frame does NOT scroll. `section_adapters.prepare_panel` injects a
    reporter into each panel page that posts its height up, and the listener
    on this page grows the frame to match — so the report is one continuous
    scroll rather than a scrollbar inside a scrollbar, and a reader working a
    toggle or a slider re-flows the page instead of hunting inside a window.
    `height` is only the placeholder used until the first message arrives.
    """
    name = Path(str(page)).name
    cap = f'<p class="cap">{html.escape(caption)}</p>' if caption else ""
    return (f'{cap}<div class="panel"><iframe src="{html.escape(name)}" '
            f'height="{height}" scrolling="no" loading="lazy" '
            f'title="{html.escape(name)}"></iframe></div>'
            f'<p class="foot"><a href="{html.escape(name)}" target="_blank">'
            f'Open {html.escape(name)} in a new tab &rarr;</a></p>')


# Grows each frame to the height its page reports. Cross-origin by necessity:
# file:// iframes are opaque origins, so the child volunteers the number and
# the parent matches it by comparing contentWindow against the event source.
_AUTOHEIGHT_LISTENER = """
<script>addEventListener("message", function(e){
  var d = e && e.data;
  if (!d || typeof d.__panelHeight !== "number") return;
  var f = document.querySelectorAll("iframe"), i;
  for (i = 0; i < f.length; i++) {
    if (f[i].contentWindow === e.source) {
      f[i].style.height = Math.ceil(d.__panelHeight) + "px";
      return;
    }
  }
});</script>
"""


def findings_table(res: "SectionResult") -> str:
    """The uniform findings table every section is rendered with."""
    return _generic_fragment(res)


def _generic_fragment(res: "SectionResult") -> str:
    """Default rendering for a section that supplies no `fragment()`.

    Every SectionResult already carries named findings with verdicts and a
    sentence each, so this is enough for a section to appear usefully on the
    report the day it is written, before anyone styles it.
    """
    rows = ""
    for f in res.findings:
        if str(f.name).endswith("_page"):
            rows += (f'<tr><td>{html.escape(f.name)}</td>'
                     f'<td class="num"><a href="{html.escape(Path(str(f.value)).name)}">'
                     f'open &rarr;</a></td></tr>')
            continue
        val = "" if f.value is None else html.escape(str(f.value))
        rows += (f'<tr><td>{html.escape(f.name)}<div class="cap" '
                 f'style="margin:2px 0 0">{html.escape(f.note)}</div></td>'
                 f'<td class="num">{val}<div class="v" '
                 f'style="color:{BADGE.get(f.verdict, "#8a8a85")}">'
                 f'{f.verdict}</div></td></tr>')
    return f"<table><tbody>{rows}</tbody></table>" if rows else ""


def _placeholder(name: str) -> str:
    return ('<p class="note">Not yet built. This section will appear here '
            'when its module lands.</p>')


def report_body(rep: "Report", registry_entry: dict,
                fragments: dict[str, str], heading_level: int = 1) -> str:
    """One submission's report as body HTML, with no page wrapper.

    Shared by the standalone page and the combined document so the two can
    never drift apart. `heading_level` demotes the title to <h2> when the
    report is one article inside the combined page.
    """
    name = html.escape(str(registry_entry.get("name", rep.submission)))
    cls = html.escape(str(registry_entry.get("class", "")))
    by = {s.section: s for s in rep.sections}
    h = f"h{heading_level}"

    body = ""
    for i, key in enumerate(SECTIONS, start=1):
        title, question = SECTION_TITLES[key]
        res = by.get(key)
        todo = "" if res else " todo"
        inner = (fragments.get(key) or _generic_fragment(res)) if res \
            else _placeholder(key)
        body += f"""
  <section class="sec{todo}" id="{rep.submission}-{key}">
    <div class="sechead"><span class="n">{i}</span>
      <h2>{html.escape(title)}</h2>
      {_badge(display_verdict(res.verdict)) if res else _badge("·")}</div>
    <p class="q">{html.escape(question)}</p>
    {inner}
  </section>"""

    conds = ""
    if rep.conditions:
        items = "".join(f"<li>{html.escape(c)}</li>" for c in rep.conditions)
        conds = (f'<div class="conds"><h3>Conditions attached</h3>'
                 f'<ol>{items}</ol></div>')

    built = sum(1 for k in SECTIONS if k in by)
    return f"""
<{h} id="{rep.submission}">{name}{_badge(rep.verdict)}</{h}>
<p class="sub">Validation report{f" &middot; {cls}" if cls else ""}
  &middot; {built} of {len(SECTIONS)} sections built &middot; priced at mid</p>
{conds}
{body}"""


def render_html(rep: "Report", registry_entry: dict, blotter: pd.DataFrame,
                market: dict, fragments: dict[str, str]) -> str:
    """The whole validation report for one submission, as one page."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(rep.submission)} &mdash; validation report</title>
<script src="{PLOTLY}"></script>
<style>{_CSS}</style></head><body><div class="wrap">
{report_body(rep, registry_entry, fragments, heading_level=1)}
</div>{_AUTOHEIGHT_LISTENER}</body></html>"""


_OVERVIEW_CSS = _CSS + """
.grid{border-collapse:collapse;width:100%;font-size:12.5px;margin:18px 0 0}
.grid th,.grid td{padding:9px 10px;border-bottom:1px solid #ececE7;
text-align:center;white-space:nowrap}
.grid th:first-child,.grid td:first-child{text-align:left}
.grid thead th{color:#8a8a85;font-weight:600;font-size:11px;
border-bottom:1px solid #d9d9d4}
.grid tbody tr:hover{background:#fafafa}
.grid td.sub{font-weight:600}
.grid td.sub a{text-decoration:none}
.grid td.sub a:hover{text-decoration:underline}
.grid td.overall{border-left:1px solid #ececE7;font-weight:650}
.cell{display:inline-block;min-width:62px;padding:2px 7px;border-radius:3px;
font-size:10.5px;font-weight:650;letter-spacing:.01em}
.legend{color:#8a8a85;font-size:11.5px;margin:16px 0 0}
.sumhead{font-size:14px;font-weight:650;margin:38px 0 0;padding-top:22px;
border-top:1px solid #ececE7}
.summary{list-style:none;margin:14px 0 0;padding:0}
.summary li{padding:11px 0;border-bottom:1px solid #f2f2ee}
.summary li:last-child{border-bottom:0}
.summary li > a{font-weight:650;text-decoration:none;font-size:13px}
.summary li > a:hover{text-decoration:underline}
.summary .cell{margin-left:8px;min-width:0}
.summary p{margin:5px 0 0;color:#52514e;font-size:12.5px;max-width:88ch}
.summary p b{color:#1a1a1a;font-weight:650}
"""

_CELL_BG = {"PASS": "#e9f6e9", "SUSPECT": "#fdf3dc", "FAIL": "#fbeaea",
            "INFO": "#f0f0ec", "·": "transparent"}
_CELL_INK = {"PASS": "#0a7a0a", "SUSPECT": "#8a6000", "FAIL": "#a92c2c",
             "INFO": "#78838f", "·": "#c8c8c2"}


def _cell(v: str) -> str:
    return (f'<span class="cell" style="background:{_CELL_BG.get(v, "#f0f0ec")};'
            f'color:{_CELL_INK.get(v, "#78838f")}">{html.escape(v)}</span>')


_VERDICT_RANK = {"PASS": 0, "SUSPECT": 1, "FAIL": 2}


def rank(reports: dict[str, "Report"]) -> list[str]:
    """Submissions worst-last: fundable at the top, void at the bottom.

    Sorted on, in order:

      1. the overall verdict — PASS, then SUSPECT, then FAIL;
      2. whether LINEAGE failed. An integrity failure is not the same kind
         of thing as a weak one. A strategy whose trades used information
         that did not exist yet has not underperformed — its whole record
         is void, and no amount of Sharpe redeems it. So it sorts below
         every other FAIL regardless of how the rest of its report reads;
      3. how many sections failed, then how many are unresolved.

    This is a rule, not a hand-placed order: whichever submission is
    caught by Lineage lands at the bottom, and the ranking survives
    Friction landing or the reveal changing the answers.
    """
    def key(sub: str):
        rep = reports[sub]
        by = {s.section: s.verdict for s in rep.sections}
        return (_VERDICT_RANK.get(rep.verdict, 1),
                1 if by.get("lineage") == "FAIL" else 0,
                sum(1 for v in by.values() if v == "FAIL"),
                sum(1 for v in by.values() if v in ("SUSPECT", "INFO")),
                sub)
    return sorted(reports, key=key)


def _driver(rep: "Report") -> tuple["SectionResult | None", "Finding | None"]:
    """The section that decided this verdict, and the finding that drove it."""
    fails = [s for s in rep.sections if s.verdict == "FAIL"]
    unresolved = [s for s in rep.sections if s.verdict in ("SUSPECT", "INFO")]
    sec = (fails or unresolved or [None])[0]
    if sec is None:
        return None, None
    bad = [f for f in sec.findings if f.verdict == "FAIL"] or \
          [f for f in sec.findings if f.verdict == "SUSPECT"]
    return sec, (bad[0] if bad else None)


def summarise(rep: "Report") -> str:
    """One sentence on where a submission falls down.

    Taken from the section that set the verdict and the finding that drove
    it, so the line cannot drift from the report it summarises — if a
    threshold moves, this moves with it.
    """
    sec, finding = _driver(rep)
    if sec is None:
        n = len(rep.sections)
        return (f"Clean on all {n} built section{'s' if n != 1 else ''} — "
                f"nothing to answer for yet.")

    title = SECTION_TITLES.get(sec.section, (sec.section, ""))[0]
    shown = display_verdict(sec.verdict)
    if finding is not None and finding.note:
        lead = f"<b>{html.escape(title)} {shown.lower()}</b> — {html.escape(finding.note)}"
    elif sec.verdict == "INFO":
        lead = (f"<b>{html.escape(title)} {shown.lower()}</b> — the test could "
                f"not run on this submission, so its question is unanswered "
                f"rather than answered well")
    else:
        lead = f"<b>{html.escape(title)} {shown.lower()}</b>"

    others = [SECTION_TITLES.get(s.section, (s.section, ""))[0]
              for s in rep.sections
              if s is not sec and s.verdict in ("FAIL", "SUSPECT", "INFO")]
    tail = (f" Also unresolved on {html.escape(', '.join(others))}." if others
            else "")
    return lead.rstrip(".") + "." + tail


def render_overview(reports: dict[str, "Report"],
                    submissions: list[str] | None = None) -> str:
    """The scoreboard: seven submissions down, six sections across.

    `grid()` already computes exactly this table; this only paints it and
    links each row to its full report. Unbuilt sections stay '·' rather than
    being scored, so the page never implies more coverage than exists.
    """
    df = grid(reports, submissions if submissions is not None else rank(reports))
    built = sorted({s.section for r in reports.values() for s in r.sections})

    head = "".join(
        f"<th>{html.escape(SECTION_TITLES[c][0] if c in SECTION_TITLES else c)}"
        f"{'<sup>†</sup>' if c in PROVISIONAL_SECTIONS else ''}</th>"
        for c in SECTIONS)
    body = ""
    for sub, row in df.iterrows():
        cells = "".join(f"<td>{_cell(display_verdict(str(row[c])))}</td>"
                        for c in SECTIONS)
        body += (f'<tr><td class="sub">'
                 f'<a href="{html.escape(str(sub))}-report.html">'
                 f'{html.escape(str(sub))}</a></td>{cells}'
                 f'<td class="overall">{_cell(str(row["OVERALL"]))}</td></tr>')

    tally = {v: sum(1 for r in reports.values() if r.verdict == v)
             for v in ("PASS", "SUSPECT", "FAIL")}

    summaries = "".join(
        f'<li><a href="{html.escape(sub)}-report.html">{html.escape(sub)}</a>'
        f'{_cell(reports[sub].verdict)}'
        f'<p>{summarise(reports[sub])}</p></li>'
        for sub in df.index)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Validation scoreboard</title>
<style>{_OVERVIEW_CSS}</style></head><body><div class="wrap">

<h1>Validation scoreboard</h1>
<p class="sub">{len(df)} submissions &middot; {len(built)} of
  {len(SECTIONS)} sections built &middot; priced at mid &middot;
  {tally['PASS']} pass, {tally['SUSPECT']} suspect, {tally['FAIL']} fail</p>

<table class="grid">
  <thead><tr><th>submission</th>{head}<th>overall</th></tr></thead>
  <tbody>{body}</tbody>
</table>

<h2 class="sumhead">Where each one falls down</h2>
<ol class="summary">{summaries}</ol>

<p class="legend">† Friction's verdicts are declared, not computed: they were
entered by hand pending the module that reprices at crossing prices, and carry
no evidence behind them yet.</p>

<p class="legend">Any section FAIL fails the submission; any SUSPECT (and no
FAIL) makes it SUSPECT. A section that could not run on a submission shows as
SUSPECT, never as a pass — the reason is in that submission's report.
&middot; marks a section not yet built. Each submission links to its full
report.</p>
</div></body></html>"""


_COMBINED_CSS = _OVERVIEW_CSS + """
.toc{columns:2;margin:16px 0 0;padding:0 0 0 18px;font-size:12.5px}
.toc li{margin:0 0 4px}
article{margin:52px 0 0;padding:34px 0 0;border-top:2px solid #1a1a1a}
article h2{font-size:19px;font-weight:660;margin:0 0 3px;letter-spacing:-.01em}
.back{font-size:11.5px;margin:6px 0 0}
@media print{
  article{break-before:page;border-top:0}
  .back{display:none}
  .panel iframe{break-inside:avoid}
}
"""


def render_combined(reports: dict[str, "Report"], registry: dict,
                    submissions: list[str] | None = None) -> str:
    """Every submission's full report in one self-contained document.

    The scoreboard first, then each report in rank order, worst last. Built
    from the same `report_body` the standalone pages use, so the two cannot
    disagree — and from the fragments each Report already carries, so nothing
    is recomputed.

    Section panels that live in their own file stay as iframes, so this page
    still expects its siblings in the same directory. It is one document to
    read and to print, not one file to email on its own.
    """
    order = submissions if submissions is not None else rank(reports)
    df = grid(reports, order)
    built = sorted({s.section for r in reports.values() for s in r.sections})
    tally = {v: sum(1 for r in reports.values() if r.verdict == v)
             for v in ("PASS", "SUSPECT", "FAIL")}

    head = "".join(
        f"<th>{html.escape(SECTION_TITLES[c][0] if c in SECTION_TITLES else c)}"
        f"{'<sup>†</sup>' if c in PROVISIONAL_SECTIONS else ''}</th>"
        for c in SECTIONS)
    rows = ""
    for sub, row in df.iterrows():
        cells = "".join(f"<td>{_cell(display_verdict(str(row[c])))}</td>"
                        for c in SECTIONS)
        rows += (f'<tr><td class="sub"><a href="#{html.escape(str(sub))}">'
                 f'{html.escape(str(sub))}</a></td>{cells}'
                 f'<td class="overall">{_cell(str(row["OVERALL"]))}</td></tr>')

    summaries = "".join(
        f'<li><a href="#{html.escape(sub)}">{html.escape(sub)}</a>'
        f'{_cell(reports[sub].verdict)}'
        f'<p>{summarise(reports[sub])}</p></li>' for sub in order)

    articles = "".join(
        f'<article>{report_body(reports[sub], registry.get(sub, {}), reports[sub].fragments, heading_level=2)}'
        f'<p class="back"><a href="#top">Back to the scoreboard</a></p></article>'
        for sub in order)

    provisional = "" if not PROVISIONAL_SECTIONS else (
        '<p class="legend">† Verdicts in this column are declared, not '
        'computed, and carry no evidence behind them yet.</p>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Validation report &mdash; all submissions</title>
<script src="{PLOTLY}"></script>
<style>{_COMBINED_CSS}</style></head><body><div class="wrap" id="top">

<h1>Validation report</h1>
<p class="sub">{len(df)} submissions &middot; {len(built)} of
  {len(SECTIONS)} sections built &middot; priced at mid &middot;
  {tally['PASS']} pass, {tally['SUSPECT']} suspect, {tally['FAIL']} fail</p>

<table class="grid">
  <thead><tr><th>submission</th>{head}<th>overall</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
{provisional}
<p class="legend">Any section FAIL fails the submission; any SUSPECT (and no
FAIL) makes it SUSPECT. A section that could not run shows as SUSPECT, never
as a pass &mdash; the reason is in that submission's report below.
&middot; marks a section not yet built.</p>

<h2 class="sumhead">Where each one falls down</h2>
<ol class="summary">{summaries}</ol>
{articles}
</div>{_AUTOHEIGHT_LISTENER}</body></html>"""


def write_combined(reports: dict[str, "Report"], registry: dict,
                   out_dir: str | Path = "reports",
                   filename: str = "validation-report.html") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(render_combined(reports, registry), encoding="utf-8")
    return path


def write_overview(reports: dict[str, "Report"],
                   out_dir: str | Path = "reports",
                   filename: str = "overview.html") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(render_overview(reports), encoding="utf-8")
    return path


def run_all(registry: dict, blotters: dict, market: dict,
            out_dir: str | Path = "reports", tape=None) -> dict[str, "Report"]:
    """Every submission through `run()`, plus the scoreboard.

    The tape is loaded once here and handed to each submission, rather than
    letting seven Friction sections each parse it.
    """
    if tape is None:
        try:
            import execution as _ex
            tape = _ex.Tape.load(Path(__file__).resolve().parent.parent / "data")
        except Exception:                                 # noqa: BLE001
            tape = None      # friction falls back to loading it itself
    reports = {k: run(registry[k], blotters[k], market, out_dir, tape)
               for k in sorted(registry)}
    if out_dir is not None:
        write_overview(reports, out_dir)
        write_combined(reports, registry, out_dir)
    return reports


def write_report_html(rep: "Report", registry_entry: dict,
                      blotter: pd.DataFrame, market: dict,
                      fragments: dict[str, str],
                      out_dir: str | Path = "reports") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{rep.submission}-report.html"
    path.write_text(render_html(rep, registry_entry, blotter, market,
                                fragments), encoding="utf-8")
    return path


def run(registry_entry: dict, blotter: pd.DataFrame, market: dict,
        out_dir: str | Path | None = "reports", tape=None) -> Report:
    """Take one submission's records, return the verdict with evidence.

    One submission in, one Report out, with each section's HTML page written
    to `out_dir` (pass None to compute verdicts without writing pages).

    `tape` is the executed trade tape, which only Friction reads; it loads
    from data/ on demand when not supplied. Pass it in when running the
    whole cohort so the 1.8M-row file is parsed once, not seven times.

    Sections are imported inside the function on purpose: every section
    module imports Finding/SectionResult from this one, so a module-level
    import here would be circular.

    Only the sections that exist are assembled. `grid()` renders the rest as
    unfilled, which is the honest picture — a Report is not complete until
    all six are in it.
    """
    import leakage_report as LR
    import evidence_report as ER

    sub = str(registry_entry.get("submission", "?"))
    sections: list[SectionResult] = []
    fragments: dict[str, str] = {}

    # LR.section always writes its own standalone page, so when pages are not
    # wanted it is pointed at a scratch directory that is thrown away.
    with tempfile.TemporaryDirectory() as tmp:
        page_dir = out_dir if out_dir is not None else tmp

        # LINEAGE — the revision test, with the participation fallback for
        # books that have no buy-versus-sell to explain. Its INFO is kept
        # rather than collapsed; the aggregation rule above handles it. No
        # `contribute` yet, so it renders through the generic findings table
        # and links out to the standalone page it already writes.
        sections.append(LR.section(sub, registry_entry, blotter,
                                   market["forecasts"], market["products"],
                                   page_dir))

        # EVIDENCE — what sample stands behind the verdicts above.
        ev, ev_html = ER.contribute(registry_entry, blotter, market, page_dir)
        sections.append(ev)
        fragments["evidence"] = ev_html

        # LUCK and SHELF-LIFE were written as standalone tools; the adapters
        # in section_adapters.py give them the contribute() shape without
        # changing their code. Each renders its own page, embedded as an
        # isolated frame so its stylesheet, Plotly version and controls
        # cannot collide with the rest of the report.
        import section_adapters as SA
        for name, fn in (("luck", SA.luck_contribute),
                         ("shelf_life", SA.shelf_life_contribute)):
            try:
                res, body = fn(registry_entry, blotter, market, page_dir)
                sections.append(res)
                fragments[name] = body
            except Exception as exc:                      # noqa: BLE001
                # A section that blows up must not take the report with it —
                # it is recorded as INFO, which aggregates as SUSPECT.
                sections.append(SectionResult(
                    name, "INFO",
                    [Finding(f"{name}_error", type(exc).__name__, "INFO",
                             f"section did not run: {exc}")]))

        # FRICTION — computed from the executed trade tape. This replaces the
        # hand-entered placeholder that stood here while the module was
        # pending: that verdict was declared, this one is measured. The tape
        # is the only thing any section needs beyond `market`, and friction
        # loads it itself when not passed one.
        try:
            import sections as friction_mod
            res = friction_mod.friction(registry_entry, blotter, market, tape)
            if res.verdict == "SUSPECT" and not res.conditions:
                res.conditions = friction_mod.conditions_for(res)
            sections.append(res)
            fragments["friction"] = findings_table(res)
        except Exception as exc:                          # noqa: BLE001
            sections.append(SectionResult(
                "friction", "INFO",
                [Finding("friction_error", type(exc).__name__, "INFO",
                         f"section did not run: {exc}")]))

        conditions: list[str] = []
        for s in sections:
            conditions.extend(s.conditions or _fallback_conditions(s))

        rep = Report(submission=sub, sections=sections, fragments=fragments)
        if rep.verdict != "PASS":
            rep.conditions = conditions

        if out_dir is not None:
            write_report_html(rep, registry_entry, blotter, market,
                              fragments, out_dir)
    return rep


if __name__ == "__main__":
    # The whole pipeline over all seven, unchanged per submission, printing
    # the scoreboard and then each report's sections, findings and conditions.
    import sys

    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root / "registry"))
    from loader import load_registry                          # noqa: E402
    from repricer import load_blotter, load_market            # noqa: E402

    _registry = load_registry(_root / "registry")
    _market = load_market(_root / "data")
    _blotters = {k: load_blotter(_root / "blotters" / f"{k}-blotter.csv")
                 for k in _registry}

    _reports = run_all(_registry, _blotters, _market, _root / "reports")
    print(grid(_reports).to_string())
    print()
    for _key in rank(_reports):
        _r = _reports[_key]
        print(f"{_key}  {_r.verdict}")
        for _sec in _r.sections:
            print(f"    {_sec.verdict:<8} {_sec.section}")
        for _c in _r.conditions:
            print(f"    CONDITION  {_c}")
        print()
