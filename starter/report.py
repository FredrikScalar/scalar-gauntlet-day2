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
"""


def _badge(v: str) -> str:
    return f'<span class="badge" style="background:{BADGE.get(v, "#8a8a85")}">{v}</span>'


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


def render_html(rep: "Report", registry_entry: dict, blotter: pd.DataFrame,
                market: dict, fragments: dict[str, str]) -> str:
    """The whole validation report for one submission, as one page."""
    name = html.escape(str(registry_entry.get("name", rep.submission)))
    cls = html.escape(str(registry_entry.get("class", "")))
    by = {s.section: s for s in rep.sections}

    body = ""
    for i, key in enumerate(SECTIONS, start=1):
        title, question = SECTION_TITLES[key]
        res = by.get(key)
        todo = "" if res else " todo"
        inner = (fragments.get(key) or _generic_fragment(res)) if res \
            else _placeholder(key)
        body += f"""
  <section class="sec{todo}" id="{key}">
    <div class="sechead"><span class="n">{i}</span>
      <h2>{html.escape(title)}</h2>
      {_badge(res.verdict) if res else _badge("·")}</div>
    <p class="q">{html.escape(question)}</p>
    {inner}
  </section>"""

    conds = ""
    if rep.conditions:
        items = "".join(f"<li>{html.escape(c)}</li>" for c in rep.conditions)
        conds = (f'<div class="conds"><h3>Conditions attached</h3>'
                 f'<ol>{items}</ol></div>')

    built = sum(1 for k in SECTIONS if k in by)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(rep.submission)} &mdash; validation report</title>
<script src="{PLOTLY}"></script>
<style>{_CSS}</style></head><body><div class="wrap">

<h1>{name}{_badge(rep.verdict)}</h1>
<p class="sub">Validation report{f" &middot; {cls}" if cls else ""}
  &middot; {built} of {len(SECTIONS)} sections built &middot; priced at mid</p>
{conds}
{body}
</div></body></html>"""


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
        out_dir: str | Path | None = "reports") -> Report:
    """Take one submission's records, return the verdict with evidence.

    One submission in, one Report out, with each section's HTML page written
    to `out_dir` (pass None to compute verdicts without writing pages).

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

        conditions: list[str] = []
        for s in sections:
            conditions.extend(s.conditions or _fallback_conditions(s))

        rep = Report(submission=sub, sections=sections)
        if rep.verdict != "PASS":
            rep.conditions = conditions

        if out_dir is not None:
            write_report_html(rep, registry_entry, blotter, market,
                              fragments, out_dir)
    return rep
