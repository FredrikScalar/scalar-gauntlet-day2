"""LINEAGE — did the strategy trade on information that didn't exist yet?

One page per submission: PASS or FAIL, one sentence, one table.

    path = write_html(sub, entry, blotter, forecasts)
    res  = section(sub, entry, blotter, forecasts)   # SectionResult for report.py

THE TEST, IN PLAIN TERMS
------------------------
Wind, solar and load forecasts are reissued through the day. For every trade
that OPENS a position we find two updates: the newest one published at or
before the trade, and the first one published after it. The first was
available to an honest strategy. The second was not.

Then we ask which of the two the strategy's buy/sell decisions line up with,
scored 0 (no relationship) to 1 (moves in lockstep). An honest strategy
scores high on the published update and ~0 on the later one.

Both scores come from one regression that includes both updates together,
not two separate ones. That matters: weather runs have momentum, so
consecutive updates resemble each other, and a strategy honestly trading the
published update would score misleadingly high on the later one if you
measured them separately.

Closing trades are excluded — they mirror the entry and would cancel the
signal out.

PAGE VERDICT vs SECTION VERDICT
-------------------------------
The page speaks `report.py`'s three verdicts. `lineage.py` returns a fourth,
INFO, for a test that could not run; the page shows that as SUSPECT:

    FAIL     -> FAIL      the test detected a loading
    SUSPECT  -> SUSPECT   it detected something short of the fail threshold
    PASS     -> PASS      it ran and found nothing
    INFO     -> SUSPECT   it could not run, so nothing was ruled out

A SUSPECT must carry conditions or it means nothing (`report.py`), so
`conditions()` returns them and the page prints them. `section()` still
passes the raw four-verdict result through untouched — Evidence needs to
tell a pass that was earned from one that was never tested.
"""
from __future__ import annotations

import html as _html
from pathlib import Path

import pandas as pd

import lineage as L
import participation as P
from report import Finding, SectionResult

VERDICT_ORDER = ["FAIL", "SUSPECT", "PASS", "INFO"]


def page_verdict(verdict: str) -> str:
    """lineage.py's four verdicts in report.py's three. See module docstring."""
    return "SUSPECT" if verdict == "INFO" else verdict


def conditions(cs: list[dict]) -> list[str]:
    """What a SUSPECT verdict must carry to mean anything (`report.py`)."""
    out = []
    untested = [c for c in cs if c["untestable"]]
    if untested:
        reasons = sorted({c["short"] for c in untested})
        out.append(
            f"The revision test could not run ({'; '.join(reasons)}), so "
            f"leakage is unruled-out rather than absent. Do not fund on this "
            f"section alone: clear it with a test that does not need a "
            f"buy-versus-sell split, and require the declared data feeds to "
            f"be produced for inspection.")
    flagged = [c for c in cs
               if not c["untestable"] and c["verdict"] == "SUSPECT"]
    for c in flagged:
        out.append(
            f"Its trades show a partial loading on the unpublished "
            f"{c['variable']} update ({c['tracks_unpublished']:.2f}, "
            f"t={c['t_unpublished']:+.1f}) — below the fail threshold but "
            f"not clean. Re-run on the next six months before capital.")
    return out


def _why_untestable(blotter: pd.DataFrame, forecasts: pd.DataFrame,
                    c: dict) -> tuple[str, str]:
    """What actually stopped the test: (short tag, full sentence).

    `lineage.py` reports the mechanical cause ("a regressor has zero
    variance"), which is true but tells a reader nothing about the strategy.
    Which end of the forecast timeline is missing decides the wording, so
    read it off the data rather than inferring it from the trade times.
    """
    marked = L.mark_open_close(blotter)
    opens = marked[marked["is_open"]]
    sides = set(opens["side"])
    if len(sides) == 1:
        side = next(iter(sides)).lower()
        return ("one-sided book",
                f"every position it opens is a {side}, so there is no "
                f"buy-versus-sell decision for the test to explain")
    if c["n"] == 0 or c["coverage"] < L.MIN_COVERAGE:
        med = ((opens["product_delivery"] - opens["exec_ts"])
               .dt.total_seconds().median() / 60)
        d = L.attach_revisions(opens, forecasts, c["variable"])
        if d["past_revision"].isna().mean() >= d["future_revision"].isna().mean():
            return ("trades before the first forecast",
                    f"it opens its positions about {med:.0f} minutes before "
                    f"delivery, at the first {c['variable']} forecast of the "
                    f"day — so there is no earlier update to measure a "
                    f"change against")
        return ("trades after the last forecast",
                f"it opens its positions about {med:.0f} minutes before "
                f"delivery, after the last {c['variable']} forecast update — "
                f"so there is no later update to compare against")
    return ("not testable", c["untestable"])


def collect(blotter: pd.DataFrame, forecasts: pd.DataFrame,
            products: pd.DataFrame, variable: str) -> dict:
    """Run the direction test; where it cannot speak, fall back.

    Direction is the primary test: it keys on the SIGN of news, which is
    unpredictable, so its null is clean. Participation keys on magnitude and
    is only identified where the strategy has volatility history — it is a
    fallback for one-sided books, never a second opinion on a book the
    direction test already settled.
    """
    marked = L.mark_open_close(blotter)
    r = L.leakage_test(blotter, forecasts, variable)
    out = {"variable": variable, "verdict": L._verdict(r),
           "n": r["n"], "coverage": r["coverage"],
           "n_opens": int(marked["is_open"].sum()),
           "method": "direction",
           "untestable": r.get("untestable", ""), "short": ""}

    if not out["untestable"]:
        # |standardised coefficient|: 0 = no relationship, 1 = lockstep.
        out["tracks_published"] = abs(r["beta_past"])
        out["tracks_unpublished"] = abs(r["beta_future"])
        out["t_unpublished"] = r["t_future"]
        return out

    why_direction = _why_untestable(blotter, forecasts, out)[0]
    pr = P.participation_test(blotter, forecasts, products, variable)
    if "untestable" not in pr:
        out.update({"method": "participation", "untestable": "", "short": "",
                    "verdict": P._verdict(pr), "n": pr["n"],
                    "tracks_published": float("nan"),
                    "tracks_unpublished": abs(pr["beta_next"]),
                    "t_unpublished": pr["t_next"],
                    "controls": pr["controls"]})
        return out

    out["short"] = why_direction
    out["untestable"] = (
        f"{_why_untestable(blotter, forecasts, out)[1]}; and the fallback "
        f"test on which hours it chose to trade is not identified either "
        f"\u2014 {pr['untestable']}, so its volatility regime cannot be "
        f"controlled for")
    return out


def reading(c: dict) -> str:
    """The one sentence that explains the verdict."""
    v = c["variable"]
    if c["untestable"]:
        return (f"The test could not run on any of the three forecasts — on "
                f"{v}, {c['untestable']}. Nothing was found against it, but "
                f"nothing was ruled out either.")
    pub, unpub = c["tracks_published"], c["tracks_unpublished"]
    if c["method"] == "participation":
        which = ("It never varies its side or its size, so the test asks "
                 "which delivery hours it chose instead. ")
        if page_verdict(c["verdict"]) == "PASS":
            return (which + f"That choice shows no relationship to {v} news "
                    f"published only after it decided ({unpub:.2f}, "
                    f"t={c['t_unpublished']:+.1f}), controlling for the "
                    f"forecast level and {c['controls']} prior revisions.")
        return (which + f"That choice loads {unpub:.2f} (t="
                f"{c['t_unpublished']:+.1f}) on {v} news published only "
                f"after it decided.")
    if page_verdict(c["verdict"]) == "PASS":
        near = abs(c["t_unpublished"]) >= 0.9 * L.T_SUSPECT
        edge = " It passes, but only just." if near else ""
        return (f"Its trades follow the {v} forecast update that was already "
                f"published ({pub:.2f}) and essentially ignore the one that "
                f"had not happened yet ({unpub:.2f}). That is what an honest "
                f"strategy looks like.{edge}")
    return (f"Its trades follow the {v} forecast update that had NOT been "
            f"published yet ({unpub:.2f} out of 1.00) and almost ignore the "
            f"one that had ({pub:.2f}). Nobody could have known that number "
            f"when these trades were placed.")


_CSS = """
html{background:#fff}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:#000;
font:15px/1.55 system-ui,-apple-system,"Segoe UI",Arial,sans-serif}
.wrap{max-width:620px;margin:0 auto;padding:48px 24px 72px}
h1{font-size:20px;font-weight:700;margin:0 0 4px}
.sub{color:#555;font-size:13px;margin:0 0 20px}
hr{border:0;border-top:1px solid #000;margin:0 0 20px}
.pass{color:#137a13}.fail{color:#b00000}.suspect{color:#8a6100}
.cond{font-size:13.5px;margin:0 0 26px;max-width:60ch;
padding:12px 14px;border:1px solid #ddd;background:#fafafa}
.cond b{display:block;margin-bottom:3px}
.q{font-size:13px;color:#555;margin:0 0 4px}
.a{font-size:24px;font-weight:700;margin:0 0 10px}
.why{font-size:15px;margin:0 0 8px;max-width:60ch}
.n{color:#555;font-size:13px;margin:0 0 28px;max-width:60ch}
table{border-collapse:collapse;width:100%;margin:0;font-size:14px}
caption{text-align:left;color:#555;font-size:13px;padding:0 0 8px}
th,td{padding:8px 10px;border-bottom:1px solid #ddd;text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:#555;font-weight:600;font-size:13px;border-bottom:1px solid #000}
td.num{font-variant-numeric:tabular-nums}
td.hit{font-weight:700;color:#b00000}
td.v{font-weight:700}
td.skip{color:#555}
.note{color:#555;font-size:13px;max-width:62ch;margin:30px 0 0;
padding-top:14px;border-top:1px solid #ddd}
.note b{color:#000;font-weight:600}
"""


# --- plotly, white ----------------------------------------------------------
BAR = "#2a78d6"          # the score, when it is where it should be
BAR_FAIL = "#b00000"     # the score, when it is not
FAIL_LEVEL = L.BETA_FAIL  # 0.10 — necessary (not sufficient) for a FAIL


def _bar(labels: list[str], values: list[float], fails: list[bool],
         notes: list[str], height: int, first: bool) -> str:
    """One horizontal bar chart of the score that decides the verdict.

    Only the not-yet-published score is plotted. The published one is not a
    verdict input — plotting both would invite reading the pair as a
    balance, which it is not.
    """
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=[BAR_FAIL if f else BAR for f in fails],
        text=[n if n else f"{v:.2f}" for v, n in zip(values, notes)],
        textposition="outside", cliponaxis=False,
        textfont=dict(size=12, color="#000"),
        hovertemplate="%{y}<br>score %{x:.2f}<extra></extra>",
    ))
    # 0.10 is the size a loading must reach to fail (alongside t >= 6). It is
    # a reference, not the whole rule, and is labelled as such.
    fig.add_vline(x=FAIL_LEVEL, line_width=1, line_color="#999")
    fig.add_annotation(x=FAIL_LEVEL, y=1, yref="paper", yanchor="bottom",
                       text=f"{FAIL_LEVEL:.2f} fail level", showarrow=False,
                       font=dict(size=11, color="#777"), xshift=2)
    fig.update_layout(
        template="simple_white", height=height,
        margin=dict(l=0, r=44, t=26, b=34),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(family='system-ui,-apple-system,"Segoe UI",Arial,sans-serif',
                  size=13, color="#000"),
        xaxis=dict(range=[0, 1.06], title="follows the update published only "
                                          "after it traded (0&#8211;1)",
                   title_font=dict(size=12, color="#555"),
                   tickfont=dict(size=11, color="#555"),
                   showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(autorange="reversed", tickfont=dict(size=13, color="#000"),
                   showgrid=False, fixedrange=True),
        showlegend=False, bargap=0.42,
    )
    return fig.to_html(full_html=False, include_plotlyjs=("cdn" if first
                                                          else False),
                       config={"displayModeBar": False,
                               "responsive": True})


def _row(c: dict) -> str:
    v = page_verdict(c["verdict"])
    name = c["variable"].title()
    if c["untestable"]:
        return (f'<tr><td>{name}</td><td class="skip" colspan="2">not tested '
                f'&mdash; {_html.escape(c["short"])}</td>'
                f'<td class="v suspect">SUSPECT</td></tr>')
    hit = ' class="num hit"' if v == "FAIL" else ' class="num"'
    pub = ('<td class="skip">n/a</td>' if c["method"] == "participation"
           else f'<td class="num">{c["tracks_published"]:.2f}</td>')
    return (f'<tr><td>{name}</td>{pub}'
            f'<td{hit}>{c["tracks_unpublished"]:.2f}</td>'
            f'<td class="v {v.lower()}">{v}</td></tr>')


def build_html(submission: str, registry_entry: dict, blotter: pd.DataFrame,
               forecasts: pd.DataFrame, products: pd.DataFrame) -> str:
    cs = [collect(blotter, forecasts, products, v) for v in L.VARIABLES]
    overall = page_verdict(next(v for v in VERDICT_ORDER
                                if v in [c["verdict"] for c in cs]))
    # the forecast that drives the verdict: worst first, then strongest evidence
    driver = min(cs, key=lambda c: (VERDICT_ORDER.index(c["verdict"]),
                                    -abs(c.get("t_unpublished", 0))))

    answer = "Yes." if overall == "FAIL" else "No."
    if driver["untestable"]:
        answer = "We cannot tell."
    conds = conditions(cs)
    conds_html = "" if not conds else (
        '<p class="cond"><b>Conditions attached to this verdict.</b> '
        + " ".join(_html.escape(c) for c in conds) + "</p>")

    # State the excluded trades and why, rather than a bare coverage figure —
    # a lone "22% could be tested" reads as a weakness when the exclusion is
    # structural (those trades open before any earlier update exists).
    if driver["untestable"]:
        tested = ""
    elif driver["method"] == "participation":
        # counts products, not trades — the untraded hours are the controls
        tested = (f"Measured across all {driver['n']:,} delivery products in "
                  f"the market: the {driver['n_opens']:,} it opened a position "
                  f"in, and the rest it left alone.")
    else:
        # State the excluded trades and why, rather than a bare coverage
        # figure — a lone "22% could be tested" reads as a weakness when the
        # exclusion is structural.
        skipped = driver["n_opens"] - driver["n"]
        tested = (f"Measured on {driver['n']:,} of its {driver['n_opens']:,} "
                  f"position-opening trades." + (
                      f" The other {skipped:,} open at the day's first "
                      f"{driver['variable']} forecast, with no earlier update "
                      f"to compare against." if skipped > 0 else ""))

    name = _html.escape(str(registry_entry.get("name", submission)))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html.escape(submission)} — leakage check</title>
<style>{_CSS}</style></head><body><div class="wrap">

<h1>{name} &mdash; <span class="{overall.lower()}">{overall}</span></h1>
<p class="sub">Lineage &middot; leakage check &middot; {len(blotter):,} trades</p>
<hr>

<p class="q">Did this strategy trade on information that did not exist yet?</p>
<p class="a {overall.lower()}">{answer}</p>
<p class="why">{_html.escape(reading(driver))}</p>
<p class="n">{tested}</p>
{conds_html}

{_bar([c["variable"].title() for c in cs],
       [0.0 if c["untestable"] else c["tracks_unpublished"] for c in cs],
       [page_verdict(c["verdict"]) == "FAIL" for c in cs],
       ["  not tested" if c["untestable"] else "" for c in cs],
       height=210, first=True)}

<table>
  <caption>How closely its trades follow each forecast update (0 = not at
    all, 1 = in lockstep)</caption>
  <thead><tr><th>forecast</th><th>already published<br>when it traded</th>
    <th>not published until<br>after it traded</th><th></th></tr></thead>
  <tbody>{"".join(_row(c) for c in cs)}</tbody>
</table>

<p class="note"><b>How to read it.</b> The right-hand column is the one that
matters: it should be near zero. A strategy cannot follow an update that had
not been published when it traded unless something gave it that information
early. The left column can be anything — following the published forecast is
just doing the job. Both numbers come from a single test that weighs the two
updates against each other, because consecutive forecast updates resemble
one another and measuring them separately would flag honest strategies.
Thresholds were fixed before any submission was looked at and are the same
for all seven. A row marked <i>not tested</i> is SUSPECT rather than PASS:
the test could not run, so it found nothing wrong and also ruled nothing
out.</p>
</div></body></html>"""


def write_html(submission: str, registry_entry: dict, blotter: pd.DataFrame,
               forecasts: pd.DataFrame, products: pd.DataFrame,
               out_dir: str | Path = "reports") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{submission}-leakage.html"
    path.write_text(build_html(submission, registry_entry, blotter, forecasts,
                               products), encoding="utf-8")
    return path


_INDEX_CSS = _CSS + """
.wrap{max-width:840px}
td.name a{color:#000;text-decoration:none;border-bottom:1px solid #bbb}
td.name a:hover{border-bottom-color:#000}
tfoot td{border-bottom:0;padding-top:14px;color:#555;font-size:13px}
"""


def build_index(rows: list[dict]) -> str:
    """The grid: one row per submission, the number that decides each one."""
    body = []
    for r in rows:
        v = r["verdict"]
        cells = []
        for c in r["cs"]:
            if c["untestable"]:
                cells.append('<td class="skip">&mdash;</td>')
            else:
                cls = "num hit" if page_verdict(c["verdict"]) != "PASS" else "num"
                cells.append(f'<td class="{cls}">'
                             f'{c["tracks_unpublished"]:.2f}</td>')
        body.append(
            f'<tr><td class="name"><a href="{r["href"]}">{_html.escape(r["name"])}'
            f'</a></td>{"".join(cells)}'
            f'<td class="skip">{_html.escape(r["note"])}</td>'
            f'<td class="v {v.lower()}">{v}</td></tr>')

    n_fail = sum(r["verdict"] == "FAIL" for r in rows)
    n_susp = sum(r["verdict"] == "SUSPECT" for r in rows)
    n_pass = sum(r["verdict"] == "PASS" for r in rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Leakage check — all submissions</title>
<style>{_INDEX_CSS}</style></head><body><div class="wrap">

<h1>Leakage check &mdash; all submissions</h1>
<p class="sub">Lineage &middot; did each strategy trade on information that
  did not exist yet?</p>
<hr>

{_bar([r["name"] for r in rows],
       [max([0.0 if c["untestable"] else c["tracks_unpublished"]
             for c in r["cs"]]) for r in rows],
       [r["verdict"] == "FAIL" for r in rows],
       ["  not tested" if all(c["untestable"] for c in r["cs"]) else ""
        for r in rows],
       height=330, first=True)}

<table>
  <caption>How closely each strategy's trades follow the forecast update that
    had <b>not been published</b> when it traded. Should be near zero.</caption>
  <thead><tr><th>strategy</th><th>wind</th><th>solar</th><th>load</th>
    <th>note</th><th></th></tr></thead>
  <tbody>{"".join(body)}</tbody>
  <tfoot><tr><td colspan="6">{n_fail} fail &middot; {n_susp} suspect
    &middot; {n_pass} pass</td></tr></tfoot>
</table>

<p class="note"><b>How to read it.</b> Each number is how closely that
strategy's buy/sell decisions follow a forecast update published only
<i>after</i> it traded, scored 0 (no relationship) to 1 (lockstep). A
strategy cannot follow an update that did not exist unless something gave it
that information early. A dash means the test could not run on that
forecast, which is a SUSPECT rather than a pass: nothing was found, and
nothing was ruled out. Click a name for the full page.</p>
</div></body></html>"""


def write_index(rows: list[dict], out_dir: str | Path = "reports") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "index.html"
    path.write_text(build_index(rows), encoding="utf-8")
    return path


def section(submission: str, registry_entry: dict, blotter: pd.DataFrame,
            forecasts: pd.DataFrame, products: pd.DataFrame,
            out_dir: str | Path = "reports") -> SectionResult:
    """The SectionResult `report.py` expects, with the page as its evidence.

    Deliberately NOT collapsed to two verdicts: Evidence has to be able to
    tell a pass that was earned from one that was never tested.
    """
    cs = [collect(blotter, forecasts, products, v) for v in L.VARIABLES]
    order = ["FAIL", "SUSPECT", "PASS", "INFO"]
    res = L.lineage(registry_entry, blotter, forecasts)
    res.verdict = next(v for v in order
                       if v in [c["verdict"] for c in cs])
    for c in cs:
        if c["method"] == "participation":
            res.findings.append(Finding(
                f"participation_{c['variable']}",
                round(float(c["t_unpublished"]), 2), c["verdict"],
                "direction test could not run; fell back to which "
                "delivery hours it chose to trade"))
    path = write_html(submission, registry_entry, blotter, forecasts,
                      products, out_dir)
    res.findings.append(Finding("leakage_page", str(path), "INFO",
                                "one-page leakage check"))
    return res


def main(root: str | Path = "..") -> None:
    import sys
    root = Path(root)
    sys.path.insert(0, str(root / "registry"))
    from loader import load_registry

    registry = load_registry(root / "registry")
    forecasts = L.load_forecasts(root / "data")
    products = P.load_products(root / "data")
    rows = []
    for key in sorted(registry):
        blotter = pd.read_csv(root / "blotters" / f"{key}-blotter.csv",
                              parse_dates=["exec_ts", "product_delivery"])
        res = section(key, registry[key], blotter, forecasts, products,
                      root / "reports")
        cs = [collect(blotter, forecasts, products, v) for v in L.VARIABLES]
        untested = [c for c in cs if c["untestable"]]
        rows.append({
            "name": str(registry[key].get("name", key)),
            "href": f"{key}-leakage.html",
            "cs": cs,
            "verdict": page_verdict(res.verdict),
            "note": (sorted({c["short"] for c in untested})[0] if untested
                     else ""),
        })
        print(f"{key:14s} page={page_verdict(res.verdict):8s} "
              f"section={res.verdict}")
    # worst first, so the grid opens on what needs attention
    rows.sort(key=lambda r: (VERDICT_ORDER.index(r["verdict"]), r["name"]))
    print("grid ->", write_index(rows, root / "reports"))


if __name__ == "__main__":
    main(Path(__file__).resolve().parent.parent)
