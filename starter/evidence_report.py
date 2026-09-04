"""EVIDENCE — one page per submission, one figure, one table.

    res = run(registry_entry, blotter, market)      # contract-shaped
    res = section(registry_entry, blotter, product_pnl, daily)   # pre-priced

The page considers a single strategy in isolation. No cohort, no comparison,
no other submission named: the question is what THIS strategy's own records
support, and a comparison would only invite reading a verdict off the
neighbours.

THE FIGURE
----------
The same annualised Sharpe, read four ways, each with the interval its own
assumptions justify:

  claimed                        what was submitted
  every product bet independent  rebuilt from the records, counted the way
                                 a blotter invites you to count it
  autocorrelation-adjusted       annualised by the Lo (2002) factor instead
                                 of sqrt(365), and counted per delivery day
  effective-sample-adjusted      the interval the sample that actually
                                 exists can carry

The point estimate moves once — the annualisation factor. The interval
widens twice — the denominator. Those are different corrections and the
figure keeps them visibly separate.

All prices are MID, which the pack is explicit is not a price anyone quoted.
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evidence as E
from evidence import Weight
from report import Finding, SectionResult

PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/4.0.0/plotly.min.js"

ACCENT, MUTED, INK, GRID = "#2a78d6", "#8a8a85", "#1a1a1a", "#e6e6e1"
BADGE = {"PASS": "#0f8a3c", "SUSPECT": "#b8860b", "FAIL": "#c22f2f",
         "INFO": "#8a8a85"}


def figure(w: Weight, claimed: float | None) -> dict:
    """One strategy's annualised Sharpe as each correction is applied."""
    rows = E.deflation(w, claimed)[::-1]      # first rung at the top
    ys = list(range(len(rows)))
    labels = [r["label"] for r in rows]

    withci = [(i, r) for i, r in enumerate(rows) if r["lo"] is not None]
    plain = [(i, r) for i, r in enumerate(rows) if r["lo"] is None]

    data = []
    if withci:
        data.append({
            "type": "scatter", "mode": "markers", "name": "95% interval",
            "x": [r["sharpe"] for _, r in withci],
            "y": [i for i, _ in withci],
            "text": [r["note"] for _, r in withci],
            "marker": {"size": 9, "color": ACCENT},
            "error_x": {"type": "data", "symmetric": False,
                        "array": [r["hi"] - r["sharpe"] for _, r in withci],
                        "arrayminus": [r["sharpe"] - r["lo"] for _, r in withci],
                        "color": ACCENT, "thickness": 7, "width": 0},
            "customdata": [[r["lo"], r["hi"]] for _, r in withci],
            "hovertemplate": ("Sharpe %{x:.2f}"
                              "<br>95%% CI %{customdata[0]:.2f} to "
                              "%{customdata[1]:.2f}"
                              "<br>%{text}<extra></extra>"),
        })
    if plain:
        data.append({
            "type": "scatter", "mode": "markers", "name": "point estimate only",
            "x": [r["sharpe"] for _, r in plain],
            "y": [i for i, _ in plain],
            "text": [r["note"] for _, r in plain],
            "marker": {"size": 10, "color": "#ffffff", "symbol": "circle",
                       "line": {"color": MUTED, "width": 2}},
            "hovertemplate": "Sharpe %{x:.2f}<br>%{text}<extra></extra>",
        })

    xs = [v for r in rows for v in (r["lo"], r["hi"], r["sharpe"])
          if v is not None and np.isfinite(v)]
    pad = max(0.6, (max(xs) - min(xs)) * 0.08)

    return {"data": data, "layout": {
        "paper_bgcolor": "#ffffff", "plot_bgcolor": "#ffffff",
        "font": {"family": "system-ui,-apple-system,Segoe UI,sans-serif",
                 "size": 12, "color": INK},
        "margin": {"l": 214, "r": 30, "t": 10, "b": 52},
        "height": 236, "showlegend": False,
        "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": -0.6,
                    "y1": len(rows) - 0.4, "xref": "x", "yref": "y",
                    "line": {"color": MUTED, "width": 1, "dash": "dot"}}],
        "xaxis": {"title": {"text": "annualised Sharpe (at mid)",
                            "font": {"size": 11}},
                  "gridcolor": GRID, "zeroline": False, "linecolor": GRID,
                  "ticks": "outside", "tickcolor": GRID,
                  "tickfont": {"size": 11},
                  "range": [min(min(xs), 0) - pad, max(xs) + pad]},
        "yaxis": {"gridcolor": "rgba(0,0,0,0)", "zeroline": False,
                  "linecolor": GRID, "automargin": True,
                  "tickfont": {"size": 12}, "tickmode": "array",
                  "tickvals": ys, "ticktext": labels,
                  "range": [-0.6, len(rows) - 0.4]},
    }}


def _rows(w: Weight) -> list[tuple[str, str, bool]]:
    """(metric, value, flagged) for the diagnostics table."""
    frag = w.drop_to_t2 is not None and w.drop_to_t2 < E.DROP_FRAGILE
    return [
        ("trades", f"{w.trades:,}", False),
        ("round trips", f"{w.round_trips:,}", False),
        ("product bets", f"{w.product_bets:,}", False),
        ("active delivery days",
         f"{w.delivery_days:,} of {w.calendar_days:,}", False),
        ("effective observations", f"{w.eff_days:.0f}",
         w.eff_days < E.MIN_EFF_DAYS),
        ("trades per active day", f"{w.trades_per_day:.0f}", False),
        ("products per delivery day", f"{w.products_per_day:.1f}", False),
        ("within-day ICC", f"{w.icc:+.2f}", False),
        ("design effect", f"×{w.design_effect:.2f}",
         w.design_effect >= E.DEFF_WARN),
        ("lag-1 autocorrelation", f"{w.rho[0]:+.3f}", False),
        ("Lo annualisation factor",
         f"{w.eta:.2f} (√{E.PERIODS} = {np.sqrt(E.PERIODS):.2f})", False),
        ("skew / kurtosis", f"{w.skew:+.2f} / {w.kurtosis:.1f}", False),
        ("t, per product bet", f"{w.t_product:.1f}", False),
        ("t, per delivery day", f"{w.t_day:.1f}", False),
        ("best days to push t below 2",
         f"{w.drop_to_t2}" if w.drop_to_t2 is not None
         else f"more than {E.DROP_MAX}", frag),
        ("Sharpe, bootstrapped per delivery day",
         f"{w.sharpe:.2f} &nbsp;[{w.ci_lo:.2f}, {w.ci_hi:.2f}]",
         w.ci_lo < E.SR_CI_FLOOR),
    ]


def _table(w: Weight) -> str:
    body = "".join(
        f'<tr><td>{_html.escape(m)}</td>'
        f'<td class="{"num hot" if flag else "num"}">{v}'
        f'{"&thinsp;▲" if flag else ""}</td></tr>'
        for m, v, flag in _rows(w))
    return f"""<table><tbody>{body}</tbody></table>
    <p class="foot">▲ past a threshold fixed before any submission was seen:
      under {E.MIN_EFF_DAYS:.0f} effective observations, design effect
      ≥&thinsp;{E.DEFF_WARN:.1f}, fewer than {E.DROP_FRAGILE} best days needed
      to push t below {E.T_FRAGILE:.0f}, or a bootstrap lower bound under
      {E.SR_CI_FLOOR:.1f}.</p>"""


_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#fff;color:#1a1a1a;
font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:32px 24px 56px}
h1{font-size:19px;font-weight:650;margin:0 0 2px}
.badge{display:inline-block;font-size:11px;font-weight:650;color:#fff;
padding:2px 9px;border-radius:3px;vertical-align:3px;margin-left:6px}
.sub{color:#8a8a85;font-size:12.5px;margin:0 0 26px}
h2{font-size:13px;font-weight:650;margin:0 0 1px}
.cap{color:#8a8a85;font-size:12px;margin:0 0 8px}
section{margin:0 0 28px;padding:0 0 24px;border-bottom:1px solid #ececE7}
section:last-of-type{border-bottom:0;padding-bottom:0}
table{border-collapse:collapse;width:100%;font-size:12.5px}
td{padding:6px 8px;border-bottom:1px solid #ececE7;text-align:right}
td:first-child{text-align:left;color:#52514e}
td.num{font-variant-numeric:tabular-nums;font-weight:600}
td.hot{color:#c22f2f}
.foot{color:#8a8a85;font-size:11.5px;margin:10px 0 0}
"""


def build_html(w: Weight, registry_entry: dict, verdict: str) -> str:
    claimed = (registry_entry.get("claimed") or {}).get("sharpe_ann")
    name = _html.escape(str(registry_entry.get("name", w.submission)))
    fig = figure(w, claimed)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html.escape(w.submission)} &mdash; evidence</title>
<script src="{PLOTLY}"></script>
<style>{_CSS}</style></head><body><div class="wrap">

<h1>{name}
  <span class="badge" style="background:{BADGE[verdict]}">{verdict}</span></h1>
<p class="sub">Evidence &middot; {w.eff_days:.0f} effective observations from
  {w.trades:,} trades over {w.calendar_days:,} calendar days &middot; at mid</p>

<section>
  <h2>Annualised Sharpe, adjusted</h2>
  <p class="cap">Same trades, four readings. The estimate moves with the
    annualisation factor; the interval widens with the sample.</p>
  <div id="fig"></div>
</section>

<section>{_table(w)}</section>

<script>
const cfg = {{responsive:true, displayModeBar:false}};
const F = {json.dumps(fig)};
Plotly.newPlot("fig", F.data, F.layout, cfg);
</script>
</div></body></html>"""


def write_html(w: Weight, registry_entry: dict, verdict: str,
               out_dir: str | Path = "reports") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{w.submission}-evidence.html"
    path.write_text(build_html(w, registry_entry, verdict), encoding="utf-8")
    return path


def fragment(w: Weight, registry_entry: dict, div_id: str = "ev_fig") -> str:
    """Body HTML for this section inside the assembled report.

    No page shell, no stylesheet: it uses the classes report.py defines, so
    the six sections read as one document rather than six pasted pages. The
    inline <script> is safe because the assembled page loads Plotly in
    <head>.
    """
    claimed = (registry_entry.get("claimed") or {}).get("sharpe_ann")
    fig = figure(w, claimed)
    return f"""
    <p class="cap">{w.eff_days:.0f} effective observations from
      {w.trades:,} trades over {w.calendar_days:,} calendar days. Same trades,
      four readings: the estimate moves with the annualisation factor, the
      interval widens with the sample.</p>
    <div id="{div_id}"></div>
    {_table(w)}
    <script>Plotly.newPlot({div_id!r}, {json.dumps(fig["data"])},
      {json.dumps(fig["layout"])}, {{responsive:true, displayModeBar:false}});
    </script>"""


def contribute(registry_entry: dict, blotter: pd.DataFrame, market: dict,
               out_dir: str | Path | None = "reports"
               ) -> tuple[SectionResult, str]:
    """report.py's section protocol: one measurement, verdict + body HTML.

    measure() runs 5,000 bootstrap resamples, so the SectionResult and the
    figure are built from a single Weight rather than measured twice.
    """
    from repricer import reprice_mid, product_pnl, daily_pnl

    b = reprice_mid(blotter, market)
    win = (str(registry_entry["backtest_window"]["start"]),
           str(registry_entry["backtest_window"]["end"]))
    w = E.measure(str(registry_entry.get("submission", "?")), blotter,
                  product_pnl(b, "mid_price"),
                  daily_pnl(b, "mid_price", window=win))
    return E.evidence_from(w), fragment(w, registry_entry)


def run(registry_entry: dict, blotter: pd.DataFrame, market: dict,
        out_dir: str | Path | None = "reports") -> SectionResult:
    """Contract-shaped entry: the same inputs report.py's run() receives.

    Everything this section needs is derived here — the caller hands over
    the raw blotter and the market tables and nothing else. Repricing is at
    MID, which is what the claimed numbers reproduce at; when the crossing
    repricer exists this is the one line that changes.
    """
    from repricer import reprice_mid, product_pnl, daily_pnl

    b = reprice_mid(blotter, market)
    win = (str(registry_entry["backtest_window"]["start"]),
           str(registry_entry["backtest_window"]["end"]))
    return section(registry_entry, blotter,
                   product_pnl(b, "mid_price"),
                   daily_pnl(b, "mid_price", window=win), out_dir)


def section(registry_entry: dict, blotter: pd.DataFrame,
            product_pnl: pd.Series, daily: pd.Series,
            out_dir: str | Path | None = "reports") -> SectionResult:
    """The SectionResult report.py expects, with the page as its evidence.

    Use this when the P&L series are already built; use run() when starting
    from the raw market tables.
    """
    w = E.measure(str(registry_entry.get("submission", "?")), blotter,
                  product_pnl, daily)
    res = E.evidence_from(w)
    if out_dir is not None:
        page = write_html(w, registry_entry, res.verdict, out_dir)
        res.findings.append(Finding("evidence_page", str(page), "INFO",
                                    "sample-weight figure and diagnostics"))
    return res


def main(root: str | Path = "..") -> None:
    import sys
    root = Path(root)
    sys.path.insert(0, str(root / "registry"))
    from loader import load_registry
    from repricer import load_market, load_blotter

    registry = load_registry(root / "registry")
    market = load_market(root / "data")

    for key in sorted(registry):
        raw = load_blotter(root / "blotters" / f"{key}-blotter.csv")
        res = run(registry[key], raw, market, root / "reports")
        print(f"{key:14s} {res.verdict:8s} {key}-evidence.html")


if __name__ == "__main__":
    main(Path(__file__).resolve().parent.parent)
