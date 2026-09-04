"""The opening slide: seven equity curves, as submitted.

    write_curves(registry, blotters, market, out_dir) -> Path

One page, seven cumulative P&L charts in a grid, ordered by claimed Sharpe
— the league table a desk would fund on. Every curve here is true: each one
reproduces its submission's claimed numbers to rounding when the blotter is
repriced at the book mid. That is the point of showing them first. The
scoreboard that follows is what happens when you ask where the money came
from, whether the test was honest, and what sample stands behind it.

Deliberate choices:

  * Priced at MID, because that is the basis the claims reproduce at, and
    the basis the book quotes. Nothing anyone would actually get filled at.
  * One hue for every curve. These are seven instances of one measure, not
    seven categories; identity comes from the panel title, so no colour has
    to carry it and no reader has to tell seven hues apart.
  * Each panel keeps its own vertical scale. Totals run from EUR 13k to
    EUR 114k, and a shared axis would flatten five of them into flat lines
    — so the totals are printed on each panel instead, and the caption says
    the scales differ.
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path

import pandas as pd

PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/4.0.0/plotly.min.js"
ACCENT, FILL, INK, MUTED, GRID = ("#2a78d6", "rgba(42,120,214,.10)",
                                  "#1a1a1a", "#8a8a85", "#ececE7")


def equity(registry_entry: dict, blotter: pd.DataFrame,
           market: dict) -> pd.Series:
    """Cumulative P&L at mid over the declared backtest window."""
    from repricer import reprice_mid, daily_pnl

    b = reprice_mid(blotter, market)
    win = (str(registry_entry["backtest_window"]["start"]),
           str(registry_entry["backtest_window"]["end"]))
    return daily_pnl(b, "mid_price", window=win).cumsum()


def _panel(key: str, entry: dict, eq: pd.Series) -> dict:
    return {
        "data": [{
            "type": "scatter", "mode": "lines",
            "x": [d.strftime("%Y-%m-%d") for d in eq.index],
            "y": [round(float(v), 1) for v in eq.values],
            "line": {"color": ACCENT, "width": 2},
            "fill": "tozeroy", "fillcolor": FILL,
            "hovertemplate": "%{x}<br>€%{y:,.0f}<extra></extra>",
        }],
        "layout": {
            "paper_bgcolor": "#fff", "plot_bgcolor": "#fff",
            "font": {"family": "system-ui,-apple-system,'Segoe UI',sans-serif",
                     "size": 11, "color": INK},
            "margin": {"l": 46, "r": 12, "t": 6, "b": 26},
            "height": 150, "showlegend": False,
            "xaxis": {"gridcolor": "rgba(0,0,0,0)", "linecolor": GRID,
                      "tickfont": {"size": 10, "color": MUTED}, "nticks": 4,
                      "ticks": "outside", "tickcolor": GRID},
            "yaxis": {"gridcolor": GRID, "zeroline": True,
                      "zerolinecolor": GRID, "linecolor": "rgba(0,0,0,0)",
                      "tickfont": {"size": 10, "color": MUTED},
                      "tickformat": "~s", "nticks": 4},
        },
    }


_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#fff;color:#1a1a1a;
font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:34px 24px 64px}
h1{font-size:21px;font-weight:660;margin:0 0 3px;letter-spacing:-.01em}
.sub{color:#8a8a85;font-size:12.5px;margin:0 0 4px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
gap:22px 26px;margin:26px 0 0}
.card{border:1px solid #ececE7;padding:14px 16px 8px}
.name{font-size:14px;font-weight:650;margin:0}
.cls{color:#8a8a85;font-size:11.5px;margin:1px 0 8px}
.stats{display:flex;gap:18px;margin:0 0 6px}
.stat b{display:block;font-size:15px;font-weight:650;
font-variant-numeric:tabular-nums}
.stat span{color:#8a8a85;font-size:10.5px}
.foot{color:#8a8a85;font-size:11.5px;margin:26px 0 0;max-width:80ch}
.foot a{color:#2a78d6}
"""


def build_html(rows: list[dict]) -> str:
    cards = ""
    for r in rows:
        cards += f"""
    <div class="card">
      <p class="name">{_html.escape(r['name'])}</p>
      <p class="cls">{_html.escape(r['cls'])}</p>
      <div class="stats">
        <div class="stat"><b>€{r['pnl']:,.0f}</b><span>total P&amp;L</span></div>
        <div class="stat"><b>{r['sharpe']:.2f}</b><span>Sharpe</span></div>
        <div class="stat"><b>{r['days']:,}</b><span>days</span></div>
      </div>
      <div id="c_{_html.escape(r['key'])}"></div>
    </div>"""

    figs = {f"c_{r['key']}": r["fig"] for r in rows}
    total = sum(r["pnl"] for r in rows)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seven strategies, as submitted</title>
<script src="{PLOTLY}"></script>
<style>{_CSS}</style></head><body><div class="wrap">

<h1>Seven strategies, as submitted</h1>
<p class="sub">Cumulative P&amp;L priced at the book mid, over each declared
  backtest window &middot; €{total:,.0f} between them &middot; ordered by
  claimed Sharpe</p>

<div class="grid">{cards}</div>

<p class="foot">Every one of these curves is true. Each reproduces its
  submission's claimed P&amp;L and Sharpe to rounding when the blotter is
  repriced at the book mid — including the broken ones. Mid is not a price
  anyone quoted: it is (bid + ask) / 2, and nothing in the market data says
  you get filled there. Each panel keeps its own vertical scale.
  &rarr; <a href="overview.html">The validation scoreboard</a></p>

<script>
const F = {json.dumps(figs)};
const cfg = {{responsive:true, displayModeBar:false}};
for (const [id, f] of Object.entries(F)) Plotly.newPlot(id, f.data, f.layout, cfg);
</script>
</div></body></html>"""


def write_curves(registry: dict, blotters: dict, market: dict,
                 out_dir: str | Path = "reports") -> Path:
    from repricer import sharpe as _sharpe

    rows = []
    for key, entry in registry.items():
        eq = equity(entry, blotters[key], market)
        daily = eq.diff().fillna(eq.iloc[0] if len(eq) else 0.0)
        rows.append({
            "key": key,
            "name": str(entry.get("name", key)),
            "cls": str(entry.get("class", "")),
            "pnl": float(eq.iloc[-1]) if len(eq) else 0.0,
            "sharpe": _sharpe(daily),
            "days": int(entry["backtest_window"]["days"]),
            "fig": _panel(key, entry, eq),
        })
    rows.sort(key=lambda r: -r["sharpe"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "curves.html"
    path.write_text(build_html(rows), encoding="utf-8")
    return path


def main(root: str | Path = "..") -> None:
    import sys
    root = Path(root)
    sys.path.insert(0, str(root / "registry"))
    from loader import load_registry
    from repricer import load_market, load_blotter

    registry = load_registry(root / "registry")
    market = load_market(root / "data")
    blotters = {k: load_blotter(root / "blotters" / f"{k}-blotter.csv")
                for k in registry}
    print("wrote", write_curves(registry, blotters, market, root / "reports"))


if __name__ == "__main__":
    main(Path(__file__).resolve().parent.parent)
