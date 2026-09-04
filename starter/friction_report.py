"""Execution stress test - interactive report, one strategy at a time.

Two questions, and only these two:

    1  FILL MODE STRESS   what the same recorded trades are worth at the mid
       the blotter records, at the touch assuming a fill there, and sweeping
       the book by size - plus, when sweeping, how much of the clip the book
       can absorb, how often it runs out of depth, and what happens to
       performance as the clip is scaled.

    2  MARKET ABSORPTION  how big the clip is next to what this market
       normally trades, and whether the market moves that size inside the
       window the strategy holds the position.

Nothing about any particular submission is written in: every observation,
grid and method note is derived from whatever registry, blotters, market and
tape are supplied.

The numbers come from `friction.py` and `execution.py`, which stay purely
descriptive. Every threshold that turns them into KEEP / SUSPECT / FAIL lives
in `verdicts.py`, so the line between what the data says and where we drew
the line stays visible.

    python friction_report.py                       # -> figures/
    python friction_report.py out.html              # explicit path
    python friction_report.py out.html --inline --fragment   # publishable
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import execution as ex
import friction as fr
import sections as sc
import friction_verdicts as vd
from repricer import daily_pnl, load_blotter, load_market, sharpe

MID_C, CROSS_C, SWEEP_C = "#7c848e", "#2d5f8b", "#b3403f"
ACCENT, GOOD, WARN, GREY = "#2d5f8b", "#3f7d58", "#b3403f", "#8a8a8a"
TEMPLATE = "plotly_white"

# clip multipliers to stress. Below 1 as well as above, because the question
# is not only "can it get bigger" but "is the current size already too big".
MULTIPLIERS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

PLOTLY_MODE = "directory"


def _registry_loader(registry_dir: Path):
    """Import registry/loader.py by path: a PyPI package called `loader`
    exists, and an installed one would shadow this project's module."""
    spec = importlib.util.spec_from_file_location(
        "_gauntlet_registry_loader", registry_dir / "loader.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"no loader.py under {registry_dir}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_registry


# --------------------------------------------------------------------------
# gather
# --------------------------------------------------------------------------

def collect(registry: dict, market: dict, tape: ex.Tape,
            blotter_dir: Path) -> dict:
    subs = sorted(registry)
    d: dict = {"subs": subs, "registry": registry, "per": {}}

    for key in subs:
        entry = registry[key]
        blot = load_blotter(blotter_dir / f"{key}-blotter.csv")
        window = (str(entry["backtest_window"]["start"]),
                  str(entry["backtest_window"]["end"]))
        b = fr.attach_book(blot, market)
        for fill in ("mid", "cross", "sweep"):
            b[f"px_{fill}"] = fr.fill_price(b, fr.FILLS[fill])

        stress = fr.size_stress(entry, blot, market, MULTIPLIERS)
        absorb = ex.absorption(entry, blot, market, tape)

        # tape print sizes for the products this strategy actually touched
        prints = []
        for p in absorb["product_id"].unique():
            lo, hi = tape.bounds.get(int(p), (0, 0))
            if lo != hi:
                prints.append(tape.qty[lo:hi])
        sizes = np.concatenate(prints) if prints else np.array([])

        d["per"][key] = {
            "entry": entry,
            "daily": {f: daily_pnl(b, f"px_{f}", window=window)
                      for f in ("mid", "cross", "sweep")},
            "crossing": fr.crossing_check(entry, blot, market),
            "depth": fr.depth_cost(entry, blot, market),
            "hold": fr.holding_time(blot),
            "stress": stress,
            "best_any": fr.best_multiplier(stress),
            "best_full": fr.best_multiplier(stress, require_full_fill=True),
            "absorb": absorb,
            "absorb_stats": ex.absorption_summary(entry, blot, market, tape),
            "print_sizes": sizes,   # binned at render time
        }

        pr = d["per"][key]
        pr["checks"] = vd.assess(sc.headline(pr["crossing"], pr["daily"]),
                                 pr["absorb_stats"])
        pr["conditions"] = vd.conditions(pr["checks"])
        pr["verdict"] = vd.overall(pr["checks"])

    _checks = {k: v["checks"] for k, v in d["per"].items()}
    d["grid"] = vd.grid(_checks)
    d["section_grid"] = vd.section_grid(_checks)
    d["crossing"] = fr.crossing_summary(registry, market, blotter_dir)
    d["absorb_table"] = ex.absorption_table(registry, market, tape, blotter_dir)

    book, prods = market["book"], market["products"]
    lvl = [c for c in book.columns if c.startswith(("bid_sz", "ask_sz"))]
    d["facts"] = {
        "products": len(prods),
        "unique_starts": int(prods["delivery_start"].nunique()),
        "book_rows": len(book),
        "book_keys": int(len(book.drop_duplicates(["ts", "product_id"]))),
        "levels": len({c.rsplit("_", 1)[1] for c in lvl}),
        "level_sizes": sorted(pd.unique(book[lvl].to_numpy().ravel())),
        "side_depth": float(book[[c for c in lvl
                                  if c.startswith("ask_sz")]].iloc[0].sum()),
        "snapshot_min": fr.SNAPSHOT.total_seconds() / 60.0,
        "unmatched_mw": float(sum(v["hold"]["unmatched_mw"]
                                  for v in d["per"].values())),
        "subs": len(subs),
    }
    return d


# --------------------------------------------------------------------------
# part 1 figures
# --------------------------------------------------------------------------

def f_modes(p: dict, key: str) -> go.Figure:
    """The three fill modes as equity curves, drawdown beneath."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.66, 0.34], vertical_spacing=0.06,
                        subplot_titles=("cumulative P&L", "drawdown"))
    for fill, col, label in [
            ("mid", MID_C, "1 · at mid (what the blotter records)"),
            ("cross", CROSS_C, "2 · at the touch, assuming we fill there"),
            ("sweep", SWEEP_C, "3 · sweeping the book by size")]:
        cum = p["daily"][fill].cumsum()
        fig.add_scatter(x=cum.index, y=cum.values, name=label, row=1, col=1,
                        line=dict(color=col, width=1.8),
                        hovertemplate="%{y:,.0f} EUR<extra></extra>")
        dd = cum - cum.cummax()
        fig.add_scatter(x=dd.index, y=dd.values, name=label, row=2, col=1,
                        showlegend=False, line=dict(color=col, width=1),
                        fill="tozeroy",
                        hovertemplate="%{y:,.0f} EUR<extra></extra>")
    fig.add_hline(y=0, line_width=1, line_color="black", row=1, col=1)
    fig.update_layout(title=f"{key} — the same trades under three fill modes",
                      template=TEMPLATE, height=560, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.13))
    return fig


def modes_table(p: dict) -> pd.DataFrame:
    base = p["daily"]["mid"].sum()
    rows = {}
    for label, fill in [("1 · mid", "mid"), ("2 · touch (BBO)", "cross"),
                        ("3 · sweep", "sweep")]:
        daily = p["daily"][fill]
        rows[label] = {
            "total_pnl_eur": float(daily.sum()),
            "sharpe_ann": sharpe(daily),
            "sortino_ann": fr.sortino(daily),
            "max_drawdown_eur": fr.max_drawdown(daily),
            "pnl_retained": float(daily.sum()) / base if base else np.nan,
            "vs_mid_eur": float(daily.sum()) - base,
        }
    return pd.DataFrame(rows).T


def f_size_pnl(p: dict, key: str) -> go.Figure:
    """P&L and Sharpe as the clip is scaled, over what the book actually fills.

    The upper panel alone is misleading: past the point where the book
    saturates, asking for a bigger clip changes nothing - the same MWh trades
    at the same price - so the bars go flat for a reason that is invisible
    unless the filled size is shown too. The lower panel is that size.
    """
    t = p["stress"]
    best_pnl_m = float(t["total_pnl_eur"].idxmax())
    bm = p["best_full"]
    best_sh_m = bm.get("multiplier", np.nan)

    # where the book stops delivering more MWh however much more we ask for
    cap = float(t["mwh_filled"].max())
    saturated = t.index[t["mwh_filled"] >= cap - 1e-6]
    sat = float(saturated[0]) if len(saturated) else np.nan
    # and where fills first fall short of the request
    short = t[t["mean_fill_frac"] < 0.999]
    first_short = float(short.index[0]) if len(short) else np.nan

    beyond = t.index.to_numpy(float) > sat if np.isfinite(sat) else \
        np.zeros(len(t), bool)
    colours = np.where(beyond, "#cfd5db",
                       np.where(t["total_pnl_eur"] >= 0, GOOD, WARN))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.64, 0.36], vertical_spacing=0.09,
                        specs=[[{"secondary_y": True}], [{}]],
                        subplot_titles=("P&L and Sharpe",
                                        "MWh the book actually filled"))

    fig.add_bar(x=t.index, y=t.total_pnl_eur, name="P&L (EUR)", opacity=0.9,
                marker_color=colours, row=1, col=1,
                customdata=np.c_[t.mwh_filled, t.mean_fill_frac],
                hovertemplate="%{x}x clip<br>%{y:,.0f} EUR<br>"
                              "filled %{customdata[0]:,.0f} MWh "
                              "(%{customdata[1]:.0%})<extra></extra>")
    fig.add_scatter(x=t.index, y=t.sharpe_ann, name="Sharpe",
                    mode="lines+markers", line=dict(color=ACCENT, width=2.4),
                    marker=dict(size=7), secondary_y=True, row=1, col=1,
                    hovertemplate="%{x}x clip<br>Sharpe %{y:.2f}<extra></extra>")

    fig.add_bar(x=t.index, y=t.mwh_wanted, name="MWh asked for",
                marker_color="#dfe4ea", row=2, col=1,
                hovertemplate="%{x}x clip<br>asked %{y:,.0f} MWh<extra></extra>")
    fig.add_bar(x=t.index, y=t.mwh_filled, name="MWh filled",
                marker_color=ACCENT, row=2, col=1,
                hovertemplate="%{x}x clip<br>filled %{y:,.0f} MWh"
                              "<extra></extra>")

    if np.isfinite(sat):
        for r in (1, 2):
            fig.add_vline(x=sat, line_color="#6b7280", line_width=1.6,
                          line_dash="dot", row=r, col=1)
        fig.add_annotation(
            x=sat, y=1.0, yref="y domain", row=1, col=1,
            text=f"book saturates at {sat:g}x ({cap:,.0f} MWh)<br>"
                 "asking for more changes nothing",
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=10.5, color="#4b5563"), align="left")

    fig.add_scatter(x=[best_pnl_m], y=[t.loc[best_pnl_m, "total_pnl_eur"]],
                    mode="markers", name=f"best P&L · {best_pnl_m:g}x",
                    marker=dict(symbol="diamond", size=15, color=GOOD,
                                line=dict(width=1.5, color="white")),
                    row=1, col=1,
                    hovertemplate=f"best P&L at {best_pnl_m:g}x<br>"
                                  "%{y:,.0f} EUR<extra></extra>")
    if pd.notna(best_sh_m):
        fig.add_scatter(x=[best_sh_m], y=[t.loc[best_sh_m, "sharpe_ann"]],
                        mode="markers", name=f"best Sharpe · {best_sh_m:g}x",
                        marker=dict(symbol="star", size=17, color=ACCENT,
                                    line=dict(width=1.2, color="white")),
                        secondary_y=True, row=1, col=1,
                        hovertemplate=f"best Sharpe at {best_sh_m:g}x<br>"
                                      "%{y:.2f}<extra></extra>")
    fig.add_hline(y=0, line_width=1, line_color="black", row=1, col=1)

    note = ""
    if np.isfinite(first_short):
        note = (f" · fills first fall short at {first_short:g}x, and the book "
                f"is fully consumed from {sat:g}x")
    sub = (f"best P&amp;L {t.loc[best_pnl_m, 'total_pnl_eur']:,.0f} EUR at "
           f"{best_pnl_m:g}x ({t.loc[best_pnl_m, 'clip_mw']:.0f} MW) · "
           f"best Sharpe {bm.get('best_value', float('nan')):.2f} at "
           f"{best_sh_m:g}x ({bm.get('clip_mw', float('nan')):.0f} MW)"
           + note)
    fig.update_layout(
        title=f"{key} — sweeping at N times the clip<br><sub>{sub}</sub>",
        template=TEMPLATE, height=620, barmode="overlay", bargap=0.35,
        legend=dict(orientation="h", y=-0.16))
    fig.update_yaxes(title_text="P&L (EUR)", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", secondary_y=True, showgrid=False,
                     row=1, col=1)
    fig.update_yaxes(title_text="MWh", row=2, col=1)
    fig.update_xaxes(title_text="clip size multiplier", row=2, col=1,
                     tickvals=list(t.index),
                     ticktext=[f"{m:g}x" for m in t.index])
    fig.update_xaxes(tickvals=list(t.index),
                     ticktext=[f"{m:g}x" for m in t.index], row=1, col=1)
    return fig


def optimum_vs_original(p: dict) -> pd.DataFrame:
    """The Sharpe-optimal clip beside the one actually traded."""
    t = p["stress"]
    m = p["best_full"].get("multiplier", np.nan)
    cols = ["clip_mw", "total_pnl_eur", "sharpe_ann", "sortino_ann",
            "max_drawdown_eur", "eur_per_mwh", "mean_fill_frac",
            "pct_past_touch", "pct_depth_exhausted"]
    base = t.loc[1.0, cols]
    if pd.isna(m):
        return base.to_frame("as traded (1x)").T
    best = t.loc[float(m), cols]
    out = pd.DataFrame({"as traded (1x)": base,
                        f"best Sharpe ({m:g}x)": best}).T
    out.loc["change"] = best - base
    return out


# --------------------------------------------------------------------------
# part 2 figures
# --------------------------------------------------------------------------

def _binned(values, bins: int = 50):
    """Pre-bin a distribution to a bar chart.

    A plotly histogram ships every raw value into the page; some of these
    arrays run to hundreds of thousands of points, which is most of the file
    size for nothing the reader can see.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.array([]), np.array([]), 0.0
    counts, edges = np.histogram(v, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return centres, counts, float(edges[1] - edges[0])


def f_print_sizes(p: dict, key: str) -> go.Figure:
    """Our clip against the size this market normally trades."""
    sz, a = p["print_sizes"], p["absorb_stats"]
    fig = go.Figure()
    x, y, w = _binned(sz, 50)
    if len(x):
        fig.add_bar(x=x, y=y, width=w, name="tape print size",
                    marker_color=ACCENT, opacity=0.85,
                    hovertemplate="%{x:.1f} MW<br>%{y:,} prints<extra></extra>")

    qs = {k: a.get(f"{k}_print_mw", np.nan)
          for k in ("p25", "median", "mean", "p75", "p90", "p95", "p99")}
    qs["median"] = a["median_print_mw"]
    qs["mean"] = a["mean_print_mw"]
    for i, (name, val) in enumerate(qs.items()):
        if not np.isfinite(val):
            continue
        fig.add_vline(x=val, line_color=GREY, line_width=1.4,
                      line_dash="dash" if name == "mean" else "dot",
                      annotation_text=name,
                      annotation_position="top" if i % 2 else "bottom",
                      annotation_font_size=10, annotation_font_color=GREY)
    fig.add_vline(x=a["clip_mw"], line_color=WARN, line_width=2.5,
                  annotation_text=f"our clip {a['clip_mw']:.0f} MW "
                                  f"({a['clip_vs_median_print']:.1f}x median)",
                  annotation_position="top right")

    shape = " · ".join(f"{k} {v:,.1f}" for k, v in qs.items()
                       if np.isfinite(v)) + " MW"
    fig.update_layout(
        title=f"{key} — how big is one clip, in this market's terms<br>"
              f"<sub>{shape}</sub>",
        xaxis_title="trade size (MW)", yaxis_title="prints on the tape",
        template=TEMPLATE, height=430, showlegend=False)
    return fig


def f_absorption(p: dict, key: str) -> go.Figure:
    """Time for the market to trade our size, with the shape of it named."""
    a, b = p["absorb_stats"], p["absorb"]
    mins = b["absorb_min"].dropna()
    fig = go.Figure()
    x, y, w = _binned(mins, 50)
    if len(x):
        fig.add_bar(x=x, y=y, width=w, name="minutes to absorb",
                    marker_color=ACCENT, opacity=0.85,
                    hovertemplate="%{x:.0f} min<br>%{y:,} clips"
                                  "<extra></extra>")

    qs: dict[str, float] = {}
    if len(mins):
        qs = {"p25": float(mins.quantile(0.25)),
              "median": float(mins.median()),
              "mean": float(mins.mean()),
              "p75": float(mins.quantile(0.75)),
              "p90": float(mins.quantile(0.90))}
        for i, (name, v) in enumerate(qs.items()):
            # stagger the labels, or five verticals collide at the top
            fig.add_vline(x=v, line_color=GREY, line_width=1.4,
                          line_dash="dash" if name == "mean" else "dot",
                          annotation_text=name,
                          annotation_position="top" if i % 2 else "bottom",
                          annotation_font_size=10, annotation_font_color=GREY)
    fig.add_vline(x=a["hold_median_min"], line_color=WARN, line_width=2.5,
                  annotation_text=f"we hold {a['hold_median_min']:.0f} min",
                  annotation_position="top right")

    shape = (" · ".join(f"{k} {v:,.0f}" for k, v in qs.items()) + " min"
             if qs else "no clip is absorbed before the product gates")
    fig.update_layout(
        title=f"{key} — how long the market takes to trade our size<br>"
              f"<sub>{shape}, against a {a['hold_median_min']:.0f} min hold — "
              f"{a['absorbed_within_hold_frac']:.0%} of clips are absorbed "
              "inside the window</sub>",
        xaxis_title="minutes for the tape to trade one clip",
        yaxis_title="clips", template=TEMPLATE, height=430, showlegend=False)
    return fig


# --------------------------------------------------------------------------
# observations - derived, never written in
# --------------------------------------------------------------------------

def standouts(d: dict, key: str) -> list[str]:
    p, c = d["per"][key], d["crossing"].loc[key]
    a, t = p["absorb_stats"], p["stress"]
    out = []

    if c.pnl_cross < 0 < c.pnl_mid:
        out.append(f"<b>Profitable at mid, loss-making at the touch</b> — "
                   f"{c.pnl_mid:,.0f} EUR becomes {c.pnl_cross:,.0f} EUR "
                   "before any question of depth or size.")
    elif c.pnl_retained < 0.5:
        out.append(f"<b>Keeps {c.pnl_retained:.0%} of its recorded P&amp;L</b> "
                   f"once it pays the spread: {c.pnl_mid:,.0f} to "
                   f"{c.pnl_cross:,.0f} EUR.")

    dep = p["depth"]
    over = float(dep.loc["clips_over_level_1", "sweep"])
    if over > 0:
        out.append(
            f"<b>{over:,.0f} clips are already larger than the depth at the "
            f"touch</b> ({dep.loc['clips_over_level_1_frac', 'sweep']:.1%} of "
            f"trades, up to {dep.loc['max_depth_used_x_level_1', 'sweep']:.1f}x "
            "it), so assuming a fill at the touch is not conservative for "
            "them - it is impossible.")

    bm = p["best_full"]
    if pd.notna(bm.get("multiplier", np.nan)):
        if bm["multiplier"] < 1.0:
            out.append(
                f"<b>Sharpe is best at {bm['multiplier']:g}x the current "
                f"clip</b> ({bm['clip_mw']:.0f} MW) — the size it trades "
                "today is already past the point where depth starts costing "
                "it.")
        else:
            at2 = t.loc[2.0, "sharpe_ann"] if 2.0 in t.index else np.nan
            out.append(
                f"<b>The current clip is the largest that costs nothing</b> — "
                f"Sharpe is flat up to {bm['multiplier']:g}x "
                f"({bm['clip_mw']:.0f} MW) and falls beyond"
                + (f", to {at2:.2f} at double the size." if pd.notna(at2)
                   else "."))

    exh = t[t["pct_depth_exhausted"] > 0]
    if len(exh):
        m = exh.index[0]
        out.append(f"<b>The book runs out entirely at {m:g}x the clip</b> "
                   f"({exh.loc[m, 'clip_mw']:.0f} MW): "
                   f"{exh.loc[m, 'pct_depth_exhausted']:.0%} of clips cannot "
                   f"be completed, and only "
                   f"{exh.loc[m, 'mean_fill_frac']:.0%} of the intended size "
                   "trades at all.")

    if a["absorb_vs_hold"] > 1:
        out.append(
            f"<b>The market needs {a['absorb_vs_hold']:.1f}x the holding "
            f"period just to trade one clip</b> — a median "
            f"{a['median_absorb_min']:.0f} min to absorb "
            f"{a['clip_mw']:.0f} MW against a "
            f"{a['hold_median_min']:.0f} min hold, and that is at 100% of "
            "the tape.")
    if a["absorbed_within_hold_frac"] < 0.5:
        out.append(
            f"<b>Only {a['absorbed_within_hold_frac']:.0%} of clips are "
            "absorbed inside the holding window</b>, so most positions are "
            "larger than the market moves in the time they are meant to "
            "exist.")
    if a["never_absorbed_frac"] > 0.05:
        out.append(
            f"<b>{a['never_absorbed_frac']:.0%} of clips are never absorbed "
            "before the product gates</b> — the tape does not trade that much "
            "in the time remaining at any participation rate.")
    return out


def standout_block(items: list[str], title: str = "What stands out") -> str:
    if not items:
        return ('<div class="note">Nothing in these numbers departs sharply '
                "from the other submissions.</div>")
    lis = "".join(f"<li>{s}</li>" for s in items)
    return (f'<div class="note warn"><b>{title}</b><ul>{lis}</ul></div>')


# --------------------------------------------------------------------------
# page furniture
# --------------------------------------------------------------------------

FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Sans:wght@400;500;600;700&'
         'family=IBM+Plex+Mono:wght@400;500;600&display=swap">')

# A single committed light ground, deliberately: plotly's chart template is
# light, and a tool read next to a terminal wants one stable surface. Every
# colour is painted from a token, so the page holds its look on any host.
CSS = """
:root { --ink:#16191d; --muted:#5c6672; --rule:#e2e6ec; --bg:#fff;
        --accent:#2d5f8b; --warn:#b3403f; --soft:#f5f7fa;
        --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",
               Roboto,Helvetica,Arial,sans-serif;
        --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,
               monospace; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:400 15px/1.62 var(--sans); -webkit-font-smoothing:antialiased; }
.wrap { max-width:1120px; margin:0 auto; padding:40px 28px 96px; }
header { border-bottom:2px solid var(--ink); }
.eyebrow { font-size:26px; font-weight:660; letter-spacing:-0.02em; }
.sub { color:var(--muted); font-size:14.5px; }
.tabs { display:flex; flex-wrap:wrap; gap:6px; margin:20px 0 -1px; }
.tabs button { font:600 13.5px/1 var(--sans); padding:10px 16px; cursor:pointer;
  border:1px solid var(--rule); border-bottom:none; border-radius:7px 7px 0 0;
  background:#fff; color:var(--muted); }
.tabs button:hover { color:var(--accent); border-color:var(--accent); }
.tabs button.on { background:var(--ink); border-color:var(--ink); color:#fff; }
.strat { display:none; }
.strat.on { display:block; }
.shead { margin:30px 0 0; }
.shead h1 { font-size:25px; margin:0 0 4px; letter-spacing:-0.02em; }
h2 { font-size:20px; margin:46px 0 4px; padding-top:20px;
     border-top:1px solid var(--rule); text-wrap:balance; }
h2.shared { border-top:2px solid var(--ink); margin-top:56px; }
h2 .num { color:var(--accent); margin-right:10px; font-family:var(--mono); }
h3 { font-size:14px; margin:28px 0 6px; color:var(--muted);
     text-transform:uppercase; letter-spacing:0.06em; font-weight:600; }
p { margin:10px 0; max-width:78ch; }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
@media (prefers-reduced-motion:reduce) { * { animation:none !important;
  transition:none !important; } }
details { background:var(--soft); border:1px solid var(--rule);
          border-radius:6px; margin:14px 0; padding:0 16px; }
details[open] { padding-bottom:12px; }
summary { cursor:pointer; padding:11px 0; font-weight:600; font-size:14px;
          color:var(--accent); }
summary:hover { text-decoration:underline; }
details dt { font-weight:640; margin-top:10px; font-size:14px; }
details dd { margin:2px 0 0 0; font-size:14px; color:#33383e; }
.note { background:var(--soft); border-left:3px solid var(--accent);
        padding:12px 16px; margin:16px 0; font-size:14.5px; }
.note.warn { border-left-color:var(--warn); }
.note ul { margin:8px 0 2px; padding-left:20px; }
.note li { margin:7px 0; }
table { border-collapse:collapse; width:100%; font-size:13px; margin:12px 0;
        font-family:var(--mono); font-variant-numeric:tabular-nums; }
th,td { padding:7px 10px; text-align:right; border-bottom:1px solid var(--rule); }
th { background:var(--soft); font-weight:600; font-family:var(--sans);
     font-size:12.5px; }
td:first-child, th:first-child { text-align:left; font-weight:600; }
tbody tr:hover { background:#fafbfc; }
.neg { color:var(--warn); }
.tablewrap { overflow-x:auto; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
         gap:12px; margin:20px 0; }
.card { border:1px solid var(--rule); border-radius:8px; padding:14px 16px; }
.card .k { font-size:11.5px; text-transform:uppercase; letter-spacing:0.06em;
           color:var(--muted); }
.card .v { font-size:22px; font-weight:600; margin-top:4px;
           font-family:var(--mono); letter-spacing:-0.01em;
           font-variant-numeric:tabular-nums; }
.card .m { font-size:12.5px; color:var(--muted); margin-top:2px; }
footer { margin-top:56px; padding-top:16px; border-top:1px solid var(--rule);
         color:var(--muted); font-size:13px; }
code { background:var(--soft); padding:1px 5px; border-radius:3px;
       font-family:var(--mono); font-size:12.5px; }
.badge { display:inline-block; font:600 11px/1 var(--mono);
  letter-spacing:0.06em; padding:5px 9px; border-radius:4px;
  border:1px solid; white-space:nowrap; }
.badge.keep { color:#2f6b46; background:#eef7f1; border-color:#bcdfc9; }
.badge.suspect { color:#8a5a12; background:#fdf5e7; border-color:#eed9ae; }
.badge.fail { color:#93332f; background:#fdeeed; border-color:#eec3c0; }
.verdictbar { display:flex; align-items:center; gap:12px; margin:22px 0 6px;
  padding:13px 16px; border-radius:8px; border:1px solid var(--rule);
  font-weight:600; font-size:15px; }
.verdictbar.keep { background:#f2f9f5; border-color:#cfe6d8; }
.verdictbar.suspect { background:#fdf8ef; border-color:#eee0c2; }
.verdictbar.fail { background:#fdf2f1; border-color:#eecfcd; }
.verdictline { padding:9px 2px; border-bottom:1px solid var(--rule);
  font-size:13.5px; }
.verdictline .vhead { display:grid; grid-template-columns:86px 92px 1fr;
  gap:12px; align-items:baseline; }
.crule { margin:4px 0 0 190px; font-size:12.5px; color:var(--muted); }
.crule.overall { margin:10px 0 0 0; padding-left:12px;
  border-left:2px solid var(--rule); }
table.rules td { text-align:left; vertical-align:top; font-family:var(--sans);
  font-size:13.5px; }
table.rules td.cname { font-family:var(--mono); font-size:12.5px;
  color:var(--accent); white-space:nowrap; width:110px; }
table.rules th { text-align:left; }
details.criteria { background:#fff; border:1px solid var(--rule);
  margin:20px 0 8px; padding:0 18px; }
details.criteria > summary { font-size:15px; padding:14px 0; }
details.criteria[open] { padding-bottom:16px; }
h4.crit { font-size:12px; text-transform:uppercase; letter-spacing:0.07em;
  color:var(--muted); margin:16px 0 2px; font-weight:600; }
p.rule { font-size:13.5px; color:#33383e; margin:6px 0 0; }
/* the criteria panel: one card per check, its rules as badge + bullets */
.rulecols { display:grid; gap:10px; margin:8px 0 4px;
  grid-template-columns:repeat(auto-fit,minmax(350px,1fr)); }
.checkrule { border:1px solid var(--rule); border-radius:8px;
  padding:11px 13px 12px; background:#fcfcfd; }
.checkrule .chead { display:flex; flex-wrap:wrap; align-items:baseline;
  gap:4px 10px; margin-bottom:9px; padding-bottom:8px;
  border-bottom:1px solid var(--rule); }
.checkrule .cname { font-family:var(--mono); font-size:12.5px;
  font-weight:600; color:var(--accent); }
.checkrule .cwhat { flex:1 1 220px; font-size:12.5px; color:var(--muted); }
.rulegroup { display:grid; grid-template-columns:78px 64px 1fr; gap:4px 9px;
  align-items:baseline; margin:6px 0; }
.rulegroup .rif { font-size:12px; font-style:italic; color:var(--muted); }
.rulegroup ul { margin:0; padding-left:17px; font-size:12.5px;
  color:#33383e; }
.rulegroup li { margin:2px 0; }
.rulegroup li::marker { color:var(--muted); }
.verdictline .cname { font-family:var(--mono); font-size:12.5px;
  color:var(--muted); }
.verdictline .cnote { color:#33383e; }
table.grid td, table.grid th { text-align:center; }
table.grid td:first-child, table.grid th:first-child { text-align:left; }
.sname { font-family:var(--sans); font-weight:640; }
.sname a { color:var(--ink); text-decoration:none;
  border-bottom:1px solid var(--rule); }
.sname a:hover { color:var(--accent); border-color:var(--accent); }
.secrow { display:flex; flex-wrap:wrap; gap:10px; margin:0 0 14px; }
.secpill { display:flex; align-items:center; gap:9px; padding:8px 13px;
  border:1px solid var(--rule); border-radius:7px; font-size:13px;
  color:var(--muted); }
.scls { display:block; font-family:var(--mono); font-size:11px;
  font-weight:400; color:var(--muted); margin-top:2px; }
"""

FMT = {
    "total_pnl_eur": "{:,.0f}", "max_drawdown_eur": "{:,.0f}",
    "vs_mid_eur": "{:,.0f}", "pnl_vs_1x": "{:,.0f}",
    "mwh_filled": "{:,.0f}", "mwh_wanted": "{:,.0f}",
    "clip_mw": "{:,.1f}", "median_print_mw": "{:,.1f}",
    "p90_print_mw": "{:,.1f}", "mean_print_mw": "{:,.1f}",
    "median_product_volume_mwh": "{:,.0f}",
    "hold_median_min": "{:,.0f}", "median_absorb_min": "{:,.0f}",
    "p90_absorb_min": "{:,.0f}",
    "pnl_retained": "{:.1%}", "mean_fill_frac": "{:.1%}",
    "pct_depth_exhausted": "{:.1%}", "pct_past_touch": "{:.1%}",
    "absorbed_within_hold_frac": "{:.1%}", "never_absorbed_frac": "{:.1%}",
    "clip_vs_median_print": "{:.2f}", "absorb_vs_hold": "{:.2f}",
}


def table_html(df: pd.DataFrame, index_name: str = "") -> str:
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for name, row in df.iterrows():
        cells = []
        for col, v in row.items():
            if isinstance(v, (int, float, np.floating)) and pd.notna(v):
                txt = FMT.get(col, "{:,.3f}").format(v)
                cls = ' class="neg"' if v < 0 else ""
            else:
                txt = "·" if (not isinstance(v, str) and pd.isna(v)) else str(v)
                cls = ""
            cells.append(f"<td{cls}>{txt}</td>")
        label = f"{name:g}x" if isinstance(name, float) else name
        rows.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    return ('<div class="tablewrap"><table><thead><tr>'
            f"<th>{index_name}</th>{head}</tr></thead><tbody>"
            f"{''.join(rows)}</tbody></table></div>")


def details(summary: str, items: list[tuple[str, str]]) -> str:
    dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in items)
    return f"<details><summary>{summary}</summary><dl>{dl}</dl></details>"


def card(k: str, v: str, m: str = "") -> str:
    return (f'<div class="card"><div class="k">{k}</div>'
            f'<div class="v">{v}</div><div class="m">{m}</div></div>')


class Page:
    def __init__(self, title: str):
        self.title, self.parts, self.first = title, [], True

    def html(self, s: str) -> None:
        self.parts.append(s)

    def fig(self, figure: go.Figure) -> None:
        self.parts.append(figure.to_html(
            full_html=False,
            include_plotlyjs=PLOTLY_MODE if self.first else False,
            config={"displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]}))
        self.first = False

    def render_fragment(self) -> str:
        body = "\n".join(self.parts)
        return (f"<title>{self.title}</title>{FONTS}<style>{CSS}</style>"
                f'<div class="wrap">{body}</div>')

    def render(self) -> str:
        return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,'
                f'initial-scale=1"><title>{self.title}</title>{FONTS}'
                f'<style>{CSS}</style></head><body>'
                f'<div class="wrap">{chr(10).join(self.parts)}</div>'
                "</body></html>")


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

BADGE = {vd.KEEP: ("keep", "KEEP"), vd.SUSPECT: ("suspect", "SUSPECT"),
         vd.FAIL: ("fail", "FAIL")}


def badge(verdict: str) -> str:
    cls, txt = BADGE[verdict]
    return f'<span class="badge {cls}">{txt}</span>'


def check_note(p: dict, name: str) -> str:
    """A check's badge, what it found, and the criterion that decided it."""
    c = next((c for c in p["checks"] if c.name == name), None)
    if c is None:
        return ""
    return (f'<div class="verdictline"><div class="vhead">{badge(c.verdict)}'
            f'<span class="cname">{c.name}</span>'
            f'<span class="cnote">{c.note}</span></div>'
            f'<div class="crule">{rule_inline(name)}</div></div>')


# Each check's criteria as bullets, keyed by the verdict they produce, so the
# page can render "FAIL if ... / SUSPECT if ..." rather than a paragraph.
RULES = {
    "mid": {
        "what": "P&amp;L at the price the blotter records",
        vd.FAIL: ["P&amp;L is not positive"],
    },
    "edge": {
        "what": "the gross edge per MWh against the cost of earning it - "
                "half the spread plus the fee",
        vd.FAIL: [f"coverage is under <b>{vd.COVERAGE_FAIL:.1f}x</b> — the "
                  "edge is inside the cost of trading it"],
        vd.SUSPECT: [f"coverage is under <b>{vd.COVERAGE_SUSPECT:.1f}x</b> — "
                     "friction takes most of it"],
    },
    "touch": {
        "what": "crossing at the touch, measured against mid",
        vd.FAIL: ["P&amp;L turns negative",
                  "P&amp;L <b>and</b> Sharpe each keep under 50% of mid"],
        vd.SUSPECT: ["<b>exactly one</b> of P&amp;L or Sharpe keeps under 50% "
                     "of mid"],
    },
    "sweep": {
        "what": "sweeping by size, measured against mid and against the touch",
        vd.FAIL: ["P&amp;L turns negative",
                  "P&amp;L <b>and</b> Sharpe each keep under 50% of mid"],
        vd.SUSPECT: ["P&amp;L or Sharpe keeps under 50% of its value "
                     "<b>at the touch</b> — the clip is too big for the book, "
                     "which is a size cost rather than a spread cost",
                     "the <b>touch</b> check is SUSPECT — sweeping cannot be "
                     "sounder than the fill it is built on"],
    },
    "clip_size": {
        "what": "our clip against the sizes this market actually prints",
        vd.SUSPECT: [f"the clip is larger than the <b>"
                     f"{vd.CLIP_QUANTILE * 100:.0f}th percentile</b> of "
                     "print sizes"],
    },
    "absorb": {
        "what": "tape time to trade one clip ÷ median holding period",
        vd.FAIL: ["the ratio is above <b>1.0x</b> — the market needs longer "
                  "than the position is meant to exist",
                  "the tape never trades one clip before the product gates"],
        vd.SUSPECT: ["the ratio is above <b>0.5x</b>"],
    },
}
OVERALL_RULE = ("Any FAIL fails the submission; any SUSPECT and no FAIL makes "
                "it SUSPECT; otherwise KEEP.")


def rule_bullets(name: str) -> str:
    """One check's criteria as FAIL / SUSPECT bullet lists."""
    spec = RULES.get(name, {})
    out = []
    for verdict in (vd.FAIL, vd.SUSPECT):
        items = spec.get(verdict)
        if not items:
            continue
        lis = "".join(f"<li>{i}</li>" for i in items)
        out.append(f'<div class="rulegroup">{badge(verdict)}'
                   f'<span class="rif">if</span><ul>{lis}</ul></div>')
    out.append(f'<div class="rulegroup">{badge(vd.KEEP)}'
               '<span class="rif">otherwise</span></div>')
    return "".join(out)


def rule_inline(name: str) -> str:
    """The same criteria on one line, to sit beside a single verdict."""
    spec = RULES.get(name, {})
    bits = []
    for verdict in (vd.FAIL, vd.SUSPECT):
        items = spec.get(verdict)
        if items:
            bits.append(f"<b>{verdict}</b> if " + "; or ".join(items))
    return " · ".join(bits) if bits else ""


def check_block(name: str) -> str:
    """A check: its name, what it measures, and its criteria as bullets."""
    spec = RULES.get(name, {})
    return (f'<div class="checkrule"><div class="chead">'
            f'<span class="cname">{name}</span>'
            f'<span class="cwhat">{spec.get("what", "")}</span></div>'
            f"{rule_bullets(name)}</div>")


def criteria_panel() -> str:
    """The criteria, grouped by section, as bullets. Open on load."""
    groups: dict[str, list[str]] = {}
    for name in RULES:
        groups.setdefault(vd.SECTION[name], []).append(check_block(name))
    body = []
    for section, blocks in groups.items():
        body.append(f'<h4 class="crit">{section}</h4>'
                    f'<div class="rulecols">{"".join(blocks)}</div>')
    body.append(f'<h4 class="crit">overall</h4><p class="rule">{OVERALL_RULE}'
                "</p>")
    return ('<details class="criteria" open><summary>Pass / suspect / fail '
            "criteria, by section</summary>" + "".join(body) + "</details>")


def rules_table() -> str:
    """The criteria again in the overview tab - same bullets, no disclosure."""
    groups: dict[str, list[str]] = {}
    for name in RULES:
        groups.setdefault(vd.SECTION[name], []).append(check_block(name))
    out = []
    for section, blocks in groups.items():
        out.append(f'<h4 class="crit">{section}</h4>'
                   f'<div class="rulecols">{"".join(blocks)}</div>')
    out.append(f'<h4 class="crit">overall</h4><p class="rule">{OVERALL_RULE}'
               "</p>")
    return "".join(out)


def overview_section(d: dict, pg: Page) -> None:
    """The seven side by side - its own tab, shown first."""
    pg.html('<section class="strat on" id="s-overview">')
    pg.html('<div class="shead"><h1>Overview</h1>'
            '<div class="sub">every submission, by section and by '
            "individual check</div></div>")

    g, sg = d["grid"], d["section_grid"]
    counts = g["OVERALL"].value_counts()
    line = " · ".join(f"{counts.get(v, 0)} {v.lower()}"
                      for v in (vd.KEEP, vd.SUSPECT, vd.FAIL))
    passed = list(g.index[g["OVERALL"] == vd.KEEP])
    pg.html(f'<div class="note"><b>{line}.</b> '
            + (f"Passing every check: <b>{', '.join(passed)}</b>."
               if passed else "No submission passes every check.")
            + f" {OVERALL_RULE}</div>")

    for title, table in (("By section", sg), ("By individual check", g)):
        rows = []
        for k in table.index:
            cells = "".join(f"<td>{badge(table.loc[k, c])}</td>"
                            for c in table.columns)
            rows.append(f'<tr><td class="sname">'
                        f'<a href="#{k}" data-go="{k}">'
                        f'{d["registry"][k]["name"]}</a>'
                        f'<span class="scls">{d["registry"][k]["class"]}'
                        f"</span></td>{cells}</tr>")
        head = "".join(f"<th>{c}</th>" for c in table.columns)
        pg.html(f"<h3>{title}</h3>")
        pg.html('<div class="tablewrap"><table class="grid"><thead><tr>'
                f"<th>strategy</th>{head}</tr></thead><tbody>"
                f"{''.join(rows)}</tbody></table></div>")

    pg.html("<h3>Pass / suspect / fail criteria</h3>")
    pg.html(rules_table())
    pg.html("</section>")


def strategy_section(d: dict, key: str, pg: Page) -> None:
    p = d["per"][key]
    e, c, a, t = p["entry"], d["crossing"].loc[key], p["absorb_stats"], p["stress"]
    pg.html(f'<section class="strat" id="s-{key}">')
    pg.html(f"""
<div class="shead">
  <h1>{e['name']}</h1>
  <div class="sub">{e['class']} · {int(len(p['absorb'])):,} trades ·
  {e['backtest_window']['days']} days · clip {a['clip_mw']:.0f} MW ·
  holds {a['hold_median_min']:.0f} min</div>
</div>""")

    ov = p["verdict"]
    where = vd.sections_failing(p["checks"])
    tail = (" — on " + " and ".join(where)) if where else ""
    pg.html(f'<div class="verdictbar {BADGE[ov][0]}">{badge(ov)}'
            f'<span>{e["name"]}{tail}</span></div>')
    pg.html('<div class="secrow">' + "".join(
        f'<div class="secpill">{badge(vd.section_verdict(p["checks"], sec))}'
        f'<span>{sec}</span></div>' for sec in vd.SECTIONS) + "</div>")
    pg.html("".join(check_note(p, n) for n in vd.CHECKS))
    pg.html(f'<div class="crule overall">{OVERALL_RULE}</div>')

    bm = p["best_full"]
    pg.html('<div class="cards">'
            + card("at mid", f"{c.pnl_mid:,.0f}", "EUR, as recorded")
            + card("at the touch", f"{c.pnl_cross:,.0f}",
                   f"{c.pnl_retained:.0%} retained")
            + card("sweeping", f"{p['daily']['sweep'].sum():,.0f}",
                   f"{float(p['depth'].loc['clips_over_level_1', 'sweep']):,.0f}"
                   " clips past the touch")
            + card("best clip multiplier",
                   f"{bm['multiplier']:g}x" if pd.notna(bm.get('multiplier'))
                   else "·",
                   f"Sharpe {bm.get('best_value', float('nan')):.2f}")
            + card("clip vs typical trade",
                   f"{a['clip_vs_median_print']:.1f}x",
                   f"median print {a['median_print_mw']:.1f} MW")
            + card("absorb vs hold", f"{a['absorb_vs_hold']:.2f}x",
                   f"{a['absorbed_within_hold_frac']:.0%} inside the window")
            + "</div>")
    pg.html(standout_block(standouts(d, key)))

    # ---- part 1 ----
    fm = [c for c in p["checks"] if c.name in ("mid", "touch", "sweep")]
    pg.html(f'<h2><span class="num">01</span>Fill mode stress test '
            f'{badge(vd.overall(fm))}</h2>')
    pg.fig(f_modes(p, key))
    pg.html(table_html(modes_table(p), "fill mode"))
    pg.html(details("What each mode assumes", [
        ("1 · mid", "The price the blotter records, "
                    "<code>(bid_1 + ask_1)/2</code>. Derived, never quoted by "
                    "anyone, and the only price at which the submitted "
                    "figures reproduce."),
        ("2 · touch (BBO)", "A buy lifts <code>ask_1</code> and a sell hits "
         "<code>bid_1</code>, <b>assuming we are filled there whatever the "
         "size</b>. Where the clip is bigger than the depth showing at the "
         "touch, that assumption is not conservative - it is impossible, and "
         "mode 3 is the honest version."),
        ("3 · sweep", f"The clip consumes levels 1..{d['facts']['levels']} by "
         "size, paying the volume-weighted price of what it takes. Identical "
         "to mode 2 whenever the clip fits at the touch."),
    ]))

    pg.html("<h3>Sweeping at N times the clip</h3>")
    pg.fig(f_size_pnl(p, key))
    pg.html(table_html(optimum_vs_original(p), "clip size"))
    tied = (f"from {bm['tied_from']:g}x" if pd.notna(bm.get("tied_from"))
            else "")
    pg.html(details("Reading the size sweep", [
        ("Why Sharpe barely moves at small sizes", "Sharpe is scale-free: "
         "doubling every trade doubles the mean daily P&amp;L and its standard "
         "deviation alike. It moves here only because the <i>cost per MWh</i> "
         "of sweeping deeper changes, so it is flat across every multiplier "
         "whose clip still fits at the touch."),
        ("The best multiplier", f"The largest size still within a hair of the "
         f"best Sharpe - here {bm.get('multiplier', float('nan')):g}x, tied "
         f"{tied}. Read it as the biggest the clip can get before depth starts "
         "costing anything, not as an interior optimum."),
        ("mean_fill_frac", "The share of the scaled clip the book could "
         "absorb. Below 100% the strategy is not trading the size it asked "
         "for."),
        ("pct_depth_exhausted", "How often the clip is bigger than the entire "
         f"visible book ({d['facts']['side_depth']:,.0f} MW a side here). "
         "Past that point P&amp;L stops changing with size, because no extra "
         "volume can trade."),
        ("A limit worth naming", "Scaling the clip is more counterfactual "
         "than repricing it: it assumes the same trades still happen and that "
         "the book does not react. There is no market-impact term, so these "
         "numbers are the optimistic case."),
    ]))

    # ---- part 2 ----
    ab = [c for c in p["checks"] if c.name in ("clip_size", "absorb")]
    pg.html(f'<h2><span class="num">02</span>Market absorption time '
            f'{badge(vd.overall(ab))}</h2>')
    pg.fig(f_print_sizes(p, key))
    pg.fig(f_absorption(p, key))
    stats = pd.DataFrame({key: {
        "clip_mw": a["clip_mw"], "median_print_mw": a["median_print_mw"],
        "p90_print_mw": a["p90_print_mw"],
        "clip_vs_median_print": a["clip_vs_median_print"],
        "median_product_volume_mwh": a["median_product_volume_mwh"],
        "hold_median_min": a["hold_median_min"],
        "median_absorb_min": a["median_absorb_min"],
        "p90_absorb_min": a["p90_absorb_min"],
        "absorb_vs_hold": a["absorb_vs_hold"],
        "absorbed_within_hold_frac": a["absorbed_within_hold_frac"],
        "never_absorbed_frac": a["never_absorbed_frac"]}}).T
    pg.html(table_html(stats, "submission"))
    pg.html(details("How absorption is measured", [
        ("The question", "Not what price we would get, but whether this "
         "market trades our size at all in the time we give it."),
        ("absorb_min", "Minutes from the recorded trade until cumulative tape "
         "volume in that product reaches one clip. Every print counts and we "
         "take all of it, so this is the market absorbing our size at "
         "<b>100% participation</b> - a floor on execution time, not an "
         "estimate. Any realistic participation rate is slower."),
        ("absorb_vs_hold", "That time divided by the median holding period. "
         "Above 1.0 the market needs longer to trade one clip than the "
         "position is meant to exist."),
        ("never_absorbed_frac", "Clips the tape never absorbs before the "
         "product gates. A leg placed at the last snapshot has no forward "
         "window at all, which is why some strategies show a large share "
         "here."),
        ("The tape is not us", "Almost no blotter trade appears in the tape, "
         "so this is the market <i>excluding</i> these strategies: a real "
         "execution would have to participate alongside it, not instead of "
         "it."),
    ]))
    pg.html("</section>")


def single_page(d: dict, fragment: bool = False) -> str:
    pg = Page("Execution Stress Test")
    subs = d["subs"]
    tabs = ('<button data-go="overview" class="on">overview</button>'
            + "".join(f'<button data-go="{k}">{k}</button>' for k in subs))
    pg.html(f"""
<header>
  <div class="eyebrow">Execution stress test</div>
  <div class="sub">{len(subs)} submissions · fill mode and market absorption ·
  one strategy at a time</div>
  <div class="tabs">{tabs}</div>
</header>

<div class="note">Every fill in every blotter is recorded at the book
<b>mid</b> — a derived number, never quoted by anyone — so mid is treated as
the claim and everything else is measured against it. The measurements are
descriptive; the thresholds that turn them into a verdict are stated in full
under each strategy and in <b>Final results</b>.</div>""")

    pg.html(criteria_panel())

    overview_section(d, pg)
    for key in subs:
        strategy_section(d, key, pg)

    fx = d["facts"]
    uniq = ("uniquely" if fx["unique_starts"] == fx["products"]
            else "NOT uniquely - delivery starts repeat, so check the join")
    dup = ("no duplicate keys" if fx["book_keys"] == fx["book_rows"]
           else f"DUPLICATE keys ({fx['book_rows'] - fx['book_keys']} extra)")
    unm = (f"{fx['unmatched_mw']:.1f} MW left unmatched across the "
           f"{fx['subs']} submissions"
           + (", confirming every position is opened and closed by a trade."
              if fx["unmatched_mw"] < 1e-6
              else " - so some positions are NOT closed by a trade."))
    lsz = fx["level_sizes"]
    pg.html('<h2 class="shared"><span class="num">03</span>Method — applies to '
            'every strategy above</h2>')
    pg.html(details("How this dataset is shaped, and how trades join to it", [
        ("Instrument", f"<code>product_delivery</code> to "
         f"<code>products.delivery_start</code> to <code>product_id</code>: "
         f"{fx['products']:,} products, {fx['unique_starts']:,} unique "
         f"delivery starts, so the delivery hour identifies it {uniq}."),
        ("Snapshot", f"<code>(exec_ts, product_id)</code> against the book's "
         f"{fx['book_rows']:,} rows, which have {dup}. Row counts survive both "
         "merges, which is what rules out a silent fan-out."),
        ("Book", f"{fx['levels']} levels a side, snapshotted every "
         f"{fx['snapshot_min']:.0f} minutes, "
         + (f"{lsz[0]:,.0f} MW showing at every level"
            if len(lsz) == 1 else
            f"{min(lsz):,.0f}-{max(lsz):,.0f} MW per level")
         + f" — {fx['side_depth']:,.0f} MW a side in total."),
        ("Holding time", f"FIFO-matched opens against closes per product. "
         f"{unm}"),
    ]))
    pg.html(details("What is assumed, and what this cannot see", [
        ("Costs", "Always adverse: they lift buys and cut sells."),
        ("Depth", f"Only visible depth, {fx['levels']} levels a side. A clip "
                  "larger than that has no modelled price for the excess."),
        ("Not modelled", "Market impact, queue position, and any reaction by "
         "other participants. Repricing recorded trades can only make "
         "execution worse; it cannot say whether the trades were achievable, "
         "which is what the absorption section is for."),
    ]))
    pg.html(f"""
<footer>Generated by <code>starter/friction_report.py</code> from
<code>friction.py</code> and <code>execution.py</code> · thresholds in <code>verdicts.py</code> · {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC</footer>""")

    # plots built in a hidden section have no width until shown
    pg.html("""
<script>
(function () {
  function resize(sec) {
    if (!sec) { return; }
    sec.querySelectorAll('.plotly-graph-div').forEach(function (el) {
      try { Plotly.Plots.resize(el); } catch (err) { /* not drawn yet */ }
    });
  }
  function show(key) {
    document.querySelectorAll('.strat').forEach(function (s) {
      s.classList.toggle('on', s.id === 's-' + key);
    });
    document.querySelectorAll('.tabs button').forEach(function (b) {
      b.classList.toggle('on', b.dataset.go === key);
    });
    resize(document.getElementById('s-' + key));
    history.replaceState(null, '', '#' + key);
  }
  document.querySelectorAll('.tabs button, a[data-go]').forEach(
    function (b) {
      b.addEventListener('click', function (ev) {
        ev.preventDefault();
        show(b.dataset.go);
        window.scrollTo({ top: 0 });
      });
    });
  var want = decodeURIComponent(location.hash.slice(1));
  if (want && document.getElementById('s-' + want)) { show(want); }
  window.addEventListener('resize', function () {
    resize(document.querySelector('.strat.on'));
  });
})();
</script>""")
    return pg.render_fragment() if fragment else pg.render()


def build_report(out_path: str | Path | None = None,
                 data_dir: str | Path | None = None,
                 registry_dir: str | Path | None = None,
                 blotter_dir: str | Path | None = None,
                 plotly: str = "directory", fragment: bool = False) -> str:
    """Build the page. `plotly="inline"` and `fragment=True` make it
    self-contained for publishing."""
    global PLOTLY_MODE
    PLOTLY_MODE = plotly
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    registry_dir = Path(registry_dir) if registry_dir else ROOT / "registry"
    blotter_dir = Path(blotter_dir) if blotter_dir else ROOT / "blotters"
    out = (Path(out_path) if out_path
           else HERE / "figures" / "execution_stress.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    registry = _registry_loader(registry_dir)(registry_dir)
    market = load_market(data_dir)
    tape = ex.Tape.load(data_dir)
    d = collect(registry, market, tape, blotter_dir)

    if plotly == "directory":
        from plotly.offline import get_plotlyjs
        (out.parent / "plotly.min.js").write_text(get_plotlyjs(),
                                                  encoding="utf-8")
    html = single_page(d, fragment=fragment)
    if plotly == "inline":
        # plotly's bundle carries a raw U+FFFD inside a regex literal; some
        # hosts reject a replacement character in served content, and in
        # JavaScript the escape is exactly equivalent
        html = html.replace("�", "\\uFFFD")
    out.write_text(html, encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = build_report(args[0] if args else None,
                        plotly="inline" if "--inline" in sys.argv else "directory",
                        fragment="--fragment" in sys.argv)
    mb = Path(path).stat().st_size / 1e6
    print(f"wrote {path} ({mb:.1f} MB)")
