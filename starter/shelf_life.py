"""Shelf-life — section 4 of the validation report.

    "Will it keep working — and does it work everywhere? An edge that lives in
     three cells of a heatmap, or died six months ago inside a still-positive
     full-sample Sharpe, is not the edge the summary table describes."

This section makes time and regime first-class. Four views, each asking one
question about the *stability* of the edge (not its size — that's Luck, and not
its execution cost — that's Friction):

  1. Seasonality of P&L      — WHERE (month / weekday / delivery-hour) is the money?
  2. P&L vs vol & turnover    — does the edge only exist in one market regime?
  3. Rolling Sharpe & t-stat  — did it decay / die partway through the sample?
  4. Rolling skewness         — is the edge riding a market-skew regime that can flip?

P&L basis: the strategy's own recorded fills (`price` column), which in this
pack sit at the book mid — so this is the claimed edge, audited for durability.
Crossing the spread is Friction's job, not this section's.

The HTML report is an INTERACTIVE Plotly dashboard: the raw series are embedded
and the statistics recompute in the browser, so an analyst can drag the rolling
window, drop the best days, or rebin the regime curves and watch the edge move.

Public API (built to drop into report.run() later, unchanged):

    shelf_life(registry_entry, blotter, market) -> SectionResult
    render_html(submission, blotter, market, registry_entry, out_path) -> Path
    generate(submission="windfall", ...) -> Path      # the one function to run
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "registry"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from report import SectionResult, Finding                       # noqa: E402
from repricer import (load_market, load_blotter,                # noqa: E402
                      product_pnl, daily_pnl, sharpe)
from loader import load_registry                                # noqa: E402

# ---- tunable thresholds (documented, so a reviewer can argue with them) -----
ROLL_WIN = 60          # default rolling window (days); the slider overrides it live
DECAY_SUSPECT = 0.60   # last-third Sharpe below this fraction of first-third -> SUSPECT
DECAY_FAIL = 0.0       # last-third Sharpe at/below this (first-third strong) -> FAIL
CONC_MONTH = 0.40      # >40% of total P&L from one calendar month -> SUSPECT
REGIME_CORR = 0.50     # |corr(daily P&L, regime)| above this -> edge is a regime bet
BREADTH_MIN = 0.60     # fewer than 60% of active months positive -> SUSPECT
TSTAT_CRIT = 1.96      # two-sided 5% band on the rolling t-stat
# Plotly is vendored (downloaded once) and inlined so each report is a single,
# self-contained, offline-portable HTML file. Falls back to the CDN if missing.
_VENDOR_PLOTLY = _HERE / "_vendor" / "plotly.min.js"
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# ===========================================================================
# P&L decomposition
# ===========================================================================
def _hourly_pnl(blotter: pd.DataFrame, price_col: str = "price") -> pd.Series:
    """Realised P&L per delivery hour (sells minus buys), indexed by the hourly
    `product_delivery` timestamp — the atom the seasonality views group on."""
    return product_pnl(blotter, price_col)


def _daily_pnl(blotter: pd.DataFrame, registry_entry: dict,
               price_col: str = "price") -> pd.Series:
    """Daily P&L over the declared backtest window, flat days included."""
    w = registry_entry.get("backtest_window", {})
    window = (str(w["start"]), str(w["end"])) if {"start", "end"} <= set(w) else None
    return daily_pnl(blotter, price_col, window=window)


# ===========================================================================
# Market regime series (derived once from the order book)
# ===========================================================================
def market_regime(market: dict) -> pd.DataFrame:
    """Per delivery DAY, averaged over that day's products:

        vol      — std of the intraday mid over the ~40 pre-gate snapshots
        turnover — quote churn, sum |Δmid| across snapshots (chosen proxy: the
                   book carries no traded volume and sizes are a constant 10 MW,
                   so depth is uninformative)
        price    — mean mid, used only to build a daily market return for skew
        ret      — day-over-day pct change of that price level
        bias     — intraday mid − day-ahead auction price (EUR/MWh): the market's
                   premium over the day-ahead (our stand-in for Tom's imbalance
                   bias, which needs an imbalance price the gauntlet data lacks)
    """
    bk = market["book"][["ts", "product_id", "mid"]].sort_values(["product_id", "ts"])
    same = bk["product_id"].eq(bk["product_id"].shift())
    churn = bk["mid"].diff().abs().where(same).groupby(bk["product_id"]).sum()
    g = bk.groupby("product_id")["mid"]
    per_prod = pd.DataFrame({"vol": g.std(), "turnover": churn, "price": g.mean()})

    da = market["da_auction"].set_index("product_id")["da_price_eur_mwh"]
    per_prod["bias"] = per_prod["price"] - da            # intraday mid − day-ahead (EUR/MWh)
    prods = market["products"][["product_id", "delivery_start"]].set_index("product_id")
    per_prod = per_prod.join(prods)
    per_prod["day"] = per_prod["delivery_start"].dt.normalize()

    daily = per_prod.groupby("day")[["vol", "turnover", "price", "bias"]].mean().sort_index()
    daily["ret"] = daily["price"].pct_change()
    return daily


def _thirds_sharpe(daily: pd.Series):
    """Sharpe on the first vs last third of the sample — the cleanest read on
    'did it decay'."""
    if (daily != 0.0).sum() < 9:
        return np.nan, np.nan
    n = len(daily)
    return sharpe(daily.iloc[: n // 3]), sharpe(daily.iloc[2 * n // 3:])


# ===========================================================================
# Verdict
# ===========================================================================
def _verdict(registry_entry: dict, daily: pd.Series, hourly: pd.Series,
             regime: pd.DataFrame) -> SectionResult:
    total = float(daily.sum())
    findings: list[Finding] = []

    # --- decay -------------------------------------------------------------
    sr_first, sr_last = _thirds_sharpe(daily)
    if np.isnan(sr_first) or sr_first <= 0:
        v, note = "INFO", "first-third Sharpe not positive — decay test not meaningful"
    elif sr_last <= DECAY_FAIL:
        v = "FAIL"
        note = f"edge has died: first-third Sharpe {sr_first:.2f} -> last-third {sr_last:.2f}"
    elif sr_last < DECAY_SUSPECT * sr_first:
        v = "SUSPECT"
        note = (f"edge softening: last-third Sharpe {sr_last:.2f} is "
                f"{sr_last / sr_first:.0%} of first-third {sr_first:.2f}")
    else:
        v, note = "PASS", f"Sharpe stable across sample: {sr_first:.2f} -> {sr_last:.2f}"
    findings.append(Finding("sharpe_decay_first_vs_last_third",
                            (round(sr_first, 2), round(sr_last, 2)), v, note))

    # --- concentration by month -------------------------------------------
    by_month = hourly.groupby(hourly.index.to_period("M")).sum()
    if total > 0 and len(by_month):
        best_share = float(by_month.max() / total)
        v = "SUSPECT" if best_share > CONC_MONTH else "PASS"
        note = f"best single month = {best_share:.0%} of total P&L ({by_month.idxmax()})"
    else:
        best_share, v, note = float("nan"), "INFO", "no positive total P&L to attribute"
    findings.append(Finding("best_month_pnl_share", round(best_share, 3), v, note))

    # --- breadth -----------------------------------------------------------
    active_months = by_month[by_month != 0.0]
    if len(active_months):
        frac_pos = float((active_months > 0).mean())
        v = "SUSPECT" if frac_pos < BREADTH_MIN else "PASS"
        note = f"{frac_pos:.0%} of {len(active_months)} active months positive"
    else:
        frac_pos, v, note = float("nan"), "INFO", "no active months"
    findings.append(Finding("frac_active_months_positive", round(frac_pos, 2), v, note))

    # --- regime dependence -------------------------------------------------
    act = daily[daily != 0.0].rename("pnl").to_frame().join(regime[["vol", "turnover"]]).dropna()
    c_vol = float(act["pnl"].corr(act["vol"])) if len(act) > 5 else float("nan")
    c_tno = float(act["pnl"].corr(act["turnover"])) if len(act) > 5 else float("nan")
    worst = np.nan if (np.isnan(c_vol) and np.isnan(c_tno)) else max(abs(c_vol), abs(c_tno))
    if np.isnan(worst):
        v, note = "INFO", "too few active days to test regime dependence"
    elif worst > REGIME_CORR:
        v = "SUSPECT"
        note = f"P&L is a regime bet: corr(vol)={c_vol:+.2f}, corr(turnover)={c_tno:+.2f}"
    else:
        v = "PASS"
        note = f"regime-neutral: corr(vol)={c_vol:+.2f}, corr(turnover)={c_tno:+.2f}"
    findings.append(Finding("pnl_regime_correlation",
                            (round(c_vol, 2), round(c_tno, 2)), v, note))

    verdicts = [f.verdict for f in findings]
    section_verdict = "FAIL" if "FAIL" in verdicts else \
                      "SUSPECT" if "SUSPECT" in verdicts else "PASS"
    result = SectionResult("shelf_life", section_verdict, findings)
    result.conditions = _suggested_conditions(section_verdict, findings)
    return result


def shelf_life(registry_entry: dict, blotter: pd.DataFrame,
               market: dict) -> SectionResult:
    """The section contract: findings + one PASS/SUSPECT/FAIL verdict."""
    return _verdict(registry_entry,
                    _daily_pnl(blotter, registry_entry),
                    _hourly_pnl(blotter),
                    market_regime(market))


def _suggested_conditions(verdict, findings) -> list[str]:
    if verdict == "PASS":
        return []
    conds = []
    for f in findings:
        if f.verdict in ("SUSPECT", "FAIL") and "decay" in f.name:
            conds.append("Monitor rolling 60-day Sharpe live; kill if it holds below 0 "
                         "for 20 consecutive trading days.")
        if f.verdict in ("SUSPECT", "FAIL") and "regime" in f.name:
            conds.append("Size down outside the regime where the edge concentrates; "
                         "re-review after one full volatility cycle.")
        if f.verdict in ("SUSPECT", "FAIL") and "month" in f.name:
            conds.append("Treat headline Sharpe as concentration-inflated; "
                         "paper-trade one more quarter before scaling.")
    return conds or ["Paper-trade one further quarter before funding."]


# ===========================================================================
# Interactive HTML report
# ===========================================================================
def _f(x) -> float | None:
    """JSON-safe float (NaN/inf -> None)."""
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def _payload(submission: str, blotter: pd.DataFrame, market: dict,
             registry_entry: dict) -> dict:
    daily = _daily_pnl(blotter, registry_entry)
    hourly = _hourly_pnl(blotter)
    regime = market_regime(market)
    result = _verdict(registry_entry, daily, hourly, regime)

    days = [d.strftime("%Y-%m-%d") for d in daily.index]
    hourly_recs = [[ts.strftime("%Y-%m-%d"), int(ts.hour), int(ts.dayofweek),
                    round(float(v), 4)] for ts, v in hourly.items()]
    regime_by_day = {d.strftime("%Y-%m-%d"):
                     [_f(r.vol), _f(r.turnover), _f(r.ret), _f(r.bias)]
                     for d, r in regime.iterrows()}
    return {
        "submission": submission,
        "klass": registry_entry.get("class", ""),
        "verdict": result.verdict,
        "conditions": getattr(result, "conditions", []) or [],
        "claimed": registry_entry.get("claimed", {}),
        "findings": [{"name": f.name, "value": f.value, "verdict": f.verdict,
                      "note": f.note} for f in result.findings],
        "days": days,
        "hourly": hourly_recs,             # [date, hour, dow, pnl]
        "regime": regime_by_day,           # date -> [vol, turnover, ret]
        "defaults": {"win": ROLL_WIN, "bins": 6, "dropN": 0, "tcrit": TSTAT_CRIT},
    }


def _plotly_tag() -> str:
    """Inline the vendored Plotly (self-contained file); fall back to the CDN."""
    if _VENDOR_PLOTLY.exists():
        return "<script>" + _VENDOR_PLOTLY.read_text(encoding="utf-8") + "</script>"
    return f'<script src="{PLOTLY_CDN}"></script>'


ALL_SECTIONS = ["head", "readout", "matrix", "concentration",
                "regime", "rolling", "verdict"]


def render_html(submission: str, blotter: pd.DataFrame, market: dict,
                registry_entry: dict, out_path: str | Path,
                sections: list[str] | None = None) -> Path:
    """Render the report. `sections` (a subset of ALL_SECTIONS) selects which
    blocks to keep; None = all. The JS is section-agnostic (every draw is a
    no-op when its target is absent), so any subset renders on its own."""
    payload = _payload(submission, blotter, market, registry_entry)
    tmpl = _TEMPLATE
    if sections is not None:
        keep = set(sections)
        tmpl = re.sub(r"<!--S:([a-z]+)-->.*?<!--/S-->",
                      lambda m: m.group(0) if m.group(1) in keep else "",
                      tmpl, flags=re.S)
        # renumber the kept section headings sequentially (e.g. "1a ·", "3 ·" -> "1 ·", "2 ·")
        n = [0]

        def _renum(m):
            n[0] += 1
            return f"{m.group(1)}{n[0]}{m.group(2)}"

        tmpl = re.sub(r"(<h[34][^>]*>)\d+[a-z]?(\s*·)", _renum, tmpl)
    html = (tmpl
            .replace("%%PLOTLY%%", _plotly_tag())
            .replace("%%TITLE%%", submission)
            .replace("%%PAYLOAD%%", json.dumps(payload)))
    out_path = Path(out_path)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# The template is a plain string (not an f-string) so the JS braces survive.
# Only %%PLOTLY%%, %%TITLE%% and %%PAYLOAD%% are substituted.
_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shelf-life · %%TITLE%%</title>
%%PLOTLY%%
<style>
  :root{--fg:#1c2230;--mut:#667;--line:#e3e7ee;--pass:#2e8b57;--susp:#e8a33d;--fail:#d1495b;--info:#888;--blue:#3b7dd8}
  body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);
       max-width:1180px;margin:22px auto;padding:0 18px;background:#fff}
  h1{margin:0 0 2px} h3{margin:26px 0 6px;border-bottom:1px solid var(--line);padding-bottom:4px}
  h4.subh{margin:24px 0 8px;font-size:15px;color:var(--fg);border-bottom:1px dashed var(--line);padding-bottom:3px}
  .sub{color:var(--mut);margin:0 0 14px}
  .banner{display:inline-block;padding:5px 15px;border-radius:6px;color:#fff;font-weight:700;font-size:17px}
  table{border-collapse:collapse;width:100%;margin:8px 0 6px;font-size:13px}
  th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}
  th{background:#f4f6fa}
  .note{color:var(--mut);font-size:12px}
  .plotctl{display:flex;flex-wrap:wrap;gap:12px 26px;margin:0 0 8px;padding:9px 14px;
           background:#f8fafc;border:1px solid var(--line);border-radius:7px}
  .ctl{display:flex;flex-direction:column;min-width:180px}
  .ctl label{font-weight:600;font-size:12px;margin-bottom:2px}
  .ctl input{width:100%}
  .ctl select{padding:3px 6px;font-size:13px;border:1px solid var(--line);border-radius:5px;background:#fff}
  .ctl .val{color:var(--blue);font-weight:700}
  .readout{display:flex;flex-wrap:wrap;gap:10px 26px;background:#f8fafc;border:1px solid var(--line);
           border-radius:8px;padding:10px 16px;margin:10px 0;font-size:13px}
  .readout b{color:var(--fg)} .readout span{color:var(--mut)}
  [hidden]{display:none!important}
  #p1table{max-width:560px;margin:10px 0 4px} #p1table td:nth-child(3){width:210px}
  #p1table .bar{display:inline-block;height:9px;background:var(--blue);border-radius:2px;margin-right:8px;vertical-align:middle;min-width:1px}
  .seasonlegend{display:flex;gap:18px;font-size:12px;color:var(--mut);margin:0 0 8px;padding:0 2px}
  .seasonlegend i{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle;margin-right:5px}
  .metric{margin:0 0 8px;font-size:13px;background:#f8fafc;border:1px solid var(--line);border-radius:7px;padding:9px 14px}
  .metric .tag{display:inline-block;padding:2px 11px;border-radius:5px;color:#fff;font-weight:700;margin:0 4px}
  .verdictcard{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:14px 18px;margin:6px 0 26px}
  .verdictcard .banner{font-size:18px} .verdictcard p{margin:11px 0 6px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .grid2 .full{grid-column:1 / -1}
  @media(max-width:820px){.grid2{grid-template-columns:1fr}}
  .plot{width:100%;height:320px}
</style></head><body>
<!--S:head-->
<div id="head"></div>
<!--/S-->
<!--S:readout-->
<div id="readout" class="readout"></div>
<!--/S-->
<!--S:matrix-->
<h3>1 · Seasonality of P&amp;L</h3>
<div class="plotctl">
  <div class="ctl"><label>Y axis (rows)</label><select id="s1y"></select></div>
  <div class="ctl"><label>X axis (columns)</label><select id="s1x"></select></div>
  <div class="ctl"><label>Drop best N days — <span id="s1dropV" class="val"></span></label>
    <input id="s1drop" type="range" min="0" max="20" step="1"></div>
</div>
<div id="p1matrix" class="plot" style="height:480px"></div>
<!--/S-->
<!--S:concentration-->
<h4 class="subh">1a · P&amp;L concentration across timeframes</h4>
<div class="plotctl">
  <div class="ctl"><label>Drop best N days — <span id="s1cfdropV" class="val"></span></label>
    <input id="s1cfdrop" type="range" min="0" max="20" step="1"></div>
  <div class="ctl"><label>Concentration curve — timeframe</label><select id="s1cf"></select></div>
</div>
<div id="p1table"></div>
<p class="note" style="max-width:560px">Gini (0–1) of P&amp;L across each timeframe's buckets, <b>time-weighted</b>
(each bucket by the trading time it covers, so a partial 2025 is judged against its own span), with the ×n/(n-1)
finite-sample correction so different bucket counts are comparable. Respects the drop-best-N slider.</p>
<div id="p1lorenz" class="plot"></div>
<!--/S-->
<!--S:regime-->
<h3>2 · P&amp;L vs market regime</h3>
<div class="plotctl">
  <div class="ctl"><label>Drop best N days — <span id="s2dropV" class="val"></span></label>
    <input id="s2drop" type="range" min="0" max="20" step="1"></div>
  <div class="ctl"><label>Rolling smoothing — <span id="s2rollV" class="val"></span> days</label>
    <input id="s2roll" type="range" min="1" max="60" step="1"></div>
</div>
<div class="grid2">
  <div id="p2pnlTime" class="plot"></div>
  <div id="p2bias"    class="plot"></div>
  <div id="p2volTime" class="plot"></div>
  <div id="p2skill"   class="plot"></div>
</div>
<p class="note" style="max-width:700px"><b>Skill ρ</b> = corr(daily P&amp;L, <b>volatility</b>), taking volatility as the
<b>spread between the strategy's two legs</b> (Tom's ρ = correlation of positions with the DA−IMB spread). ρ = +1 → P&amp;L
grows with the spread, i.e. the strategy captures it; ρ = 0 → P&amp;L independent of the spread; ρ &lt; 0 → losing more as the
spread widens (bleeding).</p>
<!--/S-->
<!--S:rolling-->
<h3>3 · Rolling Sharpe</h3>
<div class="plotctl">
  <div class="ctl"><label>Rolling window — <span id="s3winV" class="val"></span> days</label>
    <input id="s3win" type="range" min="20" max="120" step="5"></div>
</div>
<div id="p3metric" class="metric"></div>
<div id="p3sharpe" class="plot full"></div>
<!--/S-->
<!--S:verdict-->
<div id="finalverdict"></div>
<!--/S-->

<script>
const P = %%PAYLOAD%%;
const $ = id => document.getElementById(id);
const DOW = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
const VC = {PASS:"var(--pass)",SUSPECT:"var(--susp)",FAIL:"var(--fail)",INFO:"var(--info)"};
const ANN = Math.sqrt(365);
const fmt = (x,d=0)=> x===null||!isFinite(x) ? "–" : x.toLocaleString("en",{maximumFractionDigits:d,minimumFractionDigits:d});

// ---- header + static findings ----
(function(){
  if($("head")) $("head").innerHTML =
    `<h1>Shelf-life — ${P.submission.toUpperCase()}</h1>`+
    `<p class="sub">${P.klass} · section 4 of 6 · module verdict: `+
    `<span class="banner" style="background:${VC[P.verdict]}">${P.verdict}</span></p>`;
  if($("finalverdict")){
    const bad = P.findings.filter(f=>f.verdict==="SUSPECT"||f.verdict==="FAIL");
    const rationale = bad.length
      ? "Shelf-life concerns — "+bad.map(f=>f.note).join("; ")+"."
      : "The edge is stable across time and regime over the sample — no shelf-life red flags at the module's default thresholds.";
    $("finalverdict").innerHTML =
      `<h3>Final verdict — Shelf-life</h3>`+
      `<div class="verdictcard" style="border-left:6px solid ${VC[P.verdict]}">`+
      `<span class="banner" style="background:${VC[P.verdict]}">${P.verdict}</span>`+
      `<p>${rationale}</p>`+
      (P.conditions.length
        ? `<b>Conditions</b><ul>${P.conditions.map(c=>`<li>${c}</li>`).join("")}</ul>`
        : `<p class="note">No conditions — section passes.</p>`)+
      `</div>`;
  }
})();

// ---- stats helpers ----
const sum  = a => a.reduce((s,x)=>s+x,0);
const mean = a => a.length ? sum(a)/a.length : NaN;
function std(a){const m=mean(a);return a.length>1?Math.sqrt(sum(a.map(x=>(x-m)**2))/(a.length-1)):0;}
function sharpe(a){const s=std(a);return s>0?mean(a)/s*ANN:0;}
function skew(a){const n=a.length,m=mean(a),s=std(a);
  if(n<3||s===0)return null;
  const g1=sum(a.map(x=>((x-m)/s)**3))/n;
  return g1*Math.sqrt(n*(n-1))/(n-2);}
function corr(x,y){const n=x.length,mx=mean(x),my=mean(y);
  let sxy=0,sx=0,sy=0;for(let i=0;i<n;i++){sxy+=(x[i]-mx)*(y[i]-my);sx+=(x[i]-mx)**2;sy+=(y[i]-my)**2;}
  return sx>0&&sy>0?sxy/Math.sqrt(sx*sy):NaN;}
function rolling(a,win,fn){const o=Array(a.length).fill(null);
  for(let i=win-1;i<a.length;i++)o[i]=fn(a.slice(i-win+1,i+1));return o;}

// ---- derive series from raw hourly, honouring drop-best-N ----
function dailyMap(){                       // date -> total P&L (all days that traded)
  const m={};for(const r of P.hourly){m[r[0]]=(m[r[0]]||0)+r[3];}return m;}
function droppedSet(dropN){
  if(dropN<=0)return new Set();
  const m=dailyMap();
  return new Set(P.days.map(d=>[m[d]||0,d]).sort((a,b)=>b[0]-a[0]).slice(0,dropN).map(x=>x[1]));
}
function build(dropN){
  const dropped=droppedSet(dropN);
  const m=dailyMap();
  const daily=P.days.map(d=>dropped.has(d)?0:(m[d]||0));
  const hourly=P.hourly.filter(r=>!dropped.has(r[0]));
  return {daily,hourly,dropped};
}

// ---- render everything ----
const BLUE="#3b7dd8",RED="#d1495b",AMBER="#e8a33d";
const baseLayout=t=>({title:{text:t,font:{size:13}},margin:{l:55,r:15,t:34,b:40},
  font:{size:11},showlegend:false,paper_bgcolor:"#fff",plot_bgcolor:"#fff"});
const CFG={displayModeBar:false,responsive:true};

const days=P.days;
const BASE=build(0);                       // the true strategy, no days dropped
const col=v=>v>=0?BLUE:RED;

// ----- plot 1 · seasonality pivot (dropdowns: X & Y dimension; slider: drop best N) -----
const MON=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
// Matrix (X/Y) keeps its original richer set; the Gini part (table + curve) uses calendar-time buckets only
const MATRIX_DIMS=[["year","year"],["month","month (calendar)"],["monthofyear","month of year"],
                   ["season","season"],["weekcal","week (calendar)"],["week","week of year"],
                   ["weekday","weekday"],["hourofday","hour of day"]];
const CF_DIMS=[["year","years"],["seasoncal","seasons"],["month","months"],
               ["weekcal","weeks"],["day","days"],["hour","hours"],["min15","15 min"]];
const SEASON_ORD={winter:1,spring:2,summer:3,autumn:4};
function isoWeek(s){const d=new Date(s+"T00:00:00Z"),day=(d.getUTCDay()+6)%7;
  d.setUTCDate(d.getUTCDate()-day+3);const f=new Date(Date.UTC(d.getUTCFullYear(),0,4));
  return 1+Math.round(((d-f)/864e5-3+((f.getUTCDay()+6)%7))/7);}
const cap=s=>s[0].toUpperCase()+s.slice(1);
function dimVal(name,rec){const date=rec[0],hour=rec[1],dow=rec[2];
  const yr=+date.slice(0,4), mo=+date.slice(5,7);
  switch(name){
    case "year":        return {o:yr, lab:date.slice(0,4)};                                  // 2025
    case "month":       return {o:yr*12+mo, lab:MON[mo-1]+" "+String(yr).slice(2)};          // Jan 25 (calendar)
    case "monthofyear": return {o:mo, lab:MON[mo-1]};                                        // Jan..Dec (all years)
    case "season":      {const s=seasonOf(mo); return {o:SEASON_ORD[s], lab:cap(s)};}        // season of year (aggregated)
    case "seasoncal":   {const s=seasonOf(mo), sy=(mo===12)?yr+1:yr;                          // Spring 2025 (calendar)
                         return {o:sy*4+SEASON_ORD[s], lab:cap(s)+" "+sy};}
    case "week":        {const w=isoWeek(date); return {o:w, lab:"W"+w};}                     // ISO week of year
    case "weekcal":     {const w=isoWeek(date); return {o:yr*54+w, lab:"W"+w+" "+String(yr).slice(2)};}  // W1 25 (calendar)
    case "day":         return {o:Date.parse(date), lab:date};                                // 2025-01-01
    case "weekday":     return {o:dow, lab:DOW[dow]};                                         // Mon..Sun
    case "hourofday":   return {o:hour, lab:"PH"+(hour+1)};                                   // PH1..PH24 (00:00–01:00 = PH1)
    case "min15":       {const q=rec[4]||0;                                                   // 15-min slot; hourly-delivery data has no sub-hour split (q=0)
                         return {o:Date.parse(date)+hour*36e5+q*9e5, lab:date+" "+String(hour).padStart(2,"0")+":"+String(q*15).padStart(2,"0")};}
    default:            return {o:Date.parse(date)+hour*36e5, lab:date+" "+String(hour).padStart(2,"0")+"h"};  // hour (calendar)
  }}
const TOT="Σ total", INNER_TEXT_MAX=180;
const kfmt=v=>Math.abs(v)>=1000?(v/1000).toFixed(1)+"k":(v||0).toFixed(0);
function etaRatio(records, dimName){    // between-timeframe std ÷ total std of P&L, in [0,1]
  const N=records.length; if(N<2) return null;
  const gm=records.reduce((s,r)=>s+r[3],0)/N;
  const groups={}; let total=0;
  for(const r of records){total+=(r[3]-gm)**2; const k=dimVal(dimName,r).lab; (groups[k]=groups[k]||[]).push(r[3]);}
  if(total<=0) return 0;
  let between=0;
  for(const k in groups){const g=groups[k], mg=g.reduce((s,v)=>s+v,0)/g.length; between+=g.length*(mg-gm)**2;}
  return Math.sqrt(between/total);
}
function bucketLorenz(records, dimName){  // time-weighted Lorenz: x = share of time traded (record count), buckets ordered by P&L per unit time
  const bmap={}, wmap={};
  for(const r of records){const k=dimVal(dimName,r).lab; bmap[k]=(bmap[k]||0)+r[3]; wmap[k]=(wmap[k]||0)+1;}
  const keys=Object.keys(bmap), nB=keys.length;
  if(nB<2) return null;
  keys.sort((a,b)=>(bmap[a]/wmap[a])-(bmap[b]/wmap[b]));          // ascending P&L density
  const totW=keys.reduce((s,k)=>s+wmap[k],0), tot=keys.reduce((s,k)=>s+bmap[k],0);
  if(tot===0||totW===0) return null;
  const lx=[0], ly=[0]; let cumW=0, cumP=0, lossW=0;
  for(const k of keys){cumW+=wmap[k]; cumP+=bmap[k]; lx.push(cumW/totW); ly.push(cumP/tot); if(bmap[k]<0)lossW+=wmap[k];}
  let area=0; for(let i=1;i<lx.length;i++) area+=(lx[i]-lx[i-1])*(ly[i]+ly[i-1])/2;
  // Gini can exceed 1 when some buckets are negative (losing periods) — do not cap the upper bound
  const raw=Math.max(0,1-2*area), scaled=Math.max(0,raw*nB/(nB-1));
  const minY=Math.min(...ly), minI=ly.indexOf(minY);
  const nLoss=keys.filter(k=>bmap[k]<0).length, nWin=keys.filter(k=>bmap[k]>0).length;
  return {lx, ly, raw, scaled, n:nB, nWin, nLoss, winCountPct:100*nWin/nB,
          winPct:100*(1-lossW/totW), lossW, minX:lx[minI], minY};
}
function giniOf(records, dimName){ const L=bucketLorenz(records,dimName); return L?{raw:L.raw,scaled:L.scaled,n:L.n}:null; }

function drawSeasonality(){
  if(!$("p1matrix"))return;
  const dropN=+$("s1drop").value; $("s1dropV").textContent=dropN;
  const xName=$("s1x").value, yName=$("s1y").value;
  const {hourly}=build(dropN);
  const cells={}, xm=new Map(), ym=new Map();
  for(const r of hourly){
    const x=dimVal(xName,r), y=dimVal(yName,r);
    xm.set(x.lab,x.o); ym.set(y.lab,y.o);
    cells[y.lab+"|"+x.lab]=(cells[y.lab+"|"+x.lab]||0)+r[3];
  }
  const xl=[...xm.entries()].sort((a,b)=>a[1]-b[1]).map(e=>e[0]);
  const yl=[...ym.entries()].sort((a,b)=>a[1]-b[1]).map(e=>e[0]);
  // inner matrix (rows=Y, cols=X); Σ total row + column left blank in the heatmap, filled by annotations
  const SP="⠀";                       // blank spacer category between cells and totals
  const nx=xl.length;
  const z=[], colTot={}, rowTot={}; let grand=0;
  for(const y of yl){const row=[]; let rt=0;
    for(const x of xl){const v=cells[y+"|"+x]||0; row.push(v); rt+=v; colTot[x]=(colTot[x]||0)+v;}
    row.push(null,null); z.push(row); rowTot[y]=rt; grand+=rt;}   // spacer + Σ cols blank here
  z.push(Array(nx+2).fill(null));          // spacer row
  z.push(Array(nx+2).fill(null));          // Σ total row (filled by the totals layer)
  const inner=z.slice(0,yl.length).flatMap(r=>r.slice(0,nx));
  const L=Math.max(1,...inner.map(v=>Math.abs(v)));
  // the Σ total row/column dwarf individual cells, so colour them on their own scale (2nd layer)
  const z2=[];
  for(let i=0;i<yl.length;i++){const row=Array(nx).fill(null); row.push(null,rowTot[yl[i]]); z2.push(row);}
  z2.push(Array(nx+2).fill(null));                        // spacer row
  z2.push([...xl.map(x=>colTot[x]), null, grand]);        // Σ total row
  const Lt=Math.max(1,...xl.map(x=>Math.abs(colTot[x])),...yl.map(y=>Math.abs(rowTot[y])),Math.abs(grand));
  // annotate by category INDEX (numbers), not label — a numeric-string label like
  // "2025" would otherwise be read as the number 2025 and land off-axis
  const xArr=[...xl,SP,TOT], yArr=[...yl,SP,TOT];
  const xIdx={}; xArr.forEach((l,i)=>xIdx[l]=i);
  const yIdx={}; [...yArr].reverse().forEach((l,i)=>yIdx[l]=i);   // y axis renders reversed
  const txt=(x,y,v,scale,big)=>({x:xIdx[x],y:yIdx[y],text:kfmt(v),showarrow:false,
    font:{size:big?10:9,color:Math.abs(v)>0.55*scale?"#fff":"#222"}});
  const ann=[];
  xl.forEach(x=>ann.push(txt(x,TOT,colTot[x],Lt,true)));
  yl.forEach(y=>ann.push(txt(TOT,y,rowTot[y],Lt,true)));
  ann.push(txt(TOT,TOT,grand,Lt,true));
  if(xl.length*yl.length<=INNER_TEXT_MAX)
    for(let i=0;i<yl.length;i++)for(let j=0;j<xl.length;j++)
      ann.push(txt(xl[j],yl[i],z[i][j],L,false));
  const xA=[...xl,SP,TOT], yA=[...yl,SP,TOT];
  Plotly.react("p1matrix",[
    {type:"heatmap",z,x:xA,y:yA,zmin:-L,zmax:L,colorscale:"RdBu",reversescale:true,zsmooth:false,xgap:1,ygap:1,
     colorbar:{title:"cell",thickness:10,len:.48,y:.76},
     hovertemplate:"%{y} · %{x}<br>%{z:.0f} EUR<extra></extra>"},
    {type:"heatmap",z:z2,x:xA,y:yA,zmin:-Lt,zmax:Lt,colorscale:"RdBu",reversescale:true,zsmooth:false,xgap:1,ygap:1,
     colorbar:{title:"total",thickness:10,len:.48,y:.24},
     hovertemplate:"%{y} · %{x}<br>total %{z:.0f} EUR<extra></extra>"},
    {type:"scatter",mode:"markers",x:[SP],y:[SP],marker:{opacity:0,size:1},hoverinfo:"skip",showlegend:false}],
    {...baseLayout(`P&L by ${yName} (rows) × ${xName} (cols), EUR`),
     height:490, margin:{l:74,r:15,t:64,b:42},
     title:{text:`P&L by ${yName} (rows) × ${xName} (cols), EUR`,font:{size:13},y:0.985,yanchor:"top"},
     xaxis:{type:"category",categoryorder:"array",categoryarray:xA,range:[-0.5,xA.length-0.5],
            side:"top",tickangle:xl.length>14?-45:0},
     yaxis:{type:"category",categoryorder:"array",categoryarray:[...yA].reverse(),range:[-0.5,yA.length-0.5]},
     annotations:ann},CFG);

}

// ----- section 1a · concentration table + Lorenz curve (own drop slider) -----
function drawConcentration(){
  if(!$("p1table"))return;
  const dropN = $("s1cfdrop") ? +$("s1cfdrop").value : ($("s1drop")?+$("s1drop").value:0);
  if($("s1cfdrop")) $("s1cfdropV").textContent=dropN;
  const {hourly}=build(dropN);
  const TFS=[["year","yearly"],["seasoncal","seasonal"],["month","monthly"],
             ["weekcal","weekly"],["day","daily"],["hour","hourly"],["min15","15-min"]];
  const trows=TFS.map(([k,lab])=>{const g=giniOf(hourly,k), bar=g==null?0:Math.min(200,Math.round(g.scaled*100));
    return `<tr><td>${lab}</td><td>${g==null?"–":g.raw.toFixed(2)}</td>`+
           `<td><span class="bar" style="width:${bar}px"></span>${g==null?"–":g.scaled.toFixed(2)}</td></tr>`;}).join("");
  $("p1table").innerHTML=`<table><tr><th>timeframe</th><th>Gini</th><th>Gini scaled</th></tr>${trows}</table>`;
  if(!$("p1lorenz"))return;
  const cf=$("s1cf").value, cflab=(CF_DIMS.find(d=>d[0]===cf)||[cf,cf])[1];
  const L=bucketLorenz(hourly, cf);
  if(!L){Plotly.react("p1lorenz",[],{...baseLayout("not enough buckets")},CFG); return;}
  const winTxt=L.nLoss===0?`all ${cflab} profitable — 100% win rate`
                          :`win rate ${L.winCountPct.toFixed(0)}% of ${cflab} (${L.nWin}/${L.n})`;
  Plotly.react("p1lorenz",[
    {type:"scatter",mode:"lines",x:[0,1],y:[0,1],line:{color:"#bbb",dash:"dash",width:1},hoverinfo:"skip"},
    {type:"scatter",mode:"lines+markers",x:L.lx,y:L.ly,line:{color:BLUE,width:2},marker:{size:4},
     hovertemplate:"%{x:.0%} of time → %{y:.0%} of P&L<extra></extra>"},
    {type:"scatter",mode:"markers",x:[L.minX],y:[L.minY],marker:{color:RED,size:9},
     hovertemplate:`minimum ${(L.minY*100).toFixed(1)}% of P&L<extra></extra>`}],
    {...baseLayout(`P&L concentration across ${cflab} (${L.n} buckets, time-weighted) — Gini ${L.raw.toFixed(2)} · scaled ${L.scaled.toFixed(2)}`),
     xaxis:{title:"cumulative share of time traded",range:[0,1]},
     yaxis:{title:"cumulative share of P&L",range:[Math.min(0,L.minY)-0.03,1.03]},
     annotations:[{x:L.minX,y:L.minY,text:winTxt,showarrow:true,arrowhead:0,ax:38,ay:L.lossW?-24:-14,
                   font:{size:10,color:L.lossW===0?"#2e8b57":"#d1495b"}}]},CFG);
}

// ----- plot 2 · regime scatter + binned mean (sliders: bins, drop best N) -----
function drawRegime(){
  if(!$("p2volTime"))return;
  const dropN=+$("s2drop").value, roll=+$("s2roll").value;
  $("s2dropV").textContent=dropN; $("s2rollV").textContent=roll;
  const {daily}=build(dropN);

  // ---- market volatility over time (rolling-smoothed) ----
  const rmean=a=>{const o=Array(a.length).fill(null), h=(roll-1)>>1;
    for(let i=0;i<a.length;i++){const lo=i-h, hi=lo+roll;
      if(lo<0||hi>a.length) continue;                       // partial window at head/tail -> drop
      let s=0,c=0; for(let j=lo;j<hi;j++) if(a[j]!=null){s+=a[j];c++;}
      o[i]=c?s/c:null;}
    return o;};
  const tag=roll>1?`${roll}-day mean`:"daily";
  const volS=days.map(d=>{const r=P.regime[d];return r?r[0]:null;});
  const biasS=days.map(d=>{const r=P.regime[d];return r?r[3]:null;});
  const zero0=[{type:"line",xref:"paper",x0:0,x1:1,y0:0,y1:0,line:{color:"#000",width:.6}}];
  Plotly.react("p2pnlTime",[{type:"scatter",mode:"lines",x:days,y:rmean(daily),line:{color:BLUE,width:1.5}}],
    {...baseLayout(`daily P&L over time (${tag})`),yaxis:{title:"P&L (EUR/day)"},shapes:zero0},CFG);
  Plotly.react("p2bias",[{type:"scatter",mode:"lines",x:days,y:rmean(biasS),line:{color:"#c86e2a",width:1.5}}],
    {...baseLayout(`market bias (${tag}) — intraday mid − day-ahead`),yaxis:{title:"bias (EUR/MWh)"},shapes:zero0},CFG);
  Plotly.react("p2volTime",[{type:"scatter",mode:"lines",x:days,y:rmean(volS),line:{color:"#8a5cd1",width:1.5}}],
    {...baseLayout(`market volatility (${tag})`),yaxis:{title:"vol (std of mid)"}},CFG);

  // ---- skill ρ over time: corr(daily P&L, volatility) — volatility = spread between the strategy's two legs ----
  const rollCorr=(A,Bv,win)=>{const o=Array(A.length).fill(null), h=(win-1)>>1;
    for(let i=0;i<A.length;i++){const lo=i-h, hi=lo+win; if(lo<0||hi>A.length)continue;
      const xs=[],ys=[]; for(let j=lo;j<hi;j++) if(A[j]!=null&&Bv[j]!=null){xs.push(A[j]);ys.push(Bv[j]);}
      const c=xs.length>2?corr(xs,ys):NaN; o[i]=isFinite(c)?c:null;}
    return o;};
  // full-sample mean skill + bootstrapped certainty band (95% CI of a roll-day ρ estimate)
  const sx=[], sy=[];
  for(let i=0;i<days.length;i++) if(daily[i]!=null&&volS[i]!=null){sx.push(daily[i]);sy.push(volS[i]);}
  const rhoFull=sx.length>2?corr(sx,sy):NaN, nP=sx.length, Bc=500, boot=[];
  if(nP>2) for(let b=0;b<Bc;b++){const rx=[],ry=[];
    for(let k=0;k<roll;k++){const i=(Math.random()*nP)|0; rx.push(sx[i]); ry.push(sy[i]);}
    const c=corr(rx,ry); boot.push(isFinite(c)?c:0);}
  boot.sort((a,b)=>a-b);
  const clo=boot.length?boot[(0.025*Bc)|0]:null, chi=boot.length?boot[(0.975*Bc)|0]:null;
  const skillShapes=[{type:"line",xref:"paper",x0:0,x1:1,y0:0,y1:0,line:{color:"#000",width:.6}}];
  if(clo!=null){skillShapes.unshift(
    {type:"rect",xref:"paper",x0:0,x1:1,yref:"y",y0:clo,y1:chi,fillcolor:"rgba(46,139,87,0.14)",line:{width:0},layer:"below"},
    {type:"line",xref:"paper",x0:0,x1:1,y0:rhoFull,y1:rhoFull,line:{color:"#2e8b57",dash:"dash",width:1}});}
  Plotly.react("p2skill",[{type:"scatter",mode:"lines",x:days,y:rollCorr(daily,volS,roll),line:{color:"#2e8b57",width:1.6}}],
    {...baseLayout(`skill ρ over time — corr(daily P&L, volatility) · mean ${isFinite(rhoFull)?rhoFull.toFixed(2):"–"} · ${roll}-day 95% CI`),
     yaxis:{title:"skill ρ",range:[-1,1]},shapes:skillShapes,
     annotations:clo==null?[]:[{xref:"paper",x:0.01,y:rhoFull,text:`mean skill ${rhoFull.toFixed(2)} · 95% CI [${clo.toFixed(2)}, ${chi.toFixed(2)}]`,
                   showarrow:false,font:{size:10,color:"#2e8b57"},yshift:8,xanchor:"left"}]},CFG);
}

// ----- plot 3 · rolling Sharpe & t-stat (slider: window) -----
const seasonOf=m=>[12,1,2].includes(m)?"winter":[3,4,5].includes(m)?"spring":
                  [6,7,8].includes(m)?"summer":"autumn";
function bootSharpeCI(pop, m){   // 95% CI of a Sharpe estimated over m draws from pop — band width scales with the window m
  const n=pop.length, B=500, out=[]; let sum=0;
  for(let b=0;b<B;b++){let s=0,q=0;
    for(let k=0;k<m;k++){const v=pop[(Math.random()*n)|0]; s+=v; q+=v*v;}
    const mn=s/m, sd=Math.sqrt(Math.max(0,(q-s*s/m)/(m-1))), sr=sd>0?mn/sd*ANN:0;
    out.push(sr); sum+=sr;}
  out.sort((a,b)=>a-b);
  return {lo:out[(0.025*B)|0], mid:sum/B, hi:out[(0.975*B)|0]};
}
function drawRolling(){
  if(!$("p3sharpe"))return;
  const win=+$("s3win").value; $("s3winV").textContent=win;
  const daily=BASE.daily;
  const srArr=rolling(daily,win,sharpe);
  const full=sharpe(daily);
  const fullCI=bootSharpeCI(daily, win);   // 95% CI of a win-day Sharpe estimate — band width tracks the window, centred on the dashed line
  // hard metric: share of rolling-Sharpe points outside the CI band (>5% -> FAIL)
  if($("p3metric")){
    const valid=srArr.filter(v=>v!=null);
    const outside=valid.filter(v=>v<fullCI.lo||v>fullCI.hi).length;
    const pct=valid.length?100*outside/valid.length:0, failed=pct>5;
    $("p3metric").innerHTML=
      `Rolling Sharpe outside the ${win}-day 95% CI: <b>${pct.toFixed(1)}%</b> (${outside} of ${valid.length} points)`+
      `<span class="tag" style="background:${failed?VC.FAIL:VC.PASS}">${failed?"FAIL":"PASS"}</span>`+
      `<span class="note">threshold 5% — more points outside than sampling noise allows means the edge isn't stationary</span>`;
  }
  Plotly.react("p3sharpe",[
    {type:"scatter",mode:"lines",x:days,y:srArr,line:{color:BLUE,width:1.6}}],
    {...baseLayout(`rolling ${win}-day Sharpe · 95% CI band for a ${win}-day estimate`),yaxis:{title:"Sharpe"},
     shapes:[{type:"rect",xref:"paper",x0:0,x1:1,yref:"y",y0:fullCI.lo,y1:fullCI.hi,
              fillcolor:"rgba(136,136,136,0.16)",line:{width:0},layer:"below"},
             {type:"line",xref:"paper",x0:0,x1:1,y0:full,y1:full,line:{color:"#888",dash:"dash",width:1}},
             {type:"line",xref:"paper",x0:0,x1:1,y0:0,y1:0,line:{color:"#000",width:.6}}],
     annotations:[{xref:"paper",x:0.01,y:full,text:`full-sample ${full.toFixed(2)} · ${win}-day 95% CI [${fullCI.lo.toFixed(1)}, ${fullCI.hi.toFixed(1)}]`,
                   showarrow:false,font:{size:10,color:"#888"},yshift:8,xanchor:"left"}]},CFG);
}


// ----- readout (base strategy, computed once) -----
(function drawReadout(){
  if(!$("readout"))return;
  const daily=BASE.daily, n=days.length;
  const first=daily.slice(0,Math.floor(n/3)), last=daily.slice(Math.floor(2*n/3));
  const act=daily.filter(x=>x!==0);
  $("readout").innerHTML=[
    ["total P&L",`${fmt(sum(daily))} EUR`],
    ["full Sharpe",sharpe(daily).toFixed(2)],
    ["first⅓ → last⅓ Sharpe",`${sharpe(first).toFixed(2)} → ${sharpe(last).toFixed(2)}`],
    ["active days",`${act.length} / ${n} (${fmt(100*act.length/n)}%)`],
    ["best day",`${fmt(Math.max(...daily))} EUR`],
  ].map(([k,v])=>`<span>${k}: <b>${v}</b></span>`).join("");
})();

// ----- wire each control to its plot(s), then draw all (all null-safe for partial pages) -----
const on=(id,ev,fn)=>{const e=$(id); if(e)e.addEventListener(ev,fn);};
const setv=(id,v)=>{const e=$(id); if(e)e.value=v;};
on("s1drop","input",drawSeasonality);
on("s1x","change",drawSeasonality);
on("s1y","change",drawSeasonality);
on("s1cfdrop","input",drawConcentration);
on("s1cf","change",drawConcentration);
on("s2drop","input",drawRegime);
on("s2roll","input",drawRegime);
on("s3win","input",drawRolling);
(function init(){
  const mopts=MATRIX_DIMS.map(([v,l])=>`<option value="${v}">${l}</option>`).join("");
  const copts=CF_DIMS.map(([v,l])=>`<option value="${v}">${l}</option>`).join("");
  for(const id of ["s1x","s1y"]){const e=$(id); if(e)e.innerHTML=mopts;}
  {const e=$("s1cf"); if(e)e.innerHTML=copts;}
  setv("s1x","weekday"); setv("s1y","monthofyear"); setv("s1cf","month");
  setv("s1drop",P.defaults.dropN); setv("s1cfdrop",P.defaults.dropN);
  setv("s2drop",P.defaults.dropN); setv("s2roll",10);
  setv("s3win",P.defaults.win);
  drawSeasonality(); drawConcentration(); drawRegime(); drawRolling();
})();
</script>
</body></html>"""


# ===========================================================================
# The one function to run
# ===========================================================================
def generate(submission: str = "windfall",
             data_dir: str | Path = "../data",
             blotter_dir: str | Path = "../blotters",
             registry_dir: str | Path = "../registry",
             out_dir: str | Path = ".",
             sections: list[str] | None = None,
             tag: str = "") -> Path:
    """Load everything for one submission and write shelf_life[_<tag>]_<submission>.html.
    Pass `sections` (subset of ALL_SECTIONS) for a partial report, e.g.
    sections=['head','concentration','rolling'] for just 1a + section 3."""
    market = load_market(data_dir)
    registry = load_registry(registry_dir)
    if submission not in registry:
        raise KeyError(f"{submission!r} not in registry: {sorted(registry)}")
    blotter = load_blotter(Path(blotter_dir) / f"{submission}-blotter.csv")
    entry = registry[submission]
    out = Path(out_dir) / f"shelf_life{('_'+tag) if tag else ''}_{submission}.html"
    path = render_html(submission, blotter, market, entry, out, sections=sections)
    print(f"wrote {path}  (module verdict: {shelf_life(entry, blotter, market).verdict})")
    return path


if __name__ == "__main__":
    generate(sys.argv[1] if len(sys.argv) > 1 else "windfall")
