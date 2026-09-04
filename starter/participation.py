"""PARTICIPATION — the leakage test for books with no buy/sell to explain.

`lineage.py` tests DIRECTION: given a trade, did buying-versus-selling line up
with forecast news that had not been published yet? That needs a book with
two sides. bounceback and spikecatcher only ever open buys, and every trade
is capped at 10 MW, so direction and size are both constant — the only thing
that varies is WHICH delivery hours they chose to trade at all.

So the response here is participation: over every product in the market,
traded or not, does the choice respond to news published only afterwards?

    ev = participation_test(blotter, forecasts, "wind")

WHY THIS TEST NEEDS A HISTORY GATE
----------------------------------
Direction responds to the SIGN of news, and revision signs are close to
unpredictable, so one prior revision is enough to control for and the null
is clean.

Participation responds to the MAGNITUDE of news — these strategies enter
when |revision| crosses a threshold. Magnitude is volatility-clustered: a
big revision now means a big revision next, with no leakage whatsoever. So
|next revision| predicts participation for entirely innocent reasons, and
the test only means something once the volatility regime is controlled for.

One lag is not enough to do that. |revision at the decision| is a noisy
proxy for the regime, and controlling with a noisy proxy leaves residual
confounding for |next revision| to soak up. Measured on windfall — which the
direction test clears cleanly — a single-lag control yields t = +26, and
binning it non-parametrically makes it worse, not better.

The fix is more history: several past revisions pin the regime down. The
catch is that a strategy trading at the first or second vintage of the day
HAS no history, so for those the test cannot be identified at all. That is
what MIN_HISTORY gates. It is a refusal to run, not a pass: a strategy the
gate rejects is untestable here and stays SUSPECT.

Thresholds and the gate are fixed here, before any submission is looked at.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import lineage as L
from report import Finding, SectionResult

VARIABLES = L.VARIABLES

# --- gate and thresholds, fixed in advance ---------------------------------
MIN_HISTORY = 3        # past revisions needed to pin the volatility regime
GATE_COVERAGE = 0.50   # share of products that must have that much history
MIN_N = 200            # products in the panel below which we do not run
T_SUSPECT = 3.0        # same scale as lineage.py — these are |t| on the
T_FAIL = 6.0           # unpublished-news coefficient
BETA_FAIL = 0.10


def load_products(data_dir: str | Path = "../data") -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "products.csv",
                       parse_dates=["delivery_start"])


def decision_lead(blotter: pd.DataFrame) -> float:
    """Seconds before delivery at which this strategy typically decides.

    Its opening trades define it; the same lead is then applied to every
    product in the market so untraded hours get a decision time too.
    """
    marked = L.mark_open_close(blotter)
    opens = marked[marked["is_open"]]
    return float((opens["product_delivery"] - opens["exec_ts"])
                 .dt.total_seconds().median())


def panel(blotter: pd.DataFrame, forecasts: pd.DataFrame,
          products: pd.DataFrame, variable: str) -> pd.DataFrame:
    """One row per delivery product: did it trade, and what was the news?

    Untraded products are the control group, so the panel is the whole
    market rather than only the blotter.
    """
    marked = L.mark_open_close(blotter)
    opens = marked[marked["is_open"]]
    lead = decision_lead(blotter)

    u = products[["delivery_start"]].rename(columns={"delivery_start": "target_ts"})
    u = u.drop_duplicates().copy()
    u["decision_ts"] = u["target_ts"] - pd.to_timedelta(lead, unit="s")
    u["traded"] = u["target_ts"].isin(set(opens["product_delivery"])).astype(float)

    v = L.vintages(forecasts, variable).copy()
    v["k"] = v.groupby("target_ts")["issue_ts"].rank(method="first")

    # rank of the newest vintage published at or before the decision
    live = (v.merge(u[["target_ts", "decision_ts"]], on="target_ts")
             .query("issue_ts <= decision_ts")
             .groupby("target_ts")["k"].max().rename("k0").reset_index())
    u = u.merge(live, on="target_ts", how="left")

    def at(lag: int, col: str) -> np.ndarray:
        j = u[["target_ts"]].copy()
        j["k"] = u["k0"] + lag
        return j.merge(v[["target_ts", "k", col]], on=["target_ts", "k"],
                       how="left")[col].values

    out = pd.DataFrame({"traded": u["traded"].values,
                        "level": at(0, "value_mw"),
                        "k0": u["k0"].values})
    for lag in range(0, -MIN_HISTORY, -1):          # 0, -1, -2 ...
        out[f"news{lag}"] = np.abs(at(lag, "revision"))
    out["news_next"] = np.abs(at(1, "revision"))
    return out


def participation_test(blotter: pd.DataFrame, forecasts: pd.DataFrame,
                       products: pd.DataFrame, variable: str) -> dict:
    """Does WHICH products it trades respond to news published afterwards?"""
    d = panel(blotter, forecasts, products, variable)
    hist_cols = [f"news{lag}" for lag in range(0, -MIN_HISTORY, -1)]
    base = {"variable": variable, "n": 0, "history": 0.0,
            "lead_min": decision_lead(blotter) / 60}

    # --- the gate -----------------------------------------------------------
    have = d[hist_cols].notna().all(axis=1)
    base["history"] = float(have.mean())
    if base["history"] < GATE_COVERAGE:
        return {**base, "untestable":
                f"only {base['history']:.0%} of products have "
                f"{MIN_HISTORY} prior revisions at its decision time"}

    d = d.dropna(subset=hist_cols + ["level", "news_next", "traded"])
    base["n"] = len(d)
    if len(d) < MIN_N:
        return {**base, "untestable": f"only {len(d)} products in the panel"}
    if d["traded"].std() == 0:
        return {**base, "untestable": "it traded every product, or none"}

    def z(s) -> np.ndarray | None:
        a = np.asarray(s, dtype=float)
        return (a - a.mean()) / a.std() if a.std() > 0 else None

    cols = ["level"] + hist_cols + ["news_next"]
    zs = [z(d[c]) for c in cols]
    y = z(d["traded"])
    if y is None or any(c is None for c in zs):
        return {**base, "untestable": "a regressor has zero variance"}

    beta, t = L._ols_t(y, np.column_stack([np.ones(len(d))] + zs))
    if not np.isfinite(t).all():
        return {**base, "untestable": "degenerate regression (non-finite t)"}

    return {**base, "beta_next": beta[-1], "t_next": t[-1],
            "controls": len(hist_cols)}


def _verdict(r: dict) -> str:
    """INFO when the gate refused. Never let an unrunnable test PASS."""
    if "untestable" in r:
        return "INFO"
    t, b = abs(r["t_next"]), abs(r["beta_next"])
    if not (np.isfinite(t) and np.isfinite(b)):
        return "INFO"
    if t >= T_FAIL and b >= BETA_FAIL:
        return "FAIL"
    if t >= T_SUSPECT:
        return "SUSPECT"
    return "PASS"


def participation(registry_entry: dict, blotter: pd.DataFrame,
                  forecasts: pd.DataFrame,
                  products: pd.DataFrame) -> SectionResult:
    findings, verdicts = [], []
    for var in VARIABLES:
        r = participation_test(blotter, forecasts, products, var)
        v = _verdict(r)
        verdicts.append(v)
        if "untestable" in r:
            note = (f"participation test not identified: {r['untestable']} "
                    f"(decides {r['lead_min']:.0f} min before delivery)")
            val = None
        else:
            note = (f"choice of delivery hours loads {r['beta_next']:+.3f} "
                    f"(t={r['t_next']:+.1f}) on the size of the next, "
                    f"unpublished {var} revision, controlling for the level "
                    f"and {r['controls']} prior revisions; n={r['n']:,}")
            val = round(float(r["t_next"]), 2)
        findings.append(Finding(f"participation_{var}", val, v, note))

    order = ["FAIL", "SUSPECT", "PASS", "INFO"]
    return SectionResult("lineage", next(v for v in order if v in verdicts),
                         findings)
