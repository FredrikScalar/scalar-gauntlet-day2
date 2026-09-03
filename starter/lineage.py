"""LINEAGE — was the test honest?

First test in the section: information leakage against the forecast tape.

For every trade we locate its position in the vintage timeline of each
forecast variable:

    ... ---- prev ---- live --|-- next ----  ...
                              |
             past_revision    |    future_revision
             = live - prev    |    = next - live
                              |
                           exec_ts

`past_revision` was published before the trade and is legitimate signal.
`future_revision` had not been published yet, so no honest strategy can
know it.

The naive test — correlate direction against `future_revision` alone — is
wrong. Weather runs have momentum, so consecutive revisions are themselves
correlated; a perfectly honest strategy trading `past_revision` inherits an
apparent loading on `future_revision` for free. So we regress direction on
BOTH and read the *partial* effect of the unknowable one:

    direction ~ a + b1 * past_revision + b2 * future_revision

b1 is the declared edge. b2 is time travel.

Thresholds are fixed here, before any submission is looked at, and apply
unchanged to all seven.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from report import Finding, SectionResult

VARIABLES = ("wind", "solar", "load")

# --- verdict thresholds, fixed in advance ----------------------------------
T_SUSPECT = 3.0      # |t| on the future-revision coefficient
T_FAIL = 6.0
BETA_FAIL = 0.10     # standardised effect size required alongside T_FAIL
MIN_N = 30           # below this the test is not run at all
MIN_COVERAGE = 0.20  # opening trades needing a testable vintage neighbourhood


def mark_open_close(blotter: pd.DataFrame) -> pd.DataFrame:
    """Flag each trade as opening or closing its product's position.

    This is not cosmetic. Every position is opened AND closed through
    trades, so the closing trade carries the mirror image of the entry
    direction. Pooling them cancels the signal almost exactly — windfall's
    correlation with the published wind revision is -0.99 on opens, +0.99
    on closes, and 0.00 pooled. Any direction-based test must run on opens.

    A trade opens if it moves the product's position away from zero (or the
    book was flat); it closes if it moves back toward zero.
    """
    b = blotter.sort_values(["product_delivery", "exec_ts"]).copy()
    signed = np.where(b["side"].eq("BUY"), b["qty_mw"], -b["qty_mw"])
    b["_signed"] = signed
    pos_before = (b.groupby("product_delivery")["_signed"].cumsum()
                  - b["_signed"])
    b["is_open"] = (pos_before.abs().eq(0)
                    | (np.sign(b["_signed"]) == np.sign(pos_before)))
    return b.drop(columns="_signed")


def load_forecasts(data_dir: str | Path = "../data") -> pd.DataFrame:
    """Just the forecast tape — the leakage test needs no order book."""
    return pd.read_csv(Path(data_dir) / "forecasts.csv",
                       parse_dates=["issue_ts", "target_ts"])


def vintages(forecasts: pd.DataFrame, variable: str) -> pd.DataFrame:
    """One row per (target_ts, issue_ts) carrying that vintage's revision.

    A vintage's `revision` is its own value minus the previous vintage for
    the same delivery hour — i.e. the news that vintage carried.
    """
    v = (forecasts[forecasts["variable"] == variable]
         .sort_values(["target_ts", "issue_ts"])
         .copy())
    v["revision"] = v.groupby("target_ts")["value_mw"].diff()
    return v[["issue_ts", "target_ts", "value_mw", "revision"]]


def attach_revisions(blotter: pd.DataFrame, forecasts: pd.DataFrame,
                     variable: str) -> pd.DataFrame:
    """Give every trade the published and the not-yet-published revision.

    Point-in-time throughout: `live` is the newest vintage with
    `issue_ts <= exec_ts` (exact matches count as knowable), `next` is the
    earliest with `issue_ts > exec_ts` (exact matches explicitly excluded).
    """
    v = vintages(forecasts, variable).sort_values("issue_ts")
    b = blotter.sort_values("exec_ts").copy()

    live = pd.merge_asof(
        b, v.rename(columns={"revision": "past_revision"}),
        left_on="exec_ts", right_on="issue_ts",
        left_by="product_delivery", right_by="target_ts",
        direction="backward", allow_exact_matches=True)

    # The next vintage's own revision IS next - live, so we can read it off
    # directly rather than differencing values again.
    nxt = pd.merge_asof(
        b, v.rename(columns={"revision": "future_revision"}),
        left_on="exec_ts", right_on="issue_ts",
        left_by="product_delivery", right_by="target_ts",
        direction="forward", allow_exact_matches=False)

    out = live[["exec_ts", "product_delivery", "side", "past_revision"]].copy()
    out["future_revision"] = nxt["future_revision"].values
    out["direction"] = np.where(out["side"].eq("BUY"), 1.0, -1.0)
    return out


def testable(blotter: pd.DataFrame, forecasts: pd.DataFrame,
             variable: str) -> tuple[pd.DataFrame, float]:
    """Opening trades that sit inside the vintage timeline, plus coverage.

    Trades at the edges (before the first revision exists, or after the
    last vintage is issued) have no neighbourhood to test and are dropped
    — but the fraction dropped is reported, not swallowed. blackbox enters
    at gate-10h and exits at gate-15min, so its coverage is 0%: the
    revision test simply cannot speak to it.
    """
    # Use the frame mark_open_close returns, not a mask against the original:
    # it re-sorts by (product, exec_ts), so a positional mask would scramble.
    marked = mark_open_close(blotter)
    opens = marked[marked["is_open"]]
    d = attach_revisions(opens, forecasts, variable)
    keep = d.dropna(subset=["past_revision", "future_revision"])
    coverage = len(keep) / len(opens) if len(opens) else 0.0
    return keep, coverage


def _ols_t(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with t-statistics. X must already include an intercept column."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    if n <= k:
        return beta, np.full(k, np.nan)
    s2 = resid @ resid / (n - k)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(s2 * xtx_inv))
    with np.errstate(divide="ignore", invalid="ignore"):
        return beta, beta / se


def leakage_test(blotter: pd.DataFrame, forecasts: pd.DataFrame,
                 variable: str) -> dict:
    """Partial loading of OPENING trade direction on the unknowable revision."""
    d, coverage = testable(blotter, forecasts, variable)
    base = {"variable": variable, "n": len(d), "coverage": coverage}

    if len(d) < MIN_N or coverage < MIN_COVERAGE:
        return {**base, "untestable": f"n={len(d)}, coverage={coverage:.0%}"}

    # Standardise so betas are comparable across variables and submissions.
    def z(s: pd.Series) -> np.ndarray | None:
        a = s.to_numpy(dtype=float)
        sd = a.std()
        return (a - a.mean()) / sd if sd > 0 else None

    past, future, y = (z(d["past_revision"]), z(d["future_revision"]),
                       z(d["direction"]))
    if past is None or future is None or y is None:
        return {**base, "untestable": "a regressor has zero variance"}

    beta, t = _ols_t(y, np.column_stack([np.ones(len(d)), past, future]))
    if not np.isfinite(t).all():
        return {**base, "untestable": "degenerate regression (non-finite t)"}

    return {**base,
            "beta_past": beta[1], "t_past": t[1],
            "beta_future": beta[2], "t_future": t[2]}


def _verdict(r: dict) -> str:
    """INFO when the test could not run. Never let an unrunnable test PASS."""
    if "untestable" in r:
        return "INFO"
    t, b = abs(r["t_future"]), abs(r["beta_future"])
    if not (np.isfinite(t) and np.isfinite(b)):
        return "INFO"
    if t >= T_FAIL and b >= BETA_FAIL:
        return "FAIL"
    if t >= T_SUSPECT:
        return "SUSPECT"
    return "PASS"


def lineage(registry_entry: dict, blotter: pd.DataFrame,
            forecasts: pd.DataFrame) -> SectionResult:
    """The section result for the leakage test, across all three variables.

    Runs identically for every submission — no per-submission branches.
    """
    findings, verdicts = [], []
    for var in VARIABLES:
        r = leakage_test(blotter, forecasts, var)
        v = _verdict(r)
        verdicts.append(v)

        if "untestable" in r:
            note = (f"NOT EVIDENCE OF HONESTY — the revision test cannot "
                    f"speak to this submission ({r['untestable']}); "
                    f"coverage={r['coverage']:.0%}. A one-sided strategy has "
                    f"no direction to correlate; a strategy trading outside "
                    f"the vintage window has no revision neighbourhood. "
                    f"Needs a selection-based test instead.")
            val = None
        else:
            ratio = (abs(r["beta_future"]) / abs(r["beta_past"])
                     if abs(r["beta_past"]) > 1e-9 else float("inf"))
            note = (f"direction loads {ratio:.2f}x as hard on the unpublished "
                    f"revision as on the published one "
                    f"(beta_past={r['beta_past']:+.3f} t={r['t_past']:+.1f}; "
                    f"beta_future={r['beta_future']:+.3f} t={r['t_future']:+.1f}; "
                    f"opens n={r['n']}, coverage={r['coverage']:.0%})")
            val = round(float(r["t_future"]), 2)

        findings.append(Finding(f"future_revision_loading_{var}", val, v, note))

    order = ["FAIL", "SUSPECT", "PASS", "INFO"]
    section_verdict = next(v for v in order if v in verdicts)
    return SectionResult("lineage", section_verdict, findings)
