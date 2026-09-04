"""Thresholds - turning the execution stress numbers into KEEP/SUSPECT/FAIL.

Everything in `friction.py` and `execution.py` is deliberately descriptive.
This module is the only place a threshold lives, so the line between "what
the data says" and "where we drew the line" stays visible.

Two sections, six checks:

    FILL MODE
      mid        the recorded fill must at least make money
      edge       the gross edge against the round-trip cost of earning it
      touch      crossing at the touch, measured against mid
      sweep      sweeping by size: FAILs against mid like the touch, but
                 is SUSPECT on SIZE - a clip too big for the book, or a
                 touch that was already SUSPECT

    MARKET ABSORPTION
      clip_size  our clip against the sizes this market actually prints
                 (the CLIP_QUANTILE tail of them)
      absorb     time for the market to trade our clip against how long we
                 hold it

Aggregation: any FAIL fails the submission, any SUSPECT (and no FAIL) makes
it SUSPECT, otherwise KEEP.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

KEEP, SUSPECT, FAIL = "KEEP", "SUSPECT", "FAIL"
RANK = {KEEP: 0, SUSPECT: 1, FAIL: 2}

# ---- the thresholds, all of them, in one place ---------------------------
DEGRADE = 0.50          # "more than half" - retaining less than this is a drop
ABSORB_SUSPECT = 0.50   # absorption time as a share of the holding period
ABSORB_FAIL = 1.00      # absorption time exceeding the holding period
CLIP_QUANTILE = 0.99    # our clip above this quantile of tape prints

# The fee is the only number in this module that is not measured from the
# records - set it from the desk's own schedule before anyone reads a verdict.
FEE_EUR_MWH = 0.10      # exchange + clearing, charged on every MWh
COVERAGE_FAIL = 1.00    # gross edge below the round-trip cost it must pay
COVERAGE_SUSPECT = 2.00  # under twice that cost - friction takes the majority


@dataclass
class Check:
    """One test: its verdict and the sentence that justifies it."""
    name: str
    verdict: str
    note: str
    values: dict = field(default_factory=dict)


def _retained(now: float, base: float) -> float:
    """`now / base`, or NaN when the baseline cannot support a ratio."""
    if base is None or not np.isfinite(base) or base <= 0:
        return np.nan
    return float(now) / float(base)


def _degraded(now: float, base: float) -> bool:
    """Did this metric lose more than half of its baseline?"""
    r = _retained(now, base)
    return bool(np.isfinite(r) and r < DEGRADE)


# --------------------------------------------------------------------------
# fill mode
# --------------------------------------------------------------------------

def check_mid(pnl_mid: float) -> Check:
    """The recorded fill is the most generous price there is. If it does not
    make money there, nothing downstream can rescue it."""
    ok = pnl_mid > 0
    return Check("mid", KEEP if ok else FAIL,
                 f"P&L at mid is {pnl_mid:,.0f} EUR"
                 + ("" if ok else " - not profitable at the price the blotter "
                                  "itself records"),
                 {"pnl_mid": pnl_mid})


def check_edge(edge_mid: float, half_spread: float,
               fee: float = FEE_EUR_MWH) -> Check:
    """Is the gross edge big enough to pay for getting in and out?

    `touch` and `sweep` ask what survived; this asks whether there was ever
    enough to survive. Coverage is the gross edge per MWh over the cost that
    same MWh must pay - half the spread, plus the fee - so 1.0 is a strategy
    handing its entire edge to the market and 2.0 is one keeping half.

    This is deliberately NOT independent of `touch`: retention at the touch
    is exactly `1 - half_spread / edge_mid`, so at a zero fee this check and
    the P&L leg of `touch` draw the same line. The fee is what makes it say
    something new, and it is the reason the check is stated per MWh - a
    number a trader can put next to a fee schedule and argue with.
    """
    cost = half_spread + fee
    cov = edge_mid / cost if cost > 0 else np.nan
    vals = {"edge_mid": edge_mid, "half_spread": half_spread, "fee": fee,
            "round_trip_cost": cost, "coverage": cov,
            "edge_net": edge_mid - cost}
    base = (f"edge is {edge_mid:.3f} EUR/MWh against {cost:.3f} to trade it "
            f"({half_spread:.3f} half-spread + {fee:.3f} fee) - {cov:.2f}x "
            "coverage")
    if not np.isfinite(cov) or cov < COVERAGE_FAIL:
        return Check("edge", FAIL,
                     base + " - the edge is inside the cost of trading it",
                     vals)
    if cov < COVERAGE_SUSPECT:
        return Check("edge", SUSPECT,
                     base + " - friction takes most of it", vals)
    return Check("edge", KEEP, base, vals)


def _versus_mid(name: str, pnl: float, sharpe: float,
                pnl_mid: float, sharpe_mid: float) -> Check:
    """Shared rule for `touch` and `sweep`, both measured against mid.

    FAIL when the P&L turns negative, or when P&L *and* Sharpe each keep less
    than half of their value at mid. SUSPECT when exactly one of the two
    falls that far - which is what makes the two conditions distinguishable;
    reading the rule as "P&L or Sharpe" would make every SUSPECT a FAIL.
    """
    pr, sr = _retained(pnl, pnl_mid), _retained(sharpe, sharpe_mid)
    pnl_bad, sh_bad = _degraded(pnl, pnl_mid), _degraded(sharpe, sharpe_mid)
    vals = {"pnl": pnl, "sharpe": sharpe,
            "pnl_retained": pr, "sharpe_retained": sr}

    if pnl <= 0:
        return Check(name, FAIL,
                     f"P&L is {pnl:,.0f} EUR - the strategy does not survive "
                     "this fill assumption at all", vals)
    if pnl_bad and sh_bad:
        return Check(name, FAIL,
                     f"keeps {pr:.0%} of P&L and {sr:.0%} of Sharpe - both "
                     f"lose more than half", vals)
    if sh_bad:
        return Check(name, SUSPECT,
                     f"P&L holds up ({pr:.0%}) but Sharpe keeps only "
                     f"{sr:.0%}", vals)
    if pnl_bad:
        return Check(name, SUSPECT,
                     f"Sharpe holds up ({sr:.0%}) but P&L keeps only "
                     f"{pr:.0%}", vals)
    return Check(name, KEEP,
                 f"keeps {pr:.0%} of P&L and {sr:.0%} of Sharpe", vals)


def check_touch(pnl: float, sharpe: float, pnl_mid: float,
                sharpe_mid: float) -> Check:
    return _versus_mid("touch", pnl, sharpe, pnl_mid, sharpe_mid)


def check_sweep(pnl: float, sharpe: float, pnl_mid: float, sharpe_mid: float,
                pnl_touch: float, sharpe_touch: float,
                touch_verdict: str) -> Check:
    """Sweep still FAILS against mid exactly as the touch does, but its
    SUSPECT is about SIZE alone.

    Sweeping is never better than the touch, so any spread cost has already
    been judged by `touch`; measuring sweep's SUSPECT against mid a second
    time just re-charged the same spread under a different name. What only
    sweep can see is the marginal damage of a clip that did not fit at the
    touch. So it is SUSPECT when the touch is SUSPECT - it cannot be sounder
    than the fill it is built on - or when the touch is fine and sweeping
    still gives up more than half of the P&L or the Sharpe it had there.
    """
    c = _versus_mid("sweep", pnl, sharpe, pnl_mid, sharpe_mid)
    pr_t, sr_t = _retained(pnl, pnl_touch), _retained(sharpe, sharpe_touch)
    c.values |= {"pnl_vs_touch": pr_t, "sharpe_vs_touch": sr_t,
                 "touch_verdict": touch_verdict}
    if c.verdict == FAIL:
        return c
    finite = [x for x in (pr_t, sr_t) if np.isfinite(x)]
    worst = min(finite) if finite else np.nan
    if _degraded(pnl, pnl_touch) or _degraded(sharpe, sharpe_touch):
        return Check("sweep", SUSPECT,
                     f"keeps only {worst:.0%} of its value at the touch - the "
                     "clip is too big for the book, which is a size cost "
                     "rather than a spread cost", c.values)
    if touch_verdict == SUSPECT:
        return Check("sweep", SUSPECT,
                     f"the clip fits - it keeps {worst:.0%} of the touch - "
                     "but the touch itself is SUSPECT, and sweeping cannot be "
                     "sounder than the fill it is built on", c.values)
    # Anything left is a SPREAD cost, which `touch` has already judged and
    # passed. Re-charging it here would count the same cost twice, so a
    # `_versus_mid` SUSPECT does not survive into sweep's verdict.
    return Check("sweep", KEEP,
                 f"the clip fits - it keeps {worst:.0%} of the touch", c.values)


# --------------------------------------------------------------------------
# market absorption
# --------------------------------------------------------------------------

def check_clip_size(clip_mw: float, big_print_mw: float,
                    median_print_mw: float) -> Check:
    """Our clip against the sizes this market actually prints.

    `big_print_mw` is the `CLIP_QUANTILE` quantile of tape print sizes.
    """
    tag = f"p{CLIP_QUANTILE * 100:.0f}"
    big = np.isfinite(big_print_mw) and clip_mw > big_print_mw
    return Check("clip_size", SUSPECT if big else KEEP,
                 f"clip is {clip_mw:,.0f} MW against a "
                 f"{median_print_mw:,.1f} MW median print and "
                 f"{big_print_mw:,.1f} MW at {tag}"
                 + (f" - larger than {CLIP_QUANTILE:.0%} of what trades here"
                    if big else ""),
                 {"clip_mw": clip_mw, "big_print_mw": big_print_mw,
                  "clip_quantile": CLIP_QUANTILE})


def check_absorption(absorb_min: float, hold_min: float) -> Check:
    """Time for the market to trade one clip, against how long we hold it.

    The risk runs one way: a market that needs LONGER than the holding period
    to move our size cannot have supported the trade. (Read literally the
    brief inverts this - it would flag a strategy for holding longer than the
    market needs, which is the safe case - so the ratio is taken as
    absorption over hold.)
    """
    if not np.isfinite(absorb_min):
        return Check("absorb", FAIL,
                     "the tape never trades one clip before the product "
                     "gates", {"absorb_min": absorb_min, "hold_min": hold_min})
    ratio = absorb_min / hold_min if hold_min else np.inf
    vals = {"absorb_min": absorb_min, "hold_min": hold_min,
            "absorb_vs_hold": ratio}
    base = (f"the market takes {absorb_min:,.0f} min to trade one clip "
            f"against a {hold_min:,.0f} min hold ({ratio:.2f}x)")
    if ratio > ABSORB_FAIL:
        return Check("absorb", FAIL,
                     base + " - longer than the position is meant to exist",
                     vals)
    if ratio > ABSORB_SUSPECT:
        return Check("absorb", SUSPECT,
                     base + " - over half the holding period", vals)
    return Check("absorb", KEEP, base, vals)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

CHECKS = ("mid", "edge", "touch", "sweep", "clip_size", "absorb")
SECTION = {"mid": "fill mode", "edge": "fill mode", "touch": "fill mode",
           "sweep": "fill mode",
           "clip_size": "market absorption", "absorb": "market absorption"}


def assess(daily: dict, stats: dict) -> list[Check]:
    """All five checks for one submission.

    `daily` holds the mid/cross/sweep P&L series' headline numbers,
    `stats` is an `execution.absorption_summary`.
    """
    pm, sm = daily["pnl_mid"], daily["sharpe_mid"]
    checks = [check_mid(pm),
              check_edge(daily["edge_mid"], daily["half_spread"],
                         daily.get("fee", FEE_EUR_MWH)),
              touch := check_touch(daily["pnl_touch"], daily["sharpe_touch"],
                                   pm, sm),
              check_sweep(daily["pnl_sweep"], daily["sharpe_sweep"], pm, sm,
                          daily["pnl_touch"], daily["sharpe_touch"],
                          touch.verdict),
              check_clip_size(stats["clip_mw"],
                              stats.get(f"p{CLIP_QUANTILE * 100:.0f}_print_mw",
                                        np.nan),
                              stats["median_print_mw"]),
              check_absorption(stats["median_absorb_min"],
                               stats["hold_median_min"])]
    return checks


# A SUSPECT is not a soft KEEP - it is a KEEP with strings attached, and
# `report.Report` treats a SUSPECT carrying no conditions as an unfinished
# report. Every check that can return SUSPECT names its condition here.
CONDITIONS = {
    "edge": ("re-price coverage against the desk's own fee schedule before "
             f"funding; kill below {COVERAGE_FAIL:.1f}x"),
    "touch": ("fund at reduced size and reconcile realised slippage against "
              "the modelled half-spread weekly"),
    "sweep": ("cap the clip at the size that fills at the touch, and "
              "re-check if the book thins"),
    "clip_size": (f"cap the clip at the p{CLIP_QUANTILE * 100:.0f} print "
                  "size and re-measure after a month of live prints"),
    "absorb": ("paper-trade until median absorption is under "
               f"{ABSORB_SUSPECT:.0%} of the holding period"),
}


def conditions(checks: list[Check]) -> list[str]:
    """What must hold for a SUSPECT submission to be funded anyway."""
    return [CONDITIONS[c.name] for c in checks
            if c.verdict == SUSPECT and c.name in CONDITIONS]


def overall(checks: list[Check]) -> str:
    """Any FAIL fails it; any SUSPECT and no FAIL makes it SUSPECT."""
    return max((c.verdict for c in checks), key=lambda v: RANK[v])


def failing(checks: list[Check], verdict: str | None = None) -> list[str]:
    """Which checks carry the deciding verdict, named for reporting."""
    verdict = verdict or overall(checks)
    if verdict == KEEP:
        return []
    return [c.name for c in checks if c.verdict == verdict]


def sections_failing(checks: list[Check]) -> list[str]:
    """The distinct sections responsible for the overall verdict."""
    out = []
    for name in failing(checks):
        s = SECTION[name]
        if s not in out:
            out.append(s)
    return out


def grid(per_submission: dict[str, list[Check]]) -> pd.DataFrame:
    """One row per submission, one column per check, plus the overall."""
    rows = {}
    for key, checks in per_submission.items():
        row = {c.name: c.verdict for c in checks}
        row["OVERALL"] = overall(checks)
        rows[key] = row
    out = pd.DataFrame(rows).T
    cols = [c for c in CHECKS if c in out.columns] + ["OVERALL"]
    return out[cols].sort_values("OVERALL",
                                 key=lambda s: s.map(RANK), kind="stable")


SECTIONS = ("fill mode", "market absorption")


def section_verdict(checks: list[Check], section: str) -> str:
    """The worst verdict among the checks belonging to one section."""
    inside = [c.verdict for c in checks if SECTION[c.name] == section]
    return max(inside, key=lambda v: RANK[v]) if inside else KEEP


def section_grid(per_submission: dict[str, list[Check]]) -> pd.DataFrame:
    """One row per submission, one column per SECTION, plus the overall.

    The rolled-up view: `grid` shows which individual check tripped, this
    shows which part of the analysis it tripped in.
    """
    rows = {}
    for key, checks in per_submission.items():
        row = {s: section_verdict(checks, s) for s in SECTIONS}
        row["OVERALL"] = overall(checks)
        rows[key] = row
    out = pd.DataFrame(rows).T
    return out[list(SECTIONS) + ["OVERALL"]].sort_values(
        "OVERALL", key=lambda s: s.map(RANK), kind="stable")
