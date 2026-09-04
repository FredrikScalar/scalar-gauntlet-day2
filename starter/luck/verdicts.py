"""The Keep / Suspect / Fail rules, in one place.

Every threshold a reviewer might argue about lives here and nowhere else. Each
rule takes the computed record for one submission (as produced by fivetests.py)
plus the execution rung being judged, and returns (verdict, reason).

Rung note: the coin-flip rule is deliberately scored at MID regardless of the
rung being displayed. At mid the null sits on zero, so the distance measures
directional skill. At a rung that charges friction the null goes deeply
negative, and beating it only proves the strategy churned less than a random
trader — which is not skill.
"""
from __future__ import annotations

KEEP, SUSPECT, FAIL = "KEEP", "SUSPECT", "FAIL"
ORDER = {KEEP: 0, SUSPECT: 1, FAIL: 2}

# ---- tunables (documented on the report page) ---------------------------
COINFLIP_Z_KEEP     = 4.0     # z above the coin-flip mean, at mid
COINFLIP_Z_FAIL     = 2.5
COINFLIP_P_FAIL     = 0.01
BOOT_LOWER_KEEP     = 1.0     # 95% interval lower bound, annualised Sharpe
DSR_BREAK_KEEP_MULT = 10      # breakeven trials >= this x declared
DAYS_PCT_KEEP       = 15.0    # breakeven days as % of the strategy's active days
DAYS_PCT_FAIL       = 5.0
RATIO_KEEP          = 3.0     # breadth-neutral edge ratio, x the market
RATIO_FAIL          = 1.0
LONG_BEATS_MARGIN   = 0.05    # Sharpe margin by which always-long must win to fail it


def coinflip(rec, rung):
    """Null A: same trades, same clock, side randomised. Judged at mid."""
    a = rec["rungs"]["mid"]["mc_a"]
    seed = rec["seed_stability"]
    z, p = a["sharpe_z"], a["sharpe_p_value"]
    flips = "FLIP" in str(seed["beats_max"]).upper() or seed["above_q999"] == "FLIPS"
    if p > COINFLIP_P_FAIL or z < COINFLIP_Z_FAIL:
        return FAIL, f"z={z:.1f} against the coin-flip crowd, p={p:.4f}"
    if z < COINFLIP_Z_KEEP:
        return SUSPECT, f"z={z:.1f} — inside the band where the best-of-10,000 verdict is fragile"
    if flips or seed["beats_max"] != "always":
        return SUSPECT, f"z={z:.1f} but the best-of-10,000 verdict is not seed-stable"
    return KEEP, f"z={z:.1f}, beats the luckiest of 10,000 coin flips on every seed"


def bootstrap(rec, rung):
    """95% resample interval on annualised Sharpe; the wider of iid and block."""
    b = rec["rungs"][rung]["boot"]
    lo = min(b["ci"][0], b["blk_ci"][0])
    hi = max(b["ci"][1], b["blk_ci"][1])
    if lo <= 0:
        return FAIL, f"95% interval {lo:.2f} to {hi:.2f} contains zero"
    if lo < BOOT_LOWER_KEEP:
        return SUSPECT, f"95% interval lower bound only {lo:.2f}"
    return KEEP, f"95% interval {lo:.2f} to {hi:.2f}, clear of zero"


def deflated(rec, rung):
    """Breakeven trial count against the count the submitter declared."""
    d = rec["rungs"][rung]["dsr"]
    N, brk = d["declared"], d["break_analytic"]
    if brk is None:
        return KEEP, f"deflated Sharpe never falls through 0.95, declared {N} trials"
    if brk < N:
        return FAIL, f"breaks at {brk} trials, below the {N} declared"
    if brk < DSR_BREAK_KEEP_MULT*N:
        return SUSPECT, f"breaks at {brk} trials against {N} declared ({brk/N:.1f}x)"
    return KEEP, f"breaks at {brk} trials, {brk/N:.0f}x the {N} declared"


def best_days(rec, rung):
    """How many of the best days must be deleted before the P&L is gone."""
    r = rec["rungs"][rung]
    z, act = r["days_to_zero"], r["active_days"]
    if r["pnl"] <= 0:
        return FAIL, "negative P&L on this basis — nothing to erase"
    if z is None or act == 0:
        return FAIL, "no active days"
    pct = 100.0*z/act
    if pct < DAYS_PCT_FAIL:
        return FAIL, f"deleting {z} best days ({pct:.1f}% of active days) erases it"
    if pct < DAYS_PCT_KEEP:
        return SUSPECT, f"deleting {z} best days ({pct:.1f}% of active days) erases it"
    return KEEP, f"needs {z} best days ({pct:.1f}% of active days) deleted to break even"


def baseline(rec, rung):
    """Beats benchmarks that need no signal, breadth held neutral."""
    bb = rec["rungs"][rung]["base"]
    ratio = rec["edge_ratio_mid"]
    beaten = bb["always_long"]["sharpe"] > bb["strategy"]["sharpe"] + LONG_BEATS_MARGIN
    if beaten:
        return FAIL, (f"forcing its own trades long scores {bb['always_long']['sharpe']:.2f} "
                      f"against the strategy's {bb['strategy']['sharpe']:.2f}")
    if ratio < RATIO_FAIL:
        return FAIL, f"breadth-neutral edge is {ratio:.2f}x the market — below the market itself"
    if ratio < RATIO_KEEP:
        return SUSPECT, f"breadth-neutral edge only {ratio:.2f}x the market"
    return KEEP, f"breadth-neutral edge {ratio:.2f}x the market"


TESTS = [
    ("coinflip",  "Coin-flip trader",    coinflip),
    ("bootstrap", "Bootstrapped Sharpe", bootstrap),
    ("dsr",       "Deflated Sharpe",     deflated),
    ("days",      "Best-days removal",   best_days),
    ("baseline",  "Baseline comparison", baseline),
]


def score(rec, rung):
    """All five verdicts plus the overall (worst cell sets the row)."""
    out, why = {}, {}
    for tid, _name, fn in TESTS:
        v, reason = fn(rec, rung)
        out[tid] = v
        why[tid] = reason
    out["overall"] = max(out.values(), key=lambda s: ORDER[s])
    return out, why
