"""Friction lab - what if we had executed like this?

The blotters record every fill at the book mid, a price nobody quoted. This
module asks what those same trades would have been worth under execution
assumptions you can actually defend, and draws the answer.

It deliberately reaches NO verdict. Nothing here returns KEEP/SUSPECT/FAIL or
sets a threshold; it produces the numbers and figures a verdict would later
have to stand on. Wiring it into `report.py` is a separate, later step.

The checks, each a table plus a figure:

    1  crossing the spread   crossing_check / crossing_summary / plot_crossing
       a buy lifts ask_1 and a sell hits bid_1, instead of trading at mid

    2  sensitivity to fees   fee_sensitivity / fee_sensitivity_table
       a flat EUR/MWh charge in 5 cent steps, and the fee that breaks it

    3  execution delay       delay_sensitivity / delay_summary
       filling late, measured as a fraction of the strategy's own median
       holding time rather than as a fixed number of minutes

`performance` / `performance_table` / `plot_performance` report the headline
stats under any single fill assumption - see `FILLS`.

The lower-level `Scenario` / `compare` / `sweep` layer underneath composes the
axes freely - BBO cross, sweep, delay, fee, slippage - for anything the named
checks above do not cover.

Run it directly to run all three checks over all seven and write figures:

    python friction.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from repricer import daily_pnl, reprice_mid, sharpe

BID = [("bid_px_1", "bid_sz_1"), ("bid_px_2", "bid_sz_2"), ("bid_px_3", "bid_sz_3")]
ASK = [("ask_px_1", "ask_sz_1"), ("ask_px_2", "ask_sz_2"), ("ask_px_3", "ask_sz_3")]
LEVEL_COLS = [c for pair in BID + ASK for c in pair]

SNAPSHOT = pd.Timedelta(minutes=15)   # the book's snapshot cadence


@dataclass(frozen=True)
class Scenario:
    """One execution assumption. `mid` (all defaults) is the backtest's own."""
    name: str
    cross: bool = False              # pay the spread instead of trading at mid
    sweep: bool = False              # sweep levels 1..3 by size, not just the touch
    delay_snapshots: int = 0         # fill this many 15-min snapshots later
    fee_eur_mwh: float = 0.0         # flat charge per MWh, always against you
    slippage_eur_mwh: float = 0.0    # extra adverse ticks per MWh


def default_scenarios(fee: float = 0.05) -> list[Scenario]:
    """A ladder from the backtest's fill to something a desk would recognise."""
    return [
        Scenario("mid"),
        Scenario("cross", cross=True),
        Scenario("sweep", cross=True, sweep=True),
        Scenario("sweep+fee", cross=True, sweep=True, fee_eur_mwh=fee),
        Scenario("sweep+fee+15min", cross=True, sweep=True, fee_eur_mwh=fee,
                 delay_snapshots=1),
        Scenario("sweep+fee+1h", cross=True, sweep=True, fee_eur_mwh=fee,
                 delay_snapshots=4),
    ]


# --------------------------------------------------------------------------
# book plumbing
# --------------------------------------------------------------------------

def _sweep_vwap(px: np.ndarray, sz: np.ndarray, qty: np.ndarray) -> np.ndarray:
    """Size-weighted fill price consuming levels 1..N. NaN where depth < qty."""
    px, sz = np.asarray(px, float), np.nan_to_num(np.asarray(sz, float))
    remaining = qty.astype(float).copy()
    cost, filled = np.zeros(len(qty)), np.zeros(len(qty))
    for lvl in range(px.shape[1]):
        take = np.minimum(remaining, sz[:, lvl])
        cost += take * np.nan_to_num(px[:, lvl])
        filled += take
        remaining -= take
    return np.where(filled >= qty - 1e-9, cost / qty, np.nan)


def attach_book(blotter: pd.DataFrame, market: dict,
                delay_snapshots: int = 0) -> pd.DataFrame:
    """Join each trade to the book levels it would have executed against.

    With `delay_snapshots > 0` the trade is filled against the book as it
    stood that many 15-minute snapshots later. Where the product has already
    gated by then, it falls back to the last snapshot before gate - the
    latest price that actually existed - and flags the row in `delayed_gated`.
    """
    b = reprice_mid(blotter, market)
    book = (market["book"][["ts", "product_id", *LEVEL_COLS]]
            .sort_values("ts").reset_index(drop=True))

    if delay_snapshots == 0:
        out = b.merge(book, left_on=["exec_ts", "product_id"],
                      right_on=["ts", "product_id"], how="left").drop(columns="ts")
        out["delayed_gated"] = False
        if out[LEVEL_COLS].isna().any().any():
            raise ValueError("some trades have no book levels at exec_ts")
        return out

    b = b.copy()
    b["target_ts"] = b["exec_ts"] + delay_snapshots * SNAPSHOT
    last_ts = book.groupby("product_id")["ts"].max().rename("last_ts")
    out = (pd.merge_asof(b.sort_values("target_ts"), book,
                         left_on="target_ts", right_on="ts",
                         by="product_id", direction="backward")
           .drop(columns="ts")
           .merge(last_ts, on="product_id", how="left"))
    out["delayed_gated"] = out["target_ts"] > out["last_ts"]
    if out[LEVEL_COLS].isna().any().any():
        raise ValueError("delayed fill lands before the product's first snapshot")
    return out.drop(columns=["target_ts", "last_ts"])


def fill_price(b: pd.DataFrame, sc: Scenario) -> pd.Series:
    """The price this scenario says each trade actually got."""
    is_buy = b["side"].eq("BUY").values
    qty = b["qty_mw"].values
    mid = (b["bid_px_1"] + b["ask_px_1"]).values / 2.0

    if not sc.cross:
        px = mid
    elif sc.sweep:
        px = np.where(is_buy,
                      _sweep_vwap(b[[p for p, _ in ASK]].values,
                                  b[[s for _, s in ASK]].values, qty),
                      _sweep_vwap(b[[p for p, _ in BID]].values,
                                  b[[s for _, s in BID]].values, qty))
    else:
        px = np.where(is_buy, b["ask_px_1"].values, b["bid_px_1"].values)

    # costs always work against you: they lift your buys and cut your sells
    adverse = sc.fee_eur_mwh + sc.slippage_eur_mwh
    if adverse:
        px = px + np.where(is_buy, adverse, -adverse)
    return pd.Series(px, index=b.index, name=f"px_{sc.name}")


# --------------------------------------------------------------------------
# descriptive metrics - no thresholds, no verdicts
# --------------------------------------------------------------------------

def half_spread(b: pd.DataFrame) -> pd.Series:
    return (b["ask_px_1"] - b["bid_px_1"]) / 2.0


def edge_per_mwh(b: pd.DataFrame, price_col: str = "mid_price") -> float:
    """P&L per MWh traded - directly comparable to the half-spread, since
    every MWh traded pays that half-spread once."""
    mwh = float(b["qty_mw"].sum())
    return float(daily_pnl(b, price_col).sum()) / mwh if mwh else np.nan


def sortino(daily: pd.Series) -> float:
    """Annualised Sortino, matching the convention the submissions book used.

    The denominator is the standard deviation of LOSING days about their own
    mean (ddof=1) - not downside deviation about zero. That is the convention
    that reproduces all seven claimed `sortino_ann` values exactly; it is more
    flattering than downside deviation, so it is worth knowing it is theirs.
    """
    losses = daily[daily < 0]
    if len(losses) < 2:
        return 0.0
    sd = losses.std(ddof=1)
    return float(daily.mean() / sd * np.sqrt(365.0)) if sd > 0 else 0.0


def max_drawdown(daily: pd.Series) -> float:
    """Deepest peak-to-trough fall of the cumulative P&L, in EUR (<= 0)."""
    cum = daily.cumsum()
    return float((cum - cum.cummax()).min())


def active_days_frac(daily: pd.Series) -> float:
    """Share of days in the window on which the strategy traded at all."""
    return float((daily != 0).mean()) if len(daily) else np.nan


def hit_rate_active_days(daily: pd.Series) -> float:
    """Share of trading days that made money."""
    active = daily[daily != 0]
    return float((active > 0).mean()) if len(active) else np.nan


def metrics(b: pd.DataFrame, price_col: str,
            window: tuple[str, str] | None = None) -> dict:
    """The submissions book's headline stats, recomputed from the records.

    Same metric set and same conventions as the registry's `claimed` block,
    so the two are directly comparable line for line.
    """
    d = daily_pnl(b, price_col, window=window)
    mwh = float(b["qty_mw"].sum())
    return {
        "total_pnl_eur": float(d.sum()),
        "sharpe_ann": sharpe(d),
        "sortino_ann": sortino(d),
        "max_drawdown_eur": max_drawdown(d),
        "active_days_frac": active_days_frac(d),
        "hit_rate_active_days": hit_rate_active_days(d),
        "edge_eur_per_mwh": float(d.sum()) / mwh if mwh else np.nan,
        "mwh_traded": mwh,
        "trades": int(len(b)),
    }


def compare(blotter: pd.DataFrame, market: dict, scenarios: list[Scenario],
            window: tuple[str, str] | None = None) -> pd.DataFrame:
    """One row per scenario: what the same trades were worth under each."""
    rows = {}
    cache: dict[int, pd.DataFrame] = {}
    for sc in scenarios:
        if sc.delay_snapshots not in cache:
            cache[sc.delay_snapshots] = attach_book(blotter, market, sc.delay_snapshots)
        b = cache[sc.delay_snapshots].copy()
        col = f"px_{sc.name}"
        b[col] = fill_price(b, sc)
        d = daily_pnl(b, col, window=window)
        mwh = float(b["qty_mw"].sum())
        rows[sc.name] = {
            "pnl_eur": float(d.sum()),
            "sharpe_ann": sharpe(d),
            "eur_per_mwh": float(d.sum()) / mwh if mwh else np.nan,
            "unfilled_clips": int(b[col].isna().sum()),
            "gated_fills": int(b["delayed_gated"].sum()),
        }
    out = pd.DataFrame(rows).T
    base = out["pnl_eur"].iloc[0]
    out["pnl_retained"] = out["pnl_eur"] / base if base else np.nan
    out["cost_vs_base_eur"] = out["pnl_eur"] - base
    return out


def crossing_check(entry: dict, blotter: pd.DataFrame, market: dict) -> pd.DataFrame:
    """CHECK 1 - crossing the spread.

    Reprices every trade so a buy lifts ask_1 and a sell hits bid_1, and lays
    the headline stats out against the backtest's own fill:

        mid    every trade at the book mid - what the submission claims
        cross  the same trades, paying the spread

    Reports numbers only. It does not decide whether the decay is acceptable.
    """
    window = (str(entry["backtest_window"]["start"]),
              str(entry["backtest_window"]["end"]))
    b = attach_book(blotter, market)
    b["px_mid"] = fill_price(b, Scenario("mid"))
    b["px_cross"] = fill_price(b, Scenario("cross", cross=True))

    mid, cross = metrics(b, "px_mid", window), metrics(b, "px_cross", window)
    mid["half_spread_eur_mwh"] = cross["half_spread_eur_mwh"] = \
        float(half_spread(b).mean())

    out = pd.DataFrame({"mid": mid, "cross": cross})
    out["cross_minus_mid"] = out["cross"] - out["mid"]
    return out


FILLS = {
    # the backtest's own fill: a price nobody quoted
    "mid": Scenario("mid"),
    # BBO cross: buy at ask_1, sell at bid_1, WHATEVER the size. Ignores
    # whether the clip actually fits at the touch, so for a clip larger than
    # level-1 depth this is not conservative - it is impossible.
    "cross": Scenario("cross", cross=True),
    # sweep: consume levels 1..3 by size. The honest treatment of a clip that
    # does not fit at the touch; identical to `cross` when it does.
    "sweep": Scenario("sweep", cross=True, sweep=True),
}


def _priced(entry: dict, blotter: pd.DataFrame, market: dict, fill: str):
    """(blotter with a `px_<fill>` column, daily P&L over the declared window)."""
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {sorted(FILLS)}, got {fill!r}")
    window = (str(entry["backtest_window"]["start"]),
              str(entry["backtest_window"]["end"]))
    b = attach_book(blotter, market)
    b[f"px_{fill}"] = fill_price(b, FILLS[fill])
    return b, daily_pnl(b, f"px_{fill}", window=window)


def performance(entry: dict, blotter: pd.DataFrame, market: dict,
                fill: str = "cross") -> dict:
    """Final performance under one fill assumption - `cross` by default.

    The same headline stats the submissions book quotes, but computed at the
    prices you would actually get rather than at mid.
    """
    window = (str(entry["backtest_window"]["start"]),
              str(entry["backtest_window"]["end"]))
    b, _ = _priced(entry, blotter, market, fill)
    out = metrics(b, f"px_{fill}", window)
    out["fill"] = fill
    out["half_spread_eur_mwh"] = float(half_spread(b).mean())
    return out


def performance_table(registry: dict, market: dict, fill: str = "cross",
                      blotter_dir: str | Path = "../blotters") -> pd.DataFrame:
    """Final performance for every submission under one fill assumption."""
    from repricer import load_blotter

    rows = {}
    for key, entry in sorted(registry.items()):
        blot = load_blotter(Path(blotter_dir) / f"{key}-blotter.csv")
        rows[key] = performance(entry, blot, market, fill)
    cols = ["total_pnl_eur", "sharpe_ann", "sortino_ann", "max_drawdown_eur",
            "active_days_frac", "hit_rate_active_days", "edge_eur_per_mwh",
            "half_spread_eur_mwh", "mwh_traded", "trades"]
    return pd.DataFrame(rows).T[cols].sort_values("total_pnl_eur", ascending=False)


def depth_cost(entry: dict, blotter: pd.DataFrame, market: dict) -> pd.DataFrame:
    """CHECK - what the book's DEPTH costs, on top of the touch.

    Separates the two things `cross` conflates:

        cross   pay the touch (ask_1 / bid_1) for the whole clip, however big
        sweep   consume levels 1..3 by size, which is what actually happens

    `cross` is only reachable when the clip fits in level-1 depth. Where it
    does not, `cross` is not a conservative assumption but an impossible one,
    and `sweep - cross` is the size of that error. Zero means every clip fit
    at the touch and the two are the same fill.
    """
    window = (str(entry["backtest_window"]["start"]),
              str(entry["backtest_window"]["end"]))
    b = attach_book(blotter, market)
    for name in ("mid", "cross", "sweep"):
        b[f"px_{name}"] = fill_price(b, FILLS[name])

    l1 = np.where(b["side"].eq("BUY"), b["ask_sz_1"], b["bid_sz_1"])
    over = b["qty_mw"].values > l1 + 1e-9

    out = pd.DataFrame({n: metrics(b, f"px_{n}", window)
                        for n in ("mid", "cross", "sweep")})
    out["sweep_minus_cross"] = out["sweep"] - out["cross"]

    # how much of the book each clip had to eat, and how often the touch was
    # not enough on its own
    depth_used = b["qty_mw"].values / l1
    extra = pd.DataFrame(
        {"sweep": [float(over.sum()), float(over.mean()),
                   float(depth_used.max()), float(b["px_sweep"].isna().sum())]},
        index=["clips_over_level_1", "clips_over_level_1_frac",
               "max_depth_used_x_level_1", "unfillable_clips"])
    return pd.concat([out, extra.reindex(columns=out.columns)])


def depth_cost_summary(registry: dict, market: dict,
                       blotter_dir: str | Path = "../blotters") -> pd.DataFrame:
    """Sweep vs. BBO cross across all seven: where does depth actually bite?"""
    from repricer import load_blotter

    rows = {}
    for key, entry in sorted(registry.items()):
        blot = load_blotter(Path(blotter_dir) / f"{key}-blotter.csv")
        d = depth_cost(entry, blot, market)
        rows[key] = {
            "pnl_cross": d.loc["total_pnl_eur", "cross"],
            "pnl_sweep": d.loc["total_pnl_eur", "sweep"],
            "depth_cost_eur": d.loc["total_pnl_eur", "sweep_minus_cross"],
            "sharpe_cross": d.loc["sharpe_ann", "cross"],
            "sharpe_sweep": d.loc["sharpe_ann", "sweep"],
            "mdd_cross": d.loc["max_drawdown_eur", "cross"],
            "mdd_sweep": d.loc["max_drawdown_eur", "sweep"],
            "clips_over_L1": d.loc["clips_over_level_1", "sweep"],
            "clips_over_L1_frac": d.loc["clips_over_level_1_frac", "sweep"],
            "max_depth_used": d.loc["max_depth_used_x_level_1", "sweep"],
        }
    return pd.DataFrame(rows).T.sort_values("depth_cost_eur")


FEE_STEP = 0.05   # EUR/MWh - the default resolution of the fee sweep

# delay, as a share of holding time - roughly log-spaced, so the small end is
# as well resolved as the large. How much of the small end is actually
# reachable depends on the strategy: see `resolution_floor`.
HOLD_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.25, 0.50)


def delay_grid(blotter: pd.DataFrame, stat: str = "median",
               max_fraction: float = 0.50,
               max_points: int = 8) -> tuple[float, ...]:
    """Every delay this strategy can actually be tested at, up to
    `max_fraction` of its holding time.

    Built from whole 15-minute snapshots rather than from round numbers, so
    every point is representable and nothing comes back blank. The smallest
    step (one snapshot) is always included even when it already exceeds
    `max_fraction` - for a strategy that holds only one snapshot, that single
    point is the entire testable range.

    Long holders can resolve dozens of steps; the grid is thinned evenly to
    `max_points` so a chart stays readable, keeping the first and the last.
    """
    hold_min = holding_time(blotter)[f"{stat}_h"] * 60.0
    if not hold_min or np.isnan(hold_min):
        return ()
    step_min = SNAPSHOT.total_seconds() / 60.0
    n_max = max(int(np.floor(max_fraction * hold_min / step_min)), 1)
    snaps = np.arange(1, n_max + 1)
    if len(snaps) > max_points:
        pick = np.unique(np.linspace(0, len(snaps) - 1, max_points).round()
                         .astype(int))
        snaps = snaps[pick]
    return tuple(round(s * step_min / hold_min, 6) for s in snaps)


def resolution_floor(blotter: pd.DataFrame, stat: str = "median") -> float:
    """Smallest delay fraction the book can actually represent, for this book.

    Snapshots are 15 minutes apart and the requested delay is rounded to the
    nearest one (half up), so anything under half a snapshot (7.5 min) rounds
    to no delay at all. Expressed as a fraction of holding time, that floor is
    7.5 min / holding time - 1.3% for a ten-hour holder, but 50% for one that
    holds fifteen minutes. Below it, a delay check cannot say anything; at or
    above it, at least one snapshot of delay is applied.
    """
    hold_min = holding_time(blotter)[f"{stat}_h"] * 60.0
    return (SNAPSHOT.total_seconds() / 60.0 / 2.0) / hold_min if hold_min else np.nan


def round_trips(blotter: pd.DataFrame) -> pd.DataFrame:
    """FIFO-match every open against its close, within each product.

    Returns one row per matched pair: `open_ts, close_ts, qty_mw, hold_s`.
    Volume left in the queue is an unclosed position; the pack states there
    are none, and `holding_time` checks that rather than assuming it.
    """
    from collections import deque

    rows, unmatched = [], 0.0
    for _, g in blotter.groupby("product_delivery", sort=False):
        g = g.sort_values("exec_ts")
        open_lots: deque = deque()
        for ts, side, qty in zip(g["exec_ts"], g["side"], g["qty_mw"]):
            while qty > 1e-9 and open_lots and open_lots[0][2] != side:
                o_ts, o_qty, _ = open_lots[0]
                take = min(qty, o_qty)
                rows.append((o_ts, ts, take, (ts - o_ts).total_seconds()))
                qty -= take
                if o_qty - take <= 1e-9:
                    open_lots.popleft()
                else:
                    open_lots[0][1] = o_qty - take
            if qty > 1e-9:
                open_lots.append([ts, qty, side])
        unmatched += sum(lot[1] for lot in open_lots)

    out = pd.DataFrame(rows, columns=["open_ts", "close_ts", "qty_mw", "hold_s"])
    out.attrs["unmatched_mw"] = unmatched
    return out


def holding_time(blotter: pd.DataFrame) -> dict:
    """Volume-weighted mean holding time, plus the spread around it (hours)."""
    rt = round_trips(blotter)
    w = rt["qty_mw"]
    return {
        "mean_h": float((rt["hold_s"] * w).sum() / w.sum()) / 3600.0,
        "median_h": float(rt["hold_s"].median()) / 3600.0,
        "min_h": float(rt["hold_s"].min()) / 3600.0,
        "max_h": float(rt["hold_s"].max()) / 3600.0,
        "round_trips": int(len(rt)),
        "unmatched_mw": float(rt.attrs["unmatched_mw"]),
    }


def delay_sensitivity(entry: dict, blotter: pd.DataFrame, market: dict,
                      fractions: tuple[float, ...] | None = None,
                      fill: str = "cross", stat: str = "median") -> pd.DataFrame:
    """Fill late by a fraction of the strategy's own holding time.

    A fixed 15-minute delay means something very different to a strategy that
    holds for 18 minutes than to one that holds for ten hours, so the delay is
    scaled to each strategy's own clock. `stat` picks which holding time sets
    that clock - the median by default, since a few very long holds should not
    stretch the yardstick for the typical trade.

    The book is only snapshotted every 15 minutes, so the requested delay is
    rounded to the nearest whole snapshot. That rounding is reported, not
    hidden: `applied_min` and `realised_fraction` say what was actually done,
    and a fast strategy can see a small requested delay round to zero.
    """
    window = (str(entry["backtest_window"]["start"]),
              str(entry["backtest_window"]["end"]))
    hold_h = holding_time(blotter)[f"{stat}_h"]
    step_min = SNAPSHOT.total_seconds() / 60.0
    if fractions is None:
        # every delay the book can actually apply for THIS strategy, rather
        # than round numbers that mostly fall below its resolution
        fractions = delay_grid(blotter, stat)

    rows, cache = {}, {}
    for frac in (0.0, *fractions):
        target_min = frac * hold_h * 60.0
        # round half UP, not Python's banker's rounding: a delay landing
        # exactly on half a snapshot should resolve, so that `resolution_floor`
        # is the fraction at which a delay first becomes representable
        snaps = int(np.floor(target_min / step_min + 0.5))
        if snaps not in cache:
            b = attach_book(blotter, market, snaps)
            b["px"] = fill_price(b, FILLS[fill])
            cache[snaps] = b
        b = cache[snaps]
        d = daily_pnl(b, "px", window=window)
        applied_min = snaps * step_min
        rows[frac] = {
            "target_min": target_min,
            "snapshots": snaps,
            "applied_min": applied_min,
            "realised_fraction": applied_min / (hold_h * 60.0) if hold_h else np.nan,
            # a request the 15-minute book rounded away to nothing
            "below_resolution": bool(frac > 0 and snaps == 0),
            "total_pnl_eur": float(d.sum()),
            "sharpe_ann": sharpe(d),
            "sortino_ann": sortino(d),
            "max_drawdown_eur": max_drawdown(d),
            "gated_fills": int(b["delayed_gated"].sum()),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "hold_fraction"
    base = out["total_pnl_eur"].iloc[0]
    out["pnl_retained"] = out["total_pnl_eur"] / base if base else np.nan
    return out


def delay_summary(registry: dict, market: dict,
                  fractions: tuple[float, ...] | None = None,
                  fill: str = "cross", stat: str = "median",
                  blotter_dir: str | Path = "../blotters") -> pd.DataFrame:
    """Holding time and delay decay for every submission, one row each.

    Each strategy is swept over its OWN representable delays (see
    `delay_grid`), so the columns here are comparable endpoints rather than a
    shared grid of fractions that would be blank for the fast ones:
    the smallest delay the book can apply, and the largest at or under half
    the holding time.
    """
    from repricer import load_blotter

    rows = {}
    for key, entry in sorted(registry.items()):
        blot = load_blotter(Path(blotter_dir) / f"{key}-blotter.csv")
        h = holding_time(blot)
        sens = delay_sensitivity(entry, blot, market, fractions, fill, stat)
        body = sens.drop(index=0.0, errors="ignore")
        base = sens.loc[0.0, "total_pnl_eur"]
        row = {"hold_median_h": h["median_h"], "hold_mean_h": h["mean_h"],
               "round_trips": h["round_trips"],
               "unmatched_mw": h["unmatched_mw"],
               "resolution_floor": resolution_floor(blot, stat),
               "delay_points": int(len(body)),
               # the baseline everything else is read against
               "pnl_immediate": base}
        if len(body):
            lo, hi = body.index[0], body.index[-1]
            row |= {
                "smallest_frac": lo,
                "smallest_min": body.loc[lo, "applied_min"],
                "pnl_at_smallest": body.loc[lo, "total_pnl_eur"],
                "retained_at_smallest": (body.loc[lo, "total_pnl_eur"] / base
                                         if base else np.nan),
                "largest_frac": hi,
                "largest_min": body.loc[hi, "applied_min"],
                "pnl_at_largest": body.loc[hi, "total_pnl_eur"],
                "retained_at_largest": (body.loc[hi, "total_pnl_eur"] / base
                                        if base else np.nan),
                "worst_pnl": body["total_pnl_eur"].min(),
            }
        rows[key] = row
    return pd.DataFrame(rows).T.sort_values(f"hold_{stat}_h")


def plot_delay_sensitivity(registry: dict, market: dict,
                           fractions: tuple[float, ...] = HOLD_FRACTIONS,
                           fill: str = "cross", stat: str = "median",
                           logx: bool = True,
                           blotter_dir: str | Path = "../blotters",
                           out_dir: str | Path | None = "figures",
                           show: bool = False):
    """P&L against delay, with delay measured on each strategy's own clock."""
    from repricer import load_blotter

    plt = _plt(interactive=show)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key, entry in sorted(registry.items()):
        blot = load_blotter(Path(blotter_dir) / f"{key}-blotter.csv")
        sens = delay_sensitivity(entry, blot, market, fractions, fill, stat)
        scale = abs(sens["total_pnl_eur"].iloc[0]) or np.nan
        hold = holding_time(blot)[f"{stat}_h"]
        floor = resolution_floor(blot, stat)

        # x is the REQUESTED fraction: a log axis cannot show the realised one
        # where the book rounded it to zero. Points the book could not apply
        # are drawn hollow, so a flat run of them reads as "no data", not
        # "no effect".
        s = sens.drop(index=0.0, errors="ignore")
        live = ~s["below_resolution"].astype(bool)
        y = s["total_pnl_eur"] / scale
        line, = ax.plot(s.index, y, "-", lw=1.4,
                        label=f"{key}: holds {hold:.2f}h, floor {floor:.0%}")
        # `color=` on both, or the marker calls advance the colour cycle and
        # later series start repeating earlier ones
        c = line.get_color()
        ax.plot(s.index[live], y[live], "o", ms=5, color=c)
        ax.plot(s.index[~live], y[~live], "o", ms=5, color=c, mfc="white", mec=c)

    ax.axhline(0, color="k", lw=0.9)
    ax.axhline(1, color="grey", lw=0.7, ls=":")
    if logx:
        ax.set_xscale("log")
        ax.set_xticks(list(fractions))
        ax.set_xticklabels([f"{f:.0%}" for f in fractions])
        ax.minorticks_off()
    ax.set_xlabel(f"delay requested, as a fraction of that strategy's "
                  f"{stat} holding time")
    ax.set_ylabel("P&L / |undelayed P&L|")
    ax.set_title(f"execution delay at {fill} - below zero is losing money\n"
                 "hollow markers: below the 15-min book resolution, so no delay "
                 "was actually applied", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()

    if show:
        return fig
    return _save(fig, out_dir, f"delay_curves_{fill}.png")


def _daily_mwh(b: pd.DataFrame, window: tuple[str, str] | None) -> pd.Series:
    """MWh traded per delivery day, on the same index `daily_pnl` produces."""
    s = pd.Series(b["qty_mw"].values,
                  index=pd.DatetimeIndex(b["product_delivery"]))
    daily = s.groupby(s.index.normalize()).sum()
    if window is not None:
        idx = pd.date_range(window[0], window[1], freq="D", tz="UTC")
        daily = daily.reindex(idx, fill_value=0.0)
    return daily


def breakeven_fee(entry: dict, blotter: pd.DataFrame, market: dict,
                  fill: str = "cross") -> float:
    """The fee per MWh that takes P&L to exactly zero.

    A fee is charged on every MWh and always works against you, so it costs
    `fee * MWh` regardless of side or direction - which makes P&L exactly
    linear in the fee, and the breakeven fee exactly the edge per MWh.
    Negative when the strategy is already losing before any fee.
    """
    b, d = _priced(entry, blotter, market, fill)
    mwh = float(b["qty_mw"].sum())
    return float(d.sum()) / mwh if mwh else np.nan


def fee_sensitivity(entry: dict, blotter: pd.DataFrame, market: dict,
                    step: float = FEE_STEP, max_fee: float | None = None,
                    fill: str = "cross") -> pd.DataFrame:
    """How performance decays as a per-MWh fee is charged. One row per fee.

    Sweeps 0, `step`, 2*`step`, ... in EUR/MWh. `max_fee` defaults to one step
    past breakeven, so the grid always covers the point where the strategy
    dies (and at least 0.25 EUR/MWh, so a doomed strategy still gets a curve).

    P&L is exact rather than resampled: charging a fee subtracts `fee * MWh`
    from each day, so the whole sweep comes off one repricing.
    """
    window = (str(entry["backtest_window"]["start"]),
              str(entry["backtest_window"]["end"]))
    b, base = _priced(entry, blotter, market, fill)
    mwh_daily = _daily_mwh(b, window)
    mwh_total = float(b["qty_mw"].sum())

    if max_fee is None:
        be = float(base.sum()) / mwh_total if mwh_total else 0.0
        max_fee = max(np.ceil(max(be, 0.0) / step) * step + step, 0.25)

    fees = np.round(np.arange(0.0, max_fee + step / 2, step), 10)
    rows = {}
    for fee in fees:
        d = base - fee * mwh_daily          # exact: every MWh pays the fee
        rows[float(fee)] = {
            "total_pnl_eur": float(d.sum()),
            "sharpe_ann": sharpe(d),
            "sortino_ann": sortino(d),
            "max_drawdown_eur": max_drawdown(d),
            "hit_rate_active_days": hit_rate_active_days(d),
            "edge_eur_per_mwh": float(d.sum()) / mwh_total if mwh_total else np.nan,
        }
    out = pd.DataFrame(rows).T
    out.index.name = "fee_eur_mwh"
    base_pnl = out["total_pnl_eur"].iloc[0]
    out["pnl_retained"] = out["total_pnl_eur"] / base_pnl if base_pnl else np.nan
    return out


def fee_sensitivity_table(registry: dict, market: dict, step: float = FEE_STEP,
                          fill: str = "cross", n_steps: int = 6,
                          blotter_dir: str | Path = "../blotters") -> pd.DataFrame:
    """Breakeven fee per submission, plus P&L on a shared fee grid."""
    from repricer import load_blotter

    grid = np.round(np.arange(0.0, (n_steps + 0.5) * step, step), 10)
    rows = {}
    for key, entry in sorted(registry.items()):
        blot = load_blotter(Path(blotter_dir) / f"{key}-blotter.csv")
        sens = fee_sensitivity(entry, blot, market, step=step,
                               max_fee=float(grid[-1]), fill=fill)
        row = {"breakeven_fee": breakeven_fee(entry, blot, market, fill),
               "half_spread": float(half_spread(attach_book(blot, market)).mean())}
        row |= {f"pnl@{f:.2f}": sens.loc[f, "total_pnl_eur"] for f in grid}
        row["survives_5c"] = sens.loc[float(grid[1]), "total_pnl_eur"] > 0
        rows[key] = row
    return pd.DataFrame(rows).T.sort_values("breakeven_fee", ascending=False)


def plot_fee_sensitivity(entry: dict, blotter: pd.DataFrame, market: dict,
                         step: float = FEE_STEP, max_fee: float | None = None,
                         fill: str = "cross",
                         out_dir: str | Path | None = "figures",
                         show: bool = False):
    """P&L and Sharpe against the fee charged, with breakeven marked."""
    plt = _plt(interactive=show)
    sub = entry["submission"]
    sens = fee_sensitivity(entry, blotter, market, step, max_fee, fill)
    be = breakeven_fee(entry, blotter, market, fill)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sens.index, sens["total_pnl_eur"], "o-", ms=4, lw=1.6,
            color="#4c72b0", label="P&L (EUR)")
    ax.axhline(0, color="k", lw=0.8)
    if sens.index.min() <= be <= sens.index.max():
        ax.axvline(be, color="#c44e52", ls="--", lw=1.2,
                   label=f"breakeven {be:.3f} EUR/MWh")
    ax.set_xlabel("fee charged (EUR per MWh)")
    ax.set_ylabel("P&L (EUR)", color="#4c72b0")
    ax.set_title(f"{sub} - fee sensitivity at {fill}")

    ax2 = ax.twinx()
    ax2.plot(sens.index, sens["sharpe_ann"], "s--", ms=3, lw=1.2,
             color="#55a868", label="Sharpe")
    ax2.set_ylabel("Sharpe (annualised)", color="#55a868")

    lines = ax.get_lines()[:1] + ax2.get_lines()[:1] + \
        [l for l in ax.get_lines() if l.get_linestyle() == "--"]
    ax.legend(handles=lines, labels=[l.get_label() for l in lines], fontsize=8)
    fig.tight_layout()

    if show:
        return fig
    return _save(fig, out_dir, f"{sub}_fee_sensitivity.png")


def plot_fee_curves(registry: dict, market: dict, step: float = FEE_STEP,
                    max_fee: float = 0.50, fill: str = "cross",
                    blotter_dir: str | Path = "../blotters",
                    out_dir: str | Path | None = "figures", show: bool = False):
    """All seven on one shared fee grid: share of fee-free P&L retained."""
    from repricer import load_blotter

    plt = _plt(interactive=show)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for key, entry in sorted(registry.items()):
        blot = load_blotter(Path(blotter_dir) / f"{key}-blotter.csv")
        sens = fee_sensitivity(entry, blot, market, step, max_fee, fill)
        be = breakeven_fee(entry, blot, market, fill)
        # normalise by the ABSOLUTE fee-free P&L, so a strategy that is
        # already losing starts at -1 and falls, rather than dividing by a
        # negative base and appearing to improve as fees rise
        scale = abs(sens["total_pnl_eur"].iloc[0]) or np.nan
        line, = ax.plot(sens.index, sens["total_pnl_eur"] / scale, "o-",
                        ms=3, lw=1.4, label=f"{key} (breakeven {be:.2f})")
        if 0 <= be <= max_fee:
            ax.plot([be], [0], "X", ms=9, color=line.get_color())
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xlabel("fee charged (EUR per MWh)")
    ax.set_ylabel("P&L / |fee-free P&L|")
    ax.set_ylim(-1.6, 1.15)
    ax.set_title(f"fee sensitivity at {fill} - X marks breakeven, "
                 "below zero is losing money")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    if show:
        return fig
    return _save(fig, out_dir, f"fee_curves_{fill}.png")


def plot_performance(entry: dict, blotter: pd.DataFrame, market: dict,
                     fill: str = "cross", out_dir: str | Path | None = "figures",
                     show: bool = False):
    """Performance of one submission under one fill: equity, drawdown, months.

    Defaults to `cross`, so this is the strategy as it would actually have
    performed. Returns the saved path, or the figure when `show=True`.
    """
    plt = _plt(interactive=show)
    sub = entry["submission"]
    b, d = _priced(entry, blotter, market, fill)
    m = performance(entry, blotter, market, fill)

    cum = d.cumsum()
    dd = cum - cum.cummax()
    monthly = d.resample("ME").sum()
    colour = "#c44e52" if fill == "cross" else "#4c72b0"

    fig, (ax, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2.2, 1, 1.2]})

    ax.plot(cum.index, cum.values, lw=1.6, color=colour)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("cumulative P&L (EUR)")
    ax.set_title(f"{sub} - performance at {fill}\n"
                 f"P&L {m['total_pnl_eur']:,.0f} EUR   "
                 f"Sharpe {m['sharpe_ann']:.2f}   "
                 f"Sortino {m['sortino_ann']:.2f}   "
                 f"maxDD {m['max_drawdown_eur']:,.0f} EUR   "
                 f"edge {m['edge_eur_per_mwh']:.3f} EUR/MWh",
                 fontsize=10)

    ax2.fill_between(dd.index, dd.values, 0, color=colour, alpha=0.35, lw=0)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("drawdown (EUR)")

    ax3.bar(monthly.index, monthly.values, width=20,
            color=np.where(monthly.values >= 0, "#55a868", "#c44e52"))
    ax3.axhline(0, color="k", lw=0.8)
    ax3.set_ylabel("monthly P&L (EUR)")

    for a in (ax, ax2):
        a.set_xlim(cum.index[0], cum.index[-1])
    fig.tight_layout()

    if show:
        return fig
    return _save(fig, out_dir, f"{sub}_performance_{fill}.png")


def plot_crossing(entry: dict, blotter: pd.DataFrame, market: dict,
                  out_dir: str | Path | None = "figures", show: bool = False):
    """Cumulative P&L at mid vs. crossed, with the drawdown underneath.

    Returns the saved path, or the figure when `show=True` (for notebooks).
    """
    plt = _plt(interactive=show)
    sub = entry["submission"]
    window = (str(entry["backtest_window"]["start"]),
              str(entry["backtest_window"]["end"]))
    b = attach_book(blotter, market)
    b["px_mid"] = fill_price(b, Scenario("mid"))
    b["px_cross"] = fill_price(b, Scenario("cross", cross=True))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    for col, label, color in [("px_mid", "at mid (as claimed)", "#4c72b0"),
                              ("px_cross", "crossing the spread", "#c44e52")]:
        d = daily_pnl(b, col, window=window)
        cum = d.cumsum()
        ax.plot(cum.index, cum.values, lw=1.6, color=color,
                label=f"{label}: {d.sum():,.0f} EUR")
        ax2.fill_between(cum.index, (cum - cum.cummax()).values, 0,
                         color=color, alpha=0.35, lw=0)

    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("cumulative P&L (EUR)")
    ax.set_title(f"{sub} - what the spread costs")
    ax.legend(fontsize=9)
    ax2.set_ylabel("drawdown (EUR)")
    ax2.axhline(0, color="k", lw=0.8)
    fig.tight_layout()

    if show:
        return fig
    return _save(fig, out_dir, f"{sub}_crossing.png")


def crossing_summary(registry: dict, market: dict,
                     blotter_dir: str | Path = "../blotters") -> pd.DataFrame:
    """The crossing check across all seven: one row per submission."""
    from repricer import load_blotter

    rows = {}
    for key, entry in sorted(registry.items()):
        blot = load_blotter(Path(blotter_dir) / f"{key}-blotter.csv")
        chk = crossing_check(entry, blot, market)
        rows[key] = {
            "pnl_mid": chk.loc["total_pnl_eur", "mid"],
            "pnl_cross": chk.loc["total_pnl_eur", "cross"],
            "pnl_retained": (chk.loc["total_pnl_eur", "cross"]
                             / chk.loc["total_pnl_eur", "mid"]),
            "spread_cost_eur": chk.loc["total_pnl_eur", "cross_minus_mid"],
            "sharpe_mid": chk.loc["sharpe_ann", "mid"],
            "sharpe_cross": chk.loc["sharpe_ann", "cross"],
            "sortino_mid": chk.loc["sortino_ann", "mid"],
            "sortino_cross": chk.loc["sortino_ann", "cross"],
            "mdd_mid": chk.loc["max_drawdown_eur", "mid"],
            "mdd_cross": chk.loc["max_drawdown_eur", "cross"],
            "edge_mid": chk.loc["edge_eur_per_mwh", "mid"],
            "edge_cross": chk.loc["edge_eur_per_mwh", "cross"],
            "half_spread": chk.loc["half_spread_eur_mwh", "mid"],
        }
    return pd.DataFrame(rows).T


def sweep(registry: dict, market: dict, scenarios: list[Scenario],
          blotter_dir: str | Path = "../blotters") -> pd.DataFrame:
    """Run every submission through every scenario. Long format, one row each."""
    from repricer import load_blotter

    frames = []
    for key, entry in sorted(registry.items()):
        blot = load_blotter(Path(blotter_dir) / f"{key}-blotter.csv")
        w = (str(entry["backtest_window"]["start"]),
             str(entry["backtest_window"]["end"]))
        tbl = compare(blot, market, scenarios, w)
        tbl.insert(0, "submission", key)
        tbl.index.name = "scenario"
        frames.append(tbl.reset_index())
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# visuals
# --------------------------------------------------------------------------

def _plt(interactive: bool = False):
    """matplotlib, headless by default; `interactive=True` keeps the notebook
    backend so figures render inline instead of only to disk."""
    import matplotlib
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out_dir: str | Path, name: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    _plt().close(fig)
    return str(path)


def plot_decay(blotter: pd.DataFrame, market: dict, scenarios: list[Scenario],
               submission: str, window: tuple[str, str] | None = None,
               out_dir: str | Path = "figures") -> str:
    """Cumulative P&L under each scenario - the edge decaying, for one book."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cache: dict[int, pd.DataFrame] = {}
    for sc in scenarios:
        if sc.delay_snapshots not in cache:
            cache[sc.delay_snapshots] = attach_book(blotter, market, sc.delay_snapshots)
        b = cache[sc.delay_snapshots].copy()
        b[f"px_{sc.name}"] = fill_price(b, sc)
        d = daily_pnl(b, f"px_{sc.name}", window=window).cumsum()
        ax.plot(d.index, d.values, lw=1.5, label=sc.name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("cum P&L (EUR)")
    ax.set_title(f"{submission} - same trades, different execution assumptions")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out_dir, f"{submission}_decay.png")


def plot_retention_grid(swept: pd.DataFrame,
                        out_dir: str | Path = "figures") -> str:
    """Heatmap: share of mid P&L each submission keeps under each scenario."""
    plt = _plt()
    piv = swept.pivot(index="submission", columns="scenario", values="pnl_retained")
    piv = piv[[c for c in swept["scenario"].unique()]]
    piv = piv.loc[piv.iloc[:, -1].sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(1.5 + 1.3 * piv.shape[1], 1 + 0.5 * len(piv)))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv)), piv.index, fontsize=8)
    for i in range(len(piv)):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.0%}", ha="center", va="center",
                    fontsize=7.5)
    ax.set_title("share of mid P&L retained", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, out_dir, "retention_grid.png")


def plot_edge_vs_spread(registry: dict, market: dict,
                        blotter_dir: str | Path = "../blotters",
                        out_dir: str | Path = "figures") -> str:
    """Edge per MWh against the half-spread it has to clear, per submission."""
    from repricer import load_blotter

    plt = _plt()
    rows = []
    for key in sorted(registry):
        b = attach_book(load_blotter(Path(blotter_dir) / f"{key}-blotter.csv"), market)
        rows.append((key, edge_per_mwh(b), float(half_spread(b).mean())))
    df = pd.DataFrame(rows, columns=["sub", "edge", "half"]).sort_values("edge")

    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(df))
    ax.barh(y, df["edge"], color="#4c72b0", label="edge earned (EUR/MWh)")
    ax.plot(df["half"], y, "D", color="#c44e52", ms=7,
            label="half-spread paid (EUR/MWh)")
    ax.set_yticks(y, df["sub"], fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("EUR per MWh traded")
    ax.set_title("what each strategy earns vs. what it must pay to trade")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "edge_vs_spread.png")


if __name__ == "__main__":
    import importlib.util

    from repricer import load_blotter, load_market

    # resolve everything from this file, so the script behaves the same run
    # from the repo root as from inside starter/
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    FIGS = HERE / "figures"
    BLOT = ROOT / "blotters"
    FIGS.mkdir(exist_ok=True)

    # import registry/loader.py by path: there is a package called `loader`
    # on PyPI, and an installed one would shadow this project's module
    _spec = importlib.util.spec_from_file_location(
        "_gauntlet_registry_loader", ROOT / "registry" / "loader.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    market = load_market(ROOT / "data")
    registry = _mod.load_registry(ROOT / "registry")

    pd.set_option("display.width", 220, "display.max_columns", 40)

    # CHECK 1 - crossing the spread: what it costs
    summary = crossing_summary(registry, market, blotter_dir=BLOT)
    summary.to_csv(FIGS / "crossing_summary.csv")
    print("--- mid vs crossed ---")
    print(summary.round(3).to_string(), end="\n\n")

    # final performance, at the prices you would actually get
    perf = performance_table(registry, market, fill="cross",
                             blotter_dir=BLOT)
    perf.to_csv(FIGS / "performance_cross.csv")
    print("--- final performance at cross ---")
    print(perf.round(3).to_string(), end="\n\n")

    # CHECK 2 - sensitivity to fees, in 5 cent steps
    fees = fee_sensitivity_table(registry, market, blotter_dir=BLOT)
    fees.to_csv(FIGS / "fee_sensitivity.csv")
    print("--- fee sensitivity (EUR/MWh, 5 cent steps) ---")
    print(fees.round(3).to_string(), end="\n\n")
    print(plot_fee_curves(registry, market, blotter_dir=BLOT,
                          out_dir=FIGS))

    # CHECK 3 - execution delay, on each strategy's own clock
    delays = delay_summary(registry, market, blotter_dir=BLOT)
    delays.to_csv(FIGS / "delay_summary.csv")
    print("--- execution delay (as a fraction of mean holding time) ---")
    print(delays.round(3).to_string(), end="\n\n")
    print(plot_delay_sensitivity(registry, market, blotter_dir=BLOT,
                                 out_dir=FIGS))

    for key, entry in sorted(registry.items()):
        blot = load_blotter(BLOT / f"{key}-blotter.csv")
        crossing_check(entry, blot, market).to_csv(
            FIGS / f"{key}_crossing.csv")
        print(plot_crossing(entry, blot, market, out_dir=FIGS))
        print(plot_performance(entry, blot, market, fill="cross",
                               out_dir=FIGS))
        print(plot_fee_sensitivity(entry, blot, market, out_dir=FIGS))


# --------------------------------------------------------------------------
# size stress: what happens when the clip grows
# --------------------------------------------------------------------------

def _sweep_partial(px: np.ndarray, sz: np.ndarray,
                   qty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sweep levels 1..N, filling as much as the book holds.

    Unlike `_sweep_vwap` this does NOT give up when depth runs out: it
    returns the volume-weighted price of what WAS filled and how much that
    was, so a clip larger than the visible book is a partial fill rather than
    a blank. `filled < qty` is the finding.
    """
    px, sz = np.asarray(px, float), np.nan_to_num(np.asarray(sz, float))
    remaining = qty.astype(float).copy()
    cost, filled = np.zeros(len(qty)), np.zeros(len(qty))
    for lvl in range(px.shape[1]):
        take = np.minimum(remaining, sz[:, lvl])
        cost += take * np.nan_to_num(px[:, lvl])
        filled += take
        remaining -= take
    vwap = np.divide(cost, filled, out=np.full(len(qty), np.nan),
                     where=filled > 0)
    return vwap, filled


def sweep_at_multiplier(b: pd.DataFrame, multiplier: float
                        ) -> tuple[pd.Series, pd.Series]:
    """Price and filled size when every clip is scaled by `multiplier`.

    Both legs of a round trip scale together and meet the same depth, so
    positions still pair even when the book cannot fill them completely.
    """
    is_buy = b["side"].eq("BUY").values
    qty = b["qty_mw"].values * multiplier
    ask_px, ask_fill = _sweep_partial(b[[p for p, _ in ASK]].values,
                                      b[[s for _, s in ASK]].values, qty)
    bid_px, bid_fill = _sweep_partial(b[[p for p, _ in BID]].values,
                                      b[[s for _, s in BID]].values, qty)
    price = np.where(is_buy, ask_px, bid_px)
    filled = np.where(is_buy, ask_fill, bid_fill)
    return (pd.Series(price, index=b.index),
            pd.Series(filled, index=b.index))


def size_stress(entry: dict, blotter: pd.DataFrame, market: dict,
                multipliers: tuple[float, ...] = (0.5, 1, 2, 3, 4, 6, 8),
                ) -> pd.DataFrame:
    """Sweep the book at N times the recorded clip size.

    One row per multiplier: what the strategy would have made, how much of
    the scaled clip the book could actually absorb, and how often it ran out
    of depth entirely. Sharpe is scale-free in the P&L multiple - both mean
    and standard deviation scale with size - so it moves here only because
    the per-MWh cost of sweeping deeper does.
    """
    window = (str(entry["backtest_window"]["start"]),
              str(entry["backtest_window"]["end"]))
    b = attach_book(blotter, market)
    l1 = np.where(b["side"].eq("BUY"), b["ask_sz_1"], b["bid_sz_1"])
    visible = (b[[s for _, s in ASK]].sum(axis=1).where(
        b["side"].eq("BUY"), b[[s for _, s in BID]].sum(axis=1))).values

    rows = {}
    for m in multipliers:
        price, filled = sweep_at_multiplier(b, m)
        want = b["qty_mw"].values * m
        bb = b.assign(px=price, qty_mw=filled)      # P&L on what fills
        d = daily_pnl(bb, "px", window=window)
        frac = filled.values / want
        rows[m] = {
            "total_pnl_eur": float(d.sum()),
            "sharpe_ann": sharpe(d),
            "sortino_ann": sortino(d),
            "max_drawdown_eur": max_drawdown(d),
            "clip_mw": float(np.median(want)),
            "mean_fill_frac": float(np.nanmean(frac)),
            "pct_depth_exhausted": float(np.mean(want > visible + 1e-9)),
            "pct_past_touch": float(np.mean(want > l1 + 1e-9)),
            "mwh_filled": float(filled.sum()),
            "mwh_wanted": float(want.sum()),
            "eur_per_mwh": (float(d.sum()) / float(filled.sum())
                            if filled.sum() else np.nan),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "multiplier"
    base = out.loc[1.0, "total_pnl_eur"] if 1.0 in out.index else np.nan
    out["pnl_vs_1x"] = out["total_pnl_eur"] - base
    return out


def best_multiplier(stress: pd.DataFrame, metric: str = "sharpe_ann",
                    require_full_fill: bool = False, tol: float = 0.005) -> dict:
    """The clip multiplier that maximises `metric` in a `size_stress` table.

    Sharpe is scale-free - doubling every trade doubles both the mean and the
    standard deviation - so it does not improve with size and there is no
    interior optimum to find. It is FLAT across every multiplier whose clip
    still fits in the depth at the touch, then falls as the clip is forced
    deeper. The useful answer is therefore the LARGEST multiplier that is
    still within `tol` of the best value: the biggest the clip can get before
    size starts costing anything.

    With `require_full_fill`, only multipliers the book can absorb entirely
    are eligible - otherwise the winner may be one that quietly trades less
    than it asked for.
    """
    t = stress
    if require_full_fill:
        t = t[t["mean_fill_frac"] >= 0.999]
    if not len(t) or t[metric].isna().all():
        return {"multiplier": np.nan, metric: np.nan}
    best = float(t[metric].max())
    ties = t[t[metric] >= best - abs(best) * tol]
    row = float(ties.index.max())
    return {"multiplier": row,
            metric: float(t.loc[row, metric]),
            "best_value": best,
            "tied_from": float(ties.index.min()),
            "clip_mw": float(t.loc[row, "clip_mw"]),
            "total_pnl_eur": float(t.loc[row, "total_pnl_eur"]),
            "mean_fill_frac": float(t.loc[row, "mean_fill_frac"]),
            "pct_depth_exhausted": float(t.loc[row, "pct_depth_exhausted"])}
