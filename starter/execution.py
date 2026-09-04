"""Execution style - aggressive, half-passive, fully passive, and VWAP.

Everything in `friction.py` reprices against the ORDER BOOK. This module adds
the questions only the TRADE TAPE (`data/trades.csv`) can answer: would a
resting order have been filled at all, how much of it, and how long would it
take to work a clip through real volume.

Four execution styles, from certain-fill-expensive to cheap-but-uncertain:

    aggressive     cross immediately: buy ask_1, sell bid_1. Always fills.
    half_passive   post at our own touch; whatever has not filled by the time
                   the strategy meant to be flat is crossed out. What a desk
                   actually does.
    passive        post at our own touch and never voluntarily cross; only a
                   position still open at gate is force-flattened. The
                   pessimistic bound - you learn the exit never filled when
                   delivery forces your hand.
    vwap           work the clip as a share of tape volume, paying the volume
                   weighted price of the prints participated in, and taking
                   however long that takes.

No verdicts here either: these are numbers and distributions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import friction as fr
from repricer import daily_pnl, sharpe

STYLES = ("aggressive", "half_passive", "passive")

DEFAULT_PARTICIPATION = 0.20   # share of tape volume a VWAP order may take


# --------------------------------------------------------------------------
# the tape
# --------------------------------------------------------------------------

@dataclass
class Tape:
    """The trade tape, indexed per product for interval queries.

    `trade_ts` is continuous - it does NOT sit on the book's 15-minute grid -
    and sizes are fractional, so this is a real tape rather than a
    snapshot-aligned aggregate.
    """
    ts: np.ndarray
    px: np.ndarray
    qty: np.ndarray
    sell_aggressor: np.ndarray
    bounds: dict[int, tuple[int, int]]

    @classmethod
    def load(cls, data_dir: str | Path = "../data") -> "Tape":
        t = (pd.read_csv(Path(data_dir) / "trades.csv", parse_dates=["trade_ts"])
             .sort_values(["product_id", "trade_ts"]).reset_index(drop=True))
        pid = t["product_id"].values
        edges = np.flatnonzero(np.diff(pid)) + 1
        bounds = {pid[s]: (int(s), int(e))
                  for s, e in zip(np.r_[0, edges], np.r_[edges, len(pid)])}
        return cls(t["trade_ts"].values, t["price"].values, t["qty_mw"].values,
                   (t["aggressor"].values == "SELL"), bounds)

    def _slice(self, pid: int, t0, t1) -> slice:
        lo, hi = self.bounds.get(pid, (0, 0))
        if lo == hi:
            return slice(0, 0)
        a = lo + int(np.searchsorted(self.ts[lo:hi], t0, side="right"))
        b = lo + int(np.searchsorted(self.ts[lo:hi], t1, side="right"))
        return slice(a, max(a, b))

    def passive_volume(self, pid: int, t0, t1, limit: float, want_seller: bool,
                       queue_ahead: float = 0.0) -> float:
        """Aggressive flow that would have hit a resting order at `limit`.

        A seller only hits our bid at or below our limit; a buyer only lifts
        our offer at or above it. `queue_ahead` is volume that must clear
        before us - front of queue (0.0) is the optimistic case.
        """
        s = self._slice(pid, t0, t1)
        if s.start == s.stop:
            return 0.0
        m = self.sell_aggressor[s] == want_seller
        px = self.px[s]
        m &= (px <= limit) if want_seller else (px >= limit)
        return max(float(self.qty[s][m].sum()) - queue_ahead, 0.0)

    def vwap_fill(self, pid: int, t0, t1, qty: float,
                  participation: float) -> tuple[float, float, float]:
        """Work `qty` as a share of tape volume from `t0`.

        Returns (vwap, minutes_to_fill, filled_mw). Taking `participation` of
        every print means the clip is not filled instantly and not
        necessarily at all: `filled_mw < qty` means the tape ran out of
        volume before `t1`, and `minutes_to_fill` is then the whole window.
        """
        s = self._slice(pid, t0, t1)
        if s.start == s.stop or qty <= 0:
            return np.nan, np.nan, 0.0
        take = np.minimum(self.qty[s] * participation, qty)
        cum = np.cumsum(take)
        done = int(np.searchsorted(cum, qty, side="left"))
        if done >= len(cum):                      # never completes
            filled = float(cum[-1])
            mins = (self.ts[s][-1] - t0) / np.timedelta64(1, "m")
            vwap = float((take * self.px[s]).sum() / filled) if filled else np.nan
            return vwap, float(mins), filled
        # trim the last print to land exactly on qty
        take = take[:done + 1].copy()
        over = float(cum[done] - qty)
        take[-1] -= over
        vwap = float((take * self.px[s][:done + 1]).sum() / qty)
        mins = (self.ts[s][done] - t0) / np.timedelta64(1, "m")
        return vwap, float(mins), float(qty)


# --------------------------------------------------------------------------
# passive / half-passive / aggressive
# --------------------------------------------------------------------------

def _prepare(entry: dict, blotter: pd.DataFrame, market: dict) -> pd.DataFrame:
    """Blotter joined to the book, sorted per product, with our limit prices."""
    b = (fr.attach_book(blotter, market)
         .sort_values(["product_id", "exec_ts"]).reset_index(drop=True))
    is_buy = b["side"].eq("BUY").values
    b["is_buy"] = is_buy
    # posting means quoting on our OWN side: we earn the half-spread
    b["limit_px"] = np.where(is_buy, b["bid_px_1"], b["ask_px_1"])
    b["cross_px"] = np.where(is_buy, b["ask_px_1"], b["bid_px_1"])
    return b


def passive_fills(entry: dict, blotter: pd.DataFrame, market: dict, tape: Tape,
                  queue_ahead: float = 0.0) -> pd.DataFrame:
    """How much of each order the tape would actually have filled passively.

    Each order rests from its `exec_ts` until the strategy's next trade in
    that product - that is when it changed its mind - or the product's last
    pre-gate snapshot if there is no next trade.
    """
    b = _prepare(entry, blotter, market)
    last = market["book"].groupby("product_id")["ts"].max()
    rest_until = (b.groupby("product_id")["exec_ts"].shift(-1)
                  .fillna(b["product_id"].map(last)))
    b["rest_until"] = rest_until
    b["filled_mw"] = [
        min(tape.passive_volume(p, t0, t1, lim, buy, queue_ahead), q)
        for p, t0, t1, lim, buy, q in zip(
            b["product_id"].values, b["exec_ts"].values, rest_until.values,
            b["limit_px"].values, b["is_buy"].values, b["qty_mw"].values)]
    b["fill_frac"] = b["filled_mw"] / b["qty_mw"]
    return b


def style_pnl(entry: dict, blotter: pd.DataFrame, market: dict, tape: Tape,
              style: str = "half_passive",
              queue_ahead: float = 0.0) -> tuple[pd.Series, dict]:
    """Daily P&L under one execution style, plus what it took to get there.

    `aggressive` crosses everything at once. The two passive styles differ
    only in WHEN an unfilled remainder is force-closed: `half_passive` at the
    moment the strategy meant to be flat, `passive` not until gate.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {list(STYLES)}, got {style!r}")
    window = (str(entry["backtest_window"]["start"]),
              str(entry["backtest_window"]["end"]))

    if style == "aggressive":
        b = _prepare(entry, blotter, market)
        b["px"] = b["cross_px"]
        daily = daily_pnl(b, "px", window=window)
        return daily, {"mean_fill": 1.0, "filled_mw": float(b["qty_mw"].sum()),
                       "intended_mw": float(b["qty_mw"].sum()),
                       "forced_mw": 0.0, "forced_events": 0}

    b = passive_fills(entry, blotter, market, tape, queue_ahead)
    last = market["book"].groupby("product_id")["ts"].max()
    touch = market["book"].set_index(["product_id", "ts"])[["bid_px_1",
                                                            "ask_px_1"]]

    sign = np.where(b["is_buy"].values, 1.0, -1.0)
    pos_intended = sign * b["qty_mw"].values
    pos_actual = sign * b["filled_mw"].values
    cash = -pos_actual * b["limit_px"].values      # buying pays out
    pid = b["product_id"].values
    bid1, ask1 = b["bid_px_1"].values, b["ask_px_1"].values

    per_product, forced_mw, forced_n = {}, 0.0, 0
    i, n = 0, len(b)
    while i < n:
        j = i
        p = pid[i]
        while j < n and pid[j] == p:
            j += 1
        c = intended = actual = 0.0
        for k in range(i, j):
            c += cash[k]
            intended += pos_intended[k]
            actual += pos_actual[k]
            # half_passive: the strategy meant to be flat here, so take the
            # spread rather than carry the position any further
            if (style == "half_passive" and abs(intended) < 1e-9
                    and abs(actual) > 1e-9):
                c += actual * (bid1[k] if actual > 0 else ask1[k])
                forced_mw += abs(actual)
                forced_n += 1
                actual = 0.0
        if abs(actual) > 1e-9:                     # gate forces what is left
            ts = last[p]
            bid, ask = touch.loc[(p, ts)]
            c += actual * (bid if actual > 0 else ask)
            forced_mw += abs(actual)
            forced_n += 1
        per_product[p] = c
        i = j

    deliv = market["products"].set_index("product_id")["delivery_start"]
    s = pd.Series(per_product)
    # keep the tz: .values on a tz-aware series drops it and the reindex
    # below would then match nothing and silently return all zeros
    s.index = pd.DatetimeIndex(deliv.reindex(s.index))
    daily = (s.groupby(pd.DatetimeIndex(s.index).normalize()).sum()
             .reindex(pd.date_range(window[0], window[1], freq="D", tz="UTC"),
                      fill_value=0.0))
    return daily, {
        "mean_fill": float(b["fill_frac"].mean()),
        "filled_mw": float(b["filled_mw"].sum()),
        "intended_mw": float(b["qty_mw"].sum()),
        "forced_mw": forced_mw, "forced_events": forced_n,
    }


def style_comparison(entry: dict, blotter: pd.DataFrame, market: dict,
                     tape: Tape, queue_ahead: float = 0.0) -> pd.DataFrame:
    """One row per execution style: performance and what it cost to get it."""
    rows = {}
    for style in STYLES:
        daily, info = style_pnl(entry, blotter, market, tape, style,
                                queue_ahead)
        rows[style] = {
            "total_pnl_eur": float(daily.sum()),
            "sharpe_ann": sharpe(daily),
            "sortino_ann": fr.sortino(daily),
            "max_drawdown_eur": fr.max_drawdown(daily),
            "mean_fill_frac": info["mean_fill"],
            "filled_mw": info["filled_mw"],
            "intended_mw": info["intended_mw"],
            "forced_close_mw": info["forced_mw"],
            "forced_closes": info["forced_events"],
        }
    out = pd.DataFrame(rows).T
    base = out.loc["aggressive", "total_pnl_eur"]
    out["vs_aggressive_eur"] = out["total_pnl_eur"] - base
    return out


# --------------------------------------------------------------------------
# VWAP participation and the time it takes
# --------------------------------------------------------------------------

def vwap_execution(entry: dict, blotter: pd.DataFrame, market: dict,
                   tape: Tape,
                   participation: float = DEFAULT_PARTICIPATION) -> pd.DataFrame:
    """Work every clip as a share of tape volume; price and time per trade.

    Adds `vwap_px`, `minutes_to_fill` and `filled_mw`. A clip that the tape
    cannot complete before gate is left partly filled, and its
    `minutes_to_fill` is the whole remaining window - those rows are the
    finding, not an error.
    """
    b = _prepare(entry, blotter, market)
    last = market["book"].groupby("product_id")["ts"].max()
    deadline = b["product_id"].map(last)

    out = [tape.vwap_fill(p, t0, t1, q, participation)
           for p, t0, t1, q in zip(b["product_id"].values, b["exec_ts"].values,
                                   deadline.values, b["qty_mw"].values)]
    b["vwap_px"], b["minutes_to_fill"], b["vwap_filled_mw"] = map(
        np.array, zip(*out))
    b["vwap_fill_frac"] = b["vwap_filled_mw"] / b["qty_mw"]
    b["minutes_available"] = ((deadline - b["exec_ts"]).dt.total_seconds()
                              / 60.0).values
    return b


def vwap_summary(entry: dict, blotter: pd.DataFrame, market: dict, tape: Tape,
                 participation: float = DEFAULT_PARTICIPATION) -> dict:
    """Time-to-fill statistics, and the P&L of paying the VWAP achieved.

    `unfilled_clips` is the headline: a clip the tape cannot absorb before
    gate was never executable at that size and participation rate.

    P&L is reported only over products where EVERY leg filled. Dropping
    unfilled legs from a product would leave one-sided P&L - all entries and
    no exits - which is not a conservative number but a meaningless one.
    `pnl_coverage_frac` says how much of the book that leaves.
    """
    window = (str(entry["backtest_window"]["start"]),
              str(entry["backtest_window"]["end"]))
    b = vwap_execution(entry, blotter, market, tape, participation)
    done = b["vwap_fill_frac"] >= 0.999
    mins = b.loc[done, "minutes_to_fill"]

    # only products whose every leg filled can be priced; a product missing a
    # leg has no round trip and its P&L would be a naked one-sided number
    ok = b.assign(done=done).groupby("product_id")["done"].transform("all")
    priced = b[ok].copy()
    if len(priced):
        daily = daily_pnl(priced.assign(px=priced["vwap_px"]), "px",
                          window=window)
        pnl, shp = float(daily.sum()), sharpe(daily)
    else:
        pnl, shp = np.nan, np.nan

    return {
        "participation": participation,
        "clips": int(len(b)),
        "unfilled_clips": int((~done).sum()),
        "unfilled_frac": float((~done).mean()),
        "mean_fill_frac": float(b["vwap_fill_frac"].mean()),
        "median_min_to_fill": float(mins.median()) if len(mins) else np.nan,
        "p90_min_to_fill": float(mins.quantile(0.9)) if len(mins) else np.nan,
        "max_min_to_fill": float(mins.max()) if len(mins) else np.nan,
        # zero for a leg that sits on the last pre-gate snapshot: it has no
        # forward window to fill in at all
        "median_min_available": float(b["minutes_available"].median()),
        "min_min_available": float(b["minutes_available"].min()),
        "pnl_at_vwap_eur": pnl,
        "sharpe_at_vwap": shp,
        "pnl_products_covered": int(priced["product_id"].nunique()),
        "pnl_coverage_frac": (float(ok.mean()) if len(b) else np.nan),
    }


def participation_sweep(entry: dict, blotter: pd.DataFrame, market: dict,
                        tape: Tape,
                        rates: tuple[float, ...] = (0.05, 0.10, 0.20, 0.50, 1.00)
                        ) -> pd.DataFrame:
    """Time-to-fill and VWAP P&L across participation rates."""
    rows = {r: vwap_summary(entry, blotter, market, tape, r) for r in rates}
    out = pd.DataFrame(rows).T
    out.index.name = "participation"
    return out


def tape_share(entry: dict, blotter: pd.DataFrame, market: dict,
               tape: Tape) -> dict:
    """How much of the real market this strategy would have had to be.

    No counterfactual: its own volume against the volume that actually
    traded, overall and in the products it touched.
    """
    b = fr.attach_book(blotter, market)
    ours = b.groupby("product_id")["qty_mw"].sum()
    vol = {p: float(tape.qty[lo:hi].sum())
           for p, (lo, hi) in tape.bounds.items()}
    tape_total = sum(vol.values())
    share = (ours / pd.Series({p: vol.get(p, np.nan)
                               for p in ours.index})).dropna()
    return {
        "our_mwh": float(ours.sum()),
        "tape_mwh": tape_total,
        "share_of_whole_tape": float(ours.sum()) / tape_total,
        "products_touched": int(len(ours)),
        "median_share_of_product": float(share.median()),
        "p90_share_of_product": float(share.quantile(0.9)),
        "max_share_of_product": float(share.max()),
    }


# --------------------------------------------------------------------------
# cross-submission summaries
# --------------------------------------------------------------------------

def style_summary(registry: dict, market: dict, tape: Tape,
                  queue_ahead: float = 0.0,
                  blotter_dir: str | Path = "../blotters") -> pd.DataFrame:
    """Aggressive vs half-passive vs fully passive, one row per submission."""
    from repricer import load_blotter

    rows = {}
    for key, entry in sorted(registry.items()):
        blot = load_blotter(Path(blotter_dir) / f"{key}-blotter.csv")
        cmp = style_comparison(entry, blot, market, tape, queue_ahead)
        rows[key] = {
            "pnl_aggressive": cmp.loc["aggressive", "total_pnl_eur"],
            "pnl_half_passive": cmp.loc["half_passive", "total_pnl_eur"],
            "pnl_passive": cmp.loc["passive", "total_pnl_eur"],
            "sharpe_aggressive": cmp.loc["aggressive", "sharpe_ann"],
            "sharpe_half_passive": cmp.loc["half_passive", "sharpe_ann"],
            "sharpe_passive": cmp.loc["passive", "sharpe_ann"],
            "mdd_aggressive": cmp.loc["aggressive", "max_drawdown_eur"],
            "mdd_half_passive": cmp.loc["half_passive", "max_drawdown_eur"],
            "mdd_passive": cmp.loc["passive", "max_drawdown_eur"],
            "mean_fill_frac": cmp.loc["passive", "mean_fill_frac"],
            "filled_mw": cmp.loc["passive", "filled_mw"],
            "intended_mw": cmp.loc["passive", "intended_mw"],
            "forced_close_mw": cmp.loc["half_passive", "forced_close_mw"],
        }
    return pd.DataFrame(rows).T.sort_values("pnl_aggressive", ascending=False)


def vwap_time_summary(registry: dict, market: dict, tape: Tape,
                      participation: float = DEFAULT_PARTICIPATION,
                      blotter_dir: str | Path = "../blotters") -> pd.DataFrame:
    """Time to work a clip through real volume, against how long it is held.

    `fill_vs_hold` is the ratio that matters: above 1.0 the strategy needs
    longer to build the position than it intends to keep it, which means the
    trade it recorded was not executable at that participation rate.
    """
    from repricer import load_blotter

    rows = {}
    for key, entry in sorted(registry.items()):
        blot = load_blotter(Path(blotter_dir) / f"{key}-blotter.csv")
        s = vwap_summary(entry, blot, market, tape, participation)
        hold_min = fr.holding_time(blot)["median_h"] * 60.0
        rows[key] = {
            "hold_median_min": hold_min,
            "median_min_to_fill": s["median_min_to_fill"],
            "p90_min_to_fill": s["p90_min_to_fill"],
            "fill_vs_hold": (s["median_min_to_fill"] / hold_min
                             if hold_min else np.nan),
            "unfilled_clips": s["unfilled_clips"],
            "unfilled_frac": s["unfilled_frac"],
            "min_min_available": s["min_min_available"],
            "pnl_at_vwap_eur": s["pnl_at_vwap_eur"],
            "pnl_coverage_frac": s["pnl_coverage_frac"],
        }
    return pd.DataFrame(rows).T.sort_values("fill_vs_hold", ascending=False)


def tape_share_summary(registry: dict, market: dict, tape: Tape,
                       blotter_dir: str | Path = "../blotters") -> pd.DataFrame:
    """Each submission's share of the volume that actually traded."""
    from repricer import load_blotter

    rows = {}
    for key, entry in sorted(registry.items()):
        blot = load_blotter(Path(blotter_dir) / f"{key}-blotter.csv")
        rows[key] = tape_share(entry, blot, market, tape)
    return pd.DataFrame(rows).T.sort_values("median_share_of_product",
                                            ascending=False)


# --------------------------------------------------------------------------
# market absorption: how long the market takes to trade our size at all
# --------------------------------------------------------------------------

def _absorb_minutes(tape: Tape, pid: int, t0, t1, qty: float) -> float:
    """Minutes until cumulative tape volume from `t0` reaches `qty`.

    This is the market absorbing our size at 100% - every print counts, we
    take all of it. It is the floor on execution time, not an estimate of
    it: any realistic participation rate is slower. NaN if the tape never
    gets there before `t1`.
    """
    s = tape._slice(pid, t0, t1)
    if s.start == s.stop or qty <= 0:
        return np.nan
    cum = np.cumsum(tape.qty[s])
    i = int(np.searchsorted(cum, qty, side="left"))
    if i >= len(cum):
        return np.nan
    return float((tape.ts[s][i] - t0) / np.timedelta64(1, "m"))


def absorption(entry: dict, blotter: pd.DataFrame, market: dict, tape: Tape,
               ) -> pd.DataFrame:
    """Per trade: how our clip compares to what the market normally trades.

    Adds `absorb_min` (minutes for the tape to trade our size), `hold_min`
    (this strategy's median holding time) and `absorbed_in_hold`.
    """
    b = _prepare(entry, blotter, market)
    last = market["book"].groupby("product_id")["ts"].max()
    deadline = b["product_id"].map(last)
    hold_min = fr.holding_time(blotter)["median_h"] * 60.0

    b["absorb_min"] = [
        _absorb_minutes(tape, p, t0, t1, q)
        for p, t0, t1, q in zip(b["product_id"].values, b["exec_ts"].values,
                                deadline.values, b["qty_mw"].values)]
    b["hold_min"] = hold_min
    b["absorbed_in_hold"] = b["absorb_min"] <= hold_min
    b["minutes_available"] = ((deadline - b["exec_ts"]).dt.total_seconds()
                              / 60.0).values
    return b


def absorption_summary(entry: dict, blotter: pd.DataFrame, market: dict,
                       tape: Tape) -> dict:
    """Typical traded size, our size against it, and whether it is absorbed.

    The question this answers: is the size this strategy trades a normal
    amount for this market to move in the time the strategy gives it?
    """
    b = absorption(entry, blotter, market, tape)
    pids = set(b["product_id"].unique())

    prints, per_product = [], []
    for p in pids:
        lo, hi = tape.bounds.get(p, (0, 0))
        if lo == hi:
            continue
        prints.append(tape.qty[lo:hi])
        per_product.append(float(tape.qty[lo:hi].sum()))
    sizes = np.concatenate(prints) if prints else np.array([np.nan])

    clip = float(np.median(b["qty_mw"]))
    hold_min = float(b["hold_min"].iloc[0])
    absorbed = b["absorb_min"].dropna()
    return {
        "clip_mw": clip,
        "median_print_mw": float(np.median(sizes)),
        "mean_print_mw": float(np.mean(sizes)),
        "p90_print_mw": float(np.quantile(sizes, 0.9)),
        "p95_print_mw": float(np.quantile(sizes, 0.95)),
        "p99_print_mw": float(np.quantile(sizes, 0.99)),
        "p25_print_mw": float(np.quantile(sizes, 0.25)),
        "p75_print_mw": float(np.quantile(sizes, 0.75)),
        "clip_vs_median_print": clip / float(np.median(sizes)),
        # a whole product's traded volume, for scale
        "median_product_volume_mwh": float(np.median(per_product))
        if per_product else np.nan,
        "clip_vs_product_volume": (clip / float(np.median(per_product))
                                   if per_product else np.nan),
        "hold_median_min": hold_min,
        "median_absorb_min": float(absorbed.median()) if len(absorbed) else np.nan,
        "p90_absorb_min": (float(absorbed.quantile(0.9)) if len(absorbed)
                           else np.nan),
        "absorbed_within_hold_frac": float(b["absorbed_in_hold"].mean()),
        "never_absorbed_frac": float(b["absorb_min"].isna().mean()),
        "absorb_vs_hold": (float(absorbed.median()) / hold_min
                           if len(absorbed) and hold_min else np.nan),
    }


def absorption_table(registry: dict, market: dict, tape: Tape,
                     blotter_dir: str | Path = "../blotters") -> pd.DataFrame:
    """Absorption for every submission, one row each."""
    from repricer import load_blotter

    rows = {}
    for key, entry in sorted(registry.items()):
        blot = load_blotter(Path(blotter_dir) / f"{key}-blotter.csv")
        rows[key] = absorption_summary(entry, blot, market, tape)
    return pd.DataFrame(rows).T.sort_values("absorb_vs_hold", ascending=False)
