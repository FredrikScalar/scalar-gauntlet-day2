"""Tests for execution.py, on a hand-built book and tape.

No real data: every fixture is small enough to check by hand, so a failure
points at the code rather than at the market.

    pytest test_execution.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from execution import (STYLES, Tape, absorption, absorption_summary,
                       passive_fills, style_comparison,
                       style_pnl, vwap_execution, vwap_summary)

TS = pd.Timestamp("2025-01-01 10:00", tz="UTC")
DELIV = pd.Timestamp("2025-01-01 12:00", tz="UTC")
M15 = pd.Timedelta(minutes=15)


def _book_row(ts, pid, bid1=99.0, ask1=101.0, step=1.0, size=10.0):
    row = {"ts": ts, "product_id": pid}
    for lvl in range(1, 4):
        row[f"bid_px_{lvl}"] = bid1 - (lvl - 1) * step
        row[f"ask_px_{lvl}"] = ask1 + (lvl - 1) * step
        row[f"bid_sz_{lvl}"] = size
        row[f"ask_sz_{lvl}"] = size
    return row


@pytest.fixture
def market():
    """One product, five snapshots 15 min apart. Bid 99 / ask 101, mid 100."""
    book = pd.DataFrame([_book_row(TS + k * M15, 0) for k in range(5)])
    book["mid"] = (book["bid_px_1"] + book["ask_px_1"]) / 2.0
    return {"book": book,
            "products": pd.DataFrame([{"product_id": 0,
                                       "delivery_start": DELIV}])}


@pytest.fixture
def entry():
    return {"submission": "unit-test",
            "backtest_window": {"start": "2025-01-01", "end": "2025-01-01"}}


@pytest.fixture
def blotter():
    """Buy 10 MW, sell it back 30 min later. Intended position returns flat."""
    return pd.DataFrame([
        {"exec_ts": TS, "product_delivery": DELIV, "side": "BUY",
         "qty_mw": 10.0, "price": 100.0},
        {"exec_ts": TS + 2 * M15, "product_delivery": DELIV, "side": "SELL",
         "qty_mw": 10.0, "price": 100.0},
    ])


def _tape(rows) -> Tape:
    t = pd.DataFrame(rows).sort_values(["product_id", "trade_ts"])
    pid = t["product_id"].values
    edges = np.flatnonzero(np.diff(pid)) + 1
    bounds = {pid[s]: (int(s), int(e))
              for s, e in zip(np.r_[0, edges], np.r_[edges, len(pid)])}
    return Tape(t["trade_ts"].values, t["price"].values, t["qty_mw"].values,
                (t["aggressor"].values == "SELL"), bounds)


def _print(mins, px, qty, aggressor, pid=0):
    return {"trade_ts": TS + pd.Timedelta(minutes=mins), "product_id": pid,
            "price": px, "qty_mw": qty, "aggressor": aggressor}


# ---- the tape's passive-fill rule -----------------------------------------

def test_a_seller_at_our_limit_fills_a_resting_bid():
    tape = _tape([_print(5, 99.0, 4.0, "SELL")])
    got = tape.passive_volume(0, TS.to_datetime64(), (TS + M15).to_datetime64(),
                              limit=99.0, want_seller=True)
    assert got == pytest.approx(4.0)


def test_a_seller_above_our_limit_does_not_fill_us():
    """A seller happy with 100 would never hand us 99."""
    tape = _tape([_print(5, 100.0, 4.0, "SELL")])
    assert tape.passive_volume(0, TS.to_datetime64(), (TS + M15).to_datetime64(),
                               limit=99.0, want_seller=True) == 0.0


def test_a_buyer_does_not_fill_a_resting_bid():
    """Only the opposite aggressor can hit us."""
    tape = _tape([_print(5, 99.0, 4.0, "BUY")])
    assert tape.passive_volume(0, TS.to_datetime64(), (TS + M15).to_datetime64(),
                               limit=99.0, want_seller=True) == 0.0


def test_a_buyer_at_or_above_our_offer_lifts_it():
    tape = _tape([_print(5, 101.0, 3.0, "BUY"), _print(6, 102.0, 2.0, "BUY")])
    got = tape.passive_volume(0, TS.to_datetime64(), (TS + M15).to_datetime64(),
                              limit=101.0, want_seller=False)
    assert got == pytest.approx(5.0)


def test_queue_ahead_is_subtracted_before_we_fill():
    tape = _tape([_print(5, 99.0, 8.0, "SELL")])
    got = tape.passive_volume(0, TS.to_datetime64(), (TS + M15).to_datetime64(),
                              limit=99.0, want_seller=True, queue_ahead=6.0)
    assert got == pytest.approx(2.0)


def test_prints_outside_the_resting_window_do_not_count():
    tape = _tape([_print(30, 99.0, 5.0, "SELL")])
    assert tape.passive_volume(0, TS.to_datetime64(), (TS + M15).to_datetime64(),
                               limit=99.0, want_seller=True) == 0.0


# ---- fills through to a fill fraction -------------------------------------

def test_partial_fill_is_capped_at_the_clip(market, entry, blotter):
    """4 MW of eligible flow against a 10 MW order is a 40% fill."""
    tape = _tape([_print(5, 99.0, 4.0, "SELL"), _print(40, 101.0, 4.0, "BUY")])
    f = passive_fills(entry, blotter, market, tape)
    assert f.loc[0, "filled_mw"] == pytest.approx(4.0)
    assert f.loc[0, "fill_frac"] == pytest.approx(0.4)


def test_no_eligible_flow_means_no_fill_at_all(market, entry, blotter):
    tape = _tape([_print(5, 100.5, 50.0, "SELL")])     # never reaches our bid
    f = passive_fills(entry, blotter, market, tape)
    assert f["filled_mw"].sum() == 0.0


# ---- the three styles ------------------------------------------------------

def test_aggressive_always_fills_and_pays_the_spread(market, entry, blotter):
    tape = _tape([_print(5, 99.0, 1.0, "SELL")])
    daily, info = style_pnl(entry, blotter, market, tape, "aggressive")
    # buy at 101, sell at 99, 10 MW -> -20
    assert daily.sum() == pytest.approx(-20.0)
    assert info["mean_fill"] == 1.0
    assert info["forced_mw"] == 0.0


def test_fully_passive_earns_the_spread_when_both_legs_fill(market, entry,
                                                            blotter):
    """Buy filled at 99, sell filled at 101: the mirror of crossing."""
    tape = _tape([_print(5, 99.0, 10.0, "SELL"), _print(35, 101.0, 10.0, "BUY")])
    daily, info = style_pnl(entry, blotter, market, tape, "passive")
    assert daily.sum() == pytest.approx(20.0)
    assert info["forced_mw"] == pytest.approx(0.0)
    assert info["filled_mw"] == pytest.approx(20.0)


def test_an_unfilled_exit_becomes_a_forced_close(market, entry, blotter):
    """Entry fills, exit does not: 10 MW long must be sold at some point."""
    tape = _tape([_print(5, 99.0, 10.0, "SELL")])
    for style in ("half_passive", "passive"):
        _, info = style_pnl(entry, blotter, market, tape, style)
        assert info["forced_mw"] == pytest.approx(10.0)
        assert info["forced_events"] == 1


def test_half_passive_closes_earlier_than_passive(market, entry):
    """The two differ only in WHEN the remainder is crossed out, so put a
    price move between the intended exit and the gate and they must differ."""
    book = pd.DataFrame([_book_row(TS, 0), _book_row(TS + M15, 0),
                         _book_row(TS + 2 * M15, 0),
                         _book_row(TS + 3 * M15, 0, bid1=79.0, ask1=81.0),
                         _book_row(TS + 4 * M15, 0, bid1=79.0, ask1=81.0)])
    book["mid"] = (book["bid_px_1"] + book["ask_px_1"]) / 2.0
    mkt = {"book": book,
           "products": pd.DataFrame([{"product_id": 0,
                                      "delivery_start": DELIV}])}
    blot = pd.DataFrame([
        {"exec_ts": TS, "product_delivery": DELIV, "side": "BUY",
         "qty_mw": 10.0, "price": 100.0},
        {"exec_ts": TS + 2 * M15, "product_delivery": DELIV, "side": "SELL",
         "qty_mw": 10.0, "price": 100.0},
    ])
    e = {"submission": "t",
         "backtest_window": {"start": "2025-01-01", "end": "2025-01-01"}}
    tape = _tape([_print(5, 99.0, 10.0, "SELL")])      # only the entry fills

    half, _ = style_pnl(e, blot, mkt, tape, "half_passive")
    full, _ = style_pnl(e, blot, mkt, tape, "passive")
    # half-passive sells into the 99 bid at the intended exit; fully passive
    # carries to gate and sells into the 79 bid
    assert half.sum() == pytest.approx(10 * 99.0 - 10 * 99.0)
    assert half.sum() > full.sum()
    assert full.sum() == pytest.approx(10 * 79.0 - 10 * 99.0)


def test_style_comparison_covers_every_style(market, entry, blotter):
    tape = _tape([_print(5, 99.0, 10.0, "SELL")])
    cmp = style_comparison(entry, blotter, market, tape)
    assert list(cmp.index) == list(STYLES)
    assert cmp.loc["aggressive", "vs_aggressive_eur"] == pytest.approx(0.0)


def test_unknown_style_is_rejected(market, entry, blotter):
    tape = _tape([_print(5, 99.0, 1.0, "SELL")])
    with pytest.raises(ValueError, match="style must be one of"):
        style_pnl(entry, blotter, market, tape, "wishful")


# ---- VWAP participation and time to fill ----------------------------------

def test_vwap_takes_a_share_of_each_print():
    """20% of three 10 MW prints is 2 MW each: 6 MW needs all three."""
    tape = _tape([_print(1, 100.0, 10.0, "BUY"), _print(2, 102.0, 10.0, "BUY"),
                  _print(3, 104.0, 10.0, "BUY")])
    vwap, mins, filled = tape.vwap_fill(0, TS.to_datetime64(),
                                        (TS + 4 * M15).to_datetime64(),
                                        qty=6.0, participation=0.2)
    assert filled == pytest.approx(6.0)
    assert vwap == pytest.approx((2 * 100 + 2 * 102 + 2 * 104) / 6)
    assert mins == pytest.approx(3.0)


def test_vwap_trims_the_last_print_to_land_on_the_clip():
    """20% of two 10 MW prints is 2 + 2; a 3 MW clip only needs 1 of the 2nd."""
    tape = _tape([_print(1, 100.0, 10.0, "BUY"), _print(2, 110.0, 10.0, "BUY")])
    vwap, mins, filled = tape.vwap_fill(0, TS.to_datetime64(),
                                        (TS + 4 * M15).to_datetime64(),
                                        qty=3.0, participation=0.2)
    assert filled == pytest.approx(3.0)
    assert vwap == pytest.approx((2 * 100 + 1 * 110) / 3)
    assert mins == pytest.approx(2.0)


def test_vwap_reports_a_clip_the_tape_cannot_absorb():
    tape = _tape([_print(1, 100.0, 10.0, "BUY")])
    vwap, mins, filled = tape.vwap_fill(0, TS.to_datetime64(),
                                        (TS + 4 * M15).to_datetime64(),
                                        qty=50.0, participation=0.2)
    assert filled == pytest.approx(2.0)          # 20% of the only print
    assert filled < 50.0
    assert vwap == pytest.approx(100.0)


def test_higher_participation_fills_faster():
    tape = _tape([_print(k, 100.0, 10.0, "BUY") for k in range(1, 11)])
    _, slow, _ = tape.vwap_fill(0, TS.to_datetime64(),
                                (TS + 4 * M15).to_datetime64(), 10.0, 0.2)
    _, fast, _ = tape.vwap_fill(0, TS.to_datetime64(),
                                (TS + 4 * M15).to_datetime64(), 10.0, 1.0)
    assert fast < slow


def test_vwap_summary_counts_unfilled_clips(market, entry, blotter):
    tape = _tape([_print(1, 100.0, 1.0, "BUY")])     # far too thin
    s = vwap_summary(entry, blotter, market, tape, participation=0.2)
    assert s["clips"] == 2
    assert s["unfilled_clips"] == 2
    assert s["unfilled_frac"] == pytest.approx(1.0)


def test_vwap_execution_reports_the_window_it_had(market, entry, blotter):
    tape = _tape([_print(1, 100.0, 10.0, "BUY")])
    b = vwap_execution(entry, blotter, market, tape)
    # last snapshot is TS+60min; the first trade is at TS
    assert b.loc[0, "minutes_available"] == pytest.approx(60.0)


def test_vwap_pnl_skips_products_with_an_unfilled_leg(market, entry):
    """A product missing a leg has no round trip: pricing only its entry
    would give a naked one-sided number, so it must be excluded."""
    other = DELIV + pd.Timedelta(hours=1)
    blot = pd.DataFrame([
        # product 0: both legs can fill
        {"exec_ts": TS, "product_delivery": DELIV, "side": "BUY",
         "qty_mw": 1.0, "price": 100.0},
        {"exec_ts": TS + M15, "product_delivery": DELIV, "side": "SELL",
         "qty_mw": 1.0, "price": 100.0},
        # product 1: nothing on the tape, so neither leg fills
        {"exec_ts": TS, "product_delivery": other, "side": "BUY",
         "qty_mw": 1.0, "price": 100.0},
        {"exec_ts": TS + M15, "product_delivery": other, "side": "SELL",
         "qty_mw": 1.0, "price": 100.0},
    ])
    book = pd.DataFrame([_book_row(TS + k * M15, p)
                         for p in (0, 1) for k in range(5)])
    book["mid"] = (book["bid_px_1"] + book["ask_px_1"]) / 2.0
    mkt = {"book": book, "products": pd.DataFrame([
        {"product_id": 0, "delivery_start": DELIV},
        {"product_id": 1, "delivery_start": other}])}
    tape = _tape([_print(1, 100.0, 40.0, "BUY"), _print(16, 100.0, 40.0, "BUY")])

    s = vwap_summary(entry, blot, mkt, tape, participation=0.5)
    assert s["unfilled_clips"] == 2                 # both legs of product 1
    assert s["pnl_products_covered"] == 1           # only product 0 priced
    assert s["pnl_coverage_frac"] == pytest.approx(0.5)


def test_a_leg_on_the_last_snapshot_has_no_window_to_fill_in(market, entry):
    """blackbox and spikecatcher exit exactly at the last pre-gate snapshot,
    which leaves zero forward time for a VWAP order."""
    blot = pd.DataFrame([
        {"exec_ts": TS, "product_delivery": DELIV, "side": "BUY",
         "qty_mw": 1.0, "price": 100.0},
        {"exec_ts": TS + 4 * M15, "product_delivery": DELIV, "side": "SELL",
         "qty_mw": 1.0, "price": 100.0},        # == last snapshot in fixture
    ])
    tape = _tape([_print(1, 100.0, 40.0, "BUY")])
    b = vwap_execution(entry, blot, market, tape)
    assert b["minutes_available"].min() == pytest.approx(0.0)
    assert b.loc[b["minutes_available"] == 0, "vwap_fill_frac"].iloc[0] == 0.0


# ---- market absorption -----------------------------------------------------

def test_absorption_measures_when_the_tape_has_traded_our_size(market, entry,
                                                               blotter):
    """4 MW at t+1, 4 at t+2, 4 at t+3: a 10 MW clip is absorbed at t+3."""
    tape = _tape([_print(1, 100.0, 4.0, "BUY"), _print(2, 100.0, 4.0, "SELL"),
                  _print(3, 100.0, 4.0, "BUY")])
    b = absorption(entry, blotter, market, tape)
    assert b.loc[0, "absorb_min"] == pytest.approx(3.0)


def test_absorption_counts_both_sides_of_the_tape(market, entry, blotter):
    """Absorption is the market trading our size at all, so aggressor side
    does not filter it - unlike a passive fill."""
    tape = _tape([_print(1, 100.0, 10.0, "SELL")])
    b = absorption(entry, blotter, market, tape)
    assert b.loc[0, "absorb_min"] == pytest.approx(1.0)


def test_absorption_is_nan_when_the_tape_never_gets_there(market, entry,
                                                          blotter):
    tape = _tape([_print(1, 100.0, 2.0, "BUY")])
    b = absorption(entry, blotter, market, tape)
    assert np.isnan(b.loc[0, "absorb_min"])


def test_absorbed_in_hold_compares_against_the_holding_time(market, entry):
    """Hold is 30 min here; absorption at 5 min is inside it, 40 is not."""
    blot = pd.DataFrame([
        {"exec_ts": TS, "product_delivery": DELIV, "side": "BUY",
         "qty_mw": 10.0, "price": 100.0},
        {"exec_ts": TS + 2 * M15, "product_delivery": DELIV, "side": "SELL",
         "qty_mw": 10.0, "price": 100.0},
    ])
    tape = _tape([_print(5, 100.0, 10.0, "BUY"), _print(70, 100.0, 10.0, "BUY")])
    b = absorption(entry, blot, market, tape)
    assert b.loc[0, "hold_min"] == pytest.approx(30.0)
    assert bool(b.loc[0, "absorbed_in_hold"]) is True     # 5 min
    assert bool(b.loc[1, "absorbed_in_hold"]) is False    # 40 min after t+30


def test_absorption_summary_relates_our_clip_to_a_typical_print(market, entry,
                                                                blotter):
    tape = _tape([_print(1, 100.0, 2.0, "BUY"), _print(2, 100.0, 4.0, "BUY"),
                  _print(3, 100.0, 6.0, "BUY")])
    s = absorption_summary(entry, blotter, market, tape)
    assert s["clip_mw"] == pytest.approx(10.0)
    assert s["median_print_mw"] == pytest.approx(4.0)
    assert s["clip_vs_median_print"] == pytest.approx(2.5)
