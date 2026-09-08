"""Tests for the friction lab, on a hand-built book with known answers.

No real data is loaded: every fixture below is small enough to verify by
hand, so a failure points at the code rather than at the market.

    pytest test_friction.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from friction import (Scenario, active_days_frac, attach_book, compare,
                      crossing_check, depth_cost, edge_per_mwh, fill_price,
                      breakeven_fee, delay_sensitivity, fee_sensitivity,
                      best_multiplier, hit_rate_active_days, holding_time,
                      resolution_floor, size_stress, _sweep_partial,
                      round_trips,
                      max_drawdown, metrics, performance, sortino,
                      half_spread, _sweep_vwap)

TS = pd.Timestamp("2025-01-01 10:00", tz="UTC")
DELIV = pd.Timestamp("2025-01-01 12:00", tz="UTC")


def _book_row(ts, pid, bid1, ask1, step=1.0, size=10.0):
    """One snapshot: three levels a step apart, `size` MW on each."""
    row = {"ts": ts, "product_id": pid}
    for lvl in range(1, 4):
        row[f"bid_px_{lvl}"] = bid1 - (lvl - 1) * step
        row[f"ask_px_{lvl}"] = ask1 + (lvl - 1) * step
        row[f"bid_sz_{lvl}"] = size
        row[f"ask_sz_{lvl}"] = size
    return row


@pytest.fixture
def market():
    """One product, three snapshots 15 min apart, bid 99 / ask 101 (mid 100)."""
    book = pd.DataFrame([
        _book_row(TS, 0, 99.0, 101.0),
        _book_row(TS + pd.Timedelta(minutes=15), 0, 89.0, 91.0),   # mid 90
        _book_row(TS + pd.Timedelta(minutes=30), 0, 79.0, 81.0),   # mid 80
    ])
    book["mid"] = (book["bid_px_1"] + book["ask_px_1"]) / 2.0
    products = pd.DataFrame([{"product_id": 0, "delivery_start": DELIV}])
    return {"book": book, "products": products}


@pytest.fixture
def blotter():
    """A flat round trip at mid: buy 10 MW at 100, sell 10 MW at 100."""
    return pd.DataFrame([
        {"exec_ts": TS, "product_delivery": DELIV, "side": "BUY",
         "qty_mw": 10.0, "price": 100.0},
        {"exec_ts": TS, "product_delivery": DELIV, "side": "SELL",
         "qty_mw": 10.0, "price": 100.0},
    ])


# ---- the book sweeper -------------------------------------------------------

def test_sweep_fills_within_level_one():
    px = np.array([[101.0, 102.0, 103.0]])
    sz = np.array([[10.0, 10.0, 10.0]])
    assert _sweep_vwap(px, sz, np.array([10.0]))[0] == pytest.approx(101.0)


def test_sweep_spans_two_levels():
    # 15 MW = 10 at 101 + 5 at 102 -> (10*101 + 5*102) / 15
    px = np.array([[101.0, 102.0, 103.0]])
    sz = np.array([[10.0, 10.0, 10.0]])
    assert _sweep_vwap(px, sz, np.array([15.0]))[0] == pytest.approx(101.3333, abs=1e-4)


def test_sweep_returns_nan_when_book_too_thin():
    px = np.array([[101.0, 102.0, 103.0]])
    sz = np.array([[10.0, 10.0, 10.0]])
    assert np.isnan(_sweep_vwap(px, sz, np.array([40.0]))[0])


def test_sweep_treats_missing_depth_as_zero_size():
    px = np.array([[101.0, np.nan, np.nan]])
    sz = np.array([[10.0, np.nan, np.nan]])
    assert _sweep_vwap(px, sz, np.array([10.0]))[0] == pytest.approx(101.0)
    assert np.isnan(_sweep_vwap(px, sz, np.array([11.0]))[0])


# ---- fill prices per scenario ---------------------------------------------

def test_mid_scenario_reproduces_the_blotter_price(market, blotter):
    b = attach_book(blotter, market)
    assert fill_price(b, Scenario("mid")).tolist() == [100.0, 100.0]


def test_crossing_lifts_the_ask_and_hits_the_bid(market, blotter):
    b = attach_book(blotter, market)
    # buy pays 101, sell receives 99
    assert fill_price(b, Scenario("x", cross=True)).tolist() == [101.0, 99.0]


def test_sweeping_matches_crossing_when_level_one_is_deep_enough(market, blotter):
    b = attach_book(blotter, market)
    cross = fill_price(b, Scenario("x", cross=True))
    swept = fill_price(b, Scenario("y", cross=True, sweep=True))
    assert cross.tolist() == swept.tolist()


def test_sweeping_beyond_level_one_is_worse_than_crossing(market, blotter):
    big = blotter.assign(qty_mw=25.0)          # 10 + 10 + 5 across three levels
    b = attach_book(big, market)
    buy_sweep = fill_price(b, Scenario("y", cross=True, sweep=True)).iloc[0]
    assert buy_sweep > fill_price(b, Scenario("x", cross=True)).iloc[0]
    assert buy_sweep == pytest.approx((10 * 101 + 10 * 102 + 5 * 103) / 25)


def test_fees_work_against_both_sides(market, blotter):
    b = attach_book(blotter, market)
    plain = fill_price(b, Scenario("x", cross=True))
    fee = fill_price(b, Scenario("x", cross=True, fee_eur_mwh=0.5))
    assert fee.iloc[0] == pytest.approx(plain.iloc[0] + 0.5)   # buy pays more
    assert fee.iloc[1] == pytest.approx(plain.iloc[1] - 0.5)   # sell gets less


def test_slippage_and_fee_are_additive(market, blotter):
    b = attach_book(blotter, market)
    both = fill_price(b, Scenario("x", cross=True, fee_eur_mwh=0.5,
                                  slippage_eur_mwh=0.25))
    assert both.iloc[0] == pytest.approx(101.75)


# ---- delayed execution -----------------------------------------------------

def test_delay_uses_the_later_snapshot(market, blotter):
    b = attach_book(blotter, market, delay_snapshots=1)
    assert fill_price(b, Scenario("mid")).tolist() == [90.0, 90.0]
    assert not b["delayed_gated"].any()


def test_delay_past_gate_falls_back_to_the_last_snapshot(market, blotter):
    b = attach_book(blotter, market, delay_snapshots=10)
    assert fill_price(b, Scenario("mid")).tolist() == [80.0, 80.0]   # last book
    assert b["delayed_gated"].all()


# ---- P&L arithmetic --------------------------------------------------------

def test_flat_round_trip_is_zero_at_mid(market, blotter):
    tbl = compare(blotter, market, [Scenario("mid")])
    assert tbl.loc["mid", "pnl_eur"] == pytest.approx(0.0)


def test_crossing_costs_exactly_the_spread(market, blotter):
    """Buy at 101, sell at 99, 10 MW: the round trip loses spread * qty."""
    tbl = compare(blotter, market, [Scenario("mid"), Scenario("cross", cross=True)])
    assert tbl.loc["cross", "pnl_eur"] == pytest.approx(-20.0)      # (99-101)*10
    assert tbl.loc["cross", "cost_vs_base_eur"] == pytest.approx(-20.0)


def test_crossing_never_beats_mid(market, blotter):
    tbl = compare(blotter, market, [Scenario("mid"), Scenario("cross", cross=True),
                                    Scenario("sweep", cross=True, sweep=True)])
    assert tbl.loc["cross", "pnl_eur"] <= tbl.loc["mid", "pnl_eur"]
    assert tbl.loc["sweep", "pnl_eur"] <= tbl.loc["cross", "pnl_eur"]


def test_retention_is_relative_to_the_first_scenario(market):
    """A profitable round trip: buy at mid 100, sell one snapshot later."""
    profitable = pd.DataFrame([
        {"exec_ts": TS + pd.Timedelta(minutes=15), "product_delivery": DELIV,
         "side": "BUY", "qty_mw": 10.0, "price": 90.0},
        {"exec_ts": TS, "product_delivery": DELIV, "side": "SELL",
         "qty_mw": 10.0, "price": 100.0},
    ])
    market_ = market_fixture()
    tbl = compare(profitable, market_, [Scenario("mid"),
                                        Scenario("cross", cross=True)])
    assert tbl.loc["mid", "pnl_eur"] == pytest.approx(100.0)   # (100 - 90) * 10
    assert tbl.loc["mid", "pnl_retained"] == pytest.approx(1.0)
    assert tbl.loc["cross", "pnl_retained"] < 1.0


def market_fixture():
    """The `market` fixture as a plain call, for tests that need two books."""
    book = pd.DataFrame([
        _book_row(TS, 0, 99.0, 101.0),
        _book_row(TS + pd.Timedelta(minutes=15), 0, 89.0, 91.0),
        _book_row(TS + pd.Timedelta(minutes=30), 0, 79.0, 81.0),
    ])
    book["mid"] = (book["bid_px_1"] + book["ask_px_1"]) / 2.0
    return {"book": book,
            "products": pd.DataFrame([{"product_id": 0, "delivery_start": DELIV}])}


# ---- descriptive metrics ---------------------------------------------------

def test_half_spread(market, blotter):
    assert half_spread(attach_book(blotter, market)).tolist() == [1.0, 1.0]


def test_edge_per_mwh_divides_by_total_volume(market):
    """P&L 100 EUR over 20 MWh traded (two 10 MW legs) = 5 EUR/MWh."""
    profitable = pd.DataFrame([
        {"exec_ts": TS + pd.Timedelta(minutes=15), "product_delivery": DELIV,
         "side": "BUY", "qty_mw": 10.0, "price": 90.0},
        {"exec_ts": TS, "product_delivery": DELIV, "side": "SELL",
         "qty_mw": 10.0, "price": 100.0},
    ])
    b = attach_book(profitable, market_fixture())
    assert edge_per_mwh(b) == pytest.approx(5.0)


def test_unknown_delivery_is_rejected(market, blotter):
    bad = blotter.assign(product_delivery=pd.Timestamp("2030-01-01", tz="UTC"))
    with pytest.raises(ValueError):
        attach_book(bad, market)


# ---- headline metrics ------------------------------------------------------

def _daily(values):
    idx = pd.date_range("2025-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_max_drawdown_is_peak_to_trough():
    # cum: 10, 4, 14 -> deepest fall is 10 -> 4
    assert max_drawdown(_daily([10, -6, 10])) == pytest.approx(-6.0)


def test_max_drawdown_is_zero_when_never_falling():
    assert max_drawdown(_daily([1, 2, 3])) == pytest.approx(0.0)


def test_active_days_and_hit_rate_ignore_flat_days():
    d = _daily([5, 0, 0, -2, 3])          # 3 active days, 2 of them winners
    assert active_days_frac(d) == pytest.approx(3 / 5)
    assert hit_rate_active_days(d) == pytest.approx(2 / 3)


def test_sortino_uses_std_of_losing_days():
    d = _daily([10, -2, 8, -4, 6])
    losses = pd.Series([-2.0, -4.0])
    expected = d.mean() / losses.std(ddof=1) * np.sqrt(365)
    assert sortino(d) == pytest.approx(expected)


def test_sortino_is_zero_without_two_losing_days():
    assert sortino(_daily([1, 2, 3])) == 0.0
    assert sortino(_daily([1, 2, -1])) == 0.0


def test_metrics_reports_edge_per_mwh_and_volume(market, blotter):
    b = attach_book(blotter, market)
    b["px_mid"] = fill_price(b, Scenario("mid"))
    m = metrics(b, "px_mid")
    assert m["trades"] == 2
    assert m["mwh_traded"] == pytest.approx(20.0)
    assert m["total_pnl_eur"] == pytest.approx(0.0)


# ---- the crossing check ----------------------------------------------------

@pytest.fixture
def entry():
    return {"submission": "unit-test", "class": "test",
            "backtest_window": {"start": "2025-01-01", "end": "2025-01-01"},
            "claimed": {"total_pnl_eur": 0.0, "sharpe_ann": 0.0}}


def test_crossing_check_lays_claim_mid_and_cross_side_by_side(entry, market, blotter):
    chk = crossing_check(entry, blotter, market)
    assert list(chk.columns) == ["mid", "cross", "cross_minus_mid"]
    assert chk.loc["total_pnl_eur", "mid"] == pytest.approx(0.0)
    assert chk.loc["total_pnl_eur", "cross"] == pytest.approx(-20.0)
    assert chk.loc["total_pnl_eur", "cross_minus_mid"] == pytest.approx(-20.0)


def test_crossing_check_reports_the_half_spread(entry, market, blotter):
    chk = crossing_check(entry, blotter, market)
    assert chk.loc["half_spread_eur_mwh", "mid"] == pytest.approx(1.0)


# ---- final performance -----------------------------------------------------

def test_performance_defaults_to_cross(entry, market, blotter):
    p = performance(entry, blotter, market)
    assert p["fill"] == "cross"
    assert p["total_pnl_eur"] == pytest.approx(-20.0)


def test_performance_at_mid_matches_the_blotter(entry, market, blotter):
    p = performance(entry, blotter, market, fill="mid")
    assert p["total_pnl_eur"] == pytest.approx(0.0)
    assert p["half_spread_eur_mwh"] == pytest.approx(1.0)


def test_performance_rejects_an_unknown_fill(entry, market, blotter):
    with pytest.raises(ValueError, match="fill must be one of"):
        performance(entry, blotter, market, fill="nonsense")


def test_performance_agrees_with_the_crossing_check(entry, market, blotter):
    chk = crossing_check(entry, blotter, market)
    for fill in ("mid", "cross"):
        p = performance(entry, blotter, market, fill=fill)
        assert p["total_pnl_eur"] == pytest.approx(chk.loc["total_pnl_eur", fill])
        assert p["sharpe_ann"] == pytest.approx(chk.loc["sharpe_ann", fill])


def test_crossing_check_no_longer_carries_claimed(entry, market, blotter):
    assert list(crossing_check(entry, blotter, market).columns) == \
        ["mid", "cross", "cross_minus_mid"]


# ---- fee sensitivity -------------------------------------------------------

def test_pnl_is_exactly_linear_in_the_fee(entry, market, blotter):
    """Every MWh pays the fee whichever way it trades, so P&L falls by
    fee * total MWh - no curvature, no side dependence."""
    sens = fee_sensitivity(entry, blotter, market, step=0.05, max_fee=0.25)
    mwh = 20.0                                   # two 10 MW legs
    base = sens["total_pnl_eur"].iloc[0]
    for fee in sens.index:
        assert sens.loc[fee, "total_pnl_eur"] == pytest.approx(base - fee * mwh)


def test_fee_sweep_uses_five_cent_steps_by_default(entry, market, blotter):
    sens = fee_sensitivity(entry, blotter, market, max_fee=0.25)
    assert list(np.round(sens.index, 10)) == [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def test_breakeven_fee_equals_edge_per_mwh(market):
    """A round trip making 100 EUR on 20 MWh breaks even at 5.00 EUR/MWh."""
    profitable = pd.DataFrame([
        {"exec_ts": TS + pd.Timedelta(minutes=15), "product_delivery": DELIV,
         "side": "BUY", "qty_mw": 10.0, "price": 90.0},
        {"exec_ts": TS, "product_delivery": DELIV, "side": "SELL",
         "qty_mw": 10.0, "price": 100.0},
    ])
    e = {"submission": "t", "backtest_window": {"start": "2025-01-01",
                                                "end": "2025-01-01"}}
    m = market_fixture()
    assert breakeven_fee(e, profitable, m, fill="mid") == pytest.approx(5.0)
    sens = fee_sensitivity(e, profitable, m, step=1.0, max_fee=6.0, fill="mid")
    assert sens.loc[5.0, "total_pnl_eur"] == pytest.approx(0.0)
    assert sens.loc[6.0, "total_pnl_eur"] < 0


def test_breakeven_is_negative_when_already_losing(entry, market, blotter):
    """The crossed round trip loses 20 EUR on 20 MWh before any fee."""
    assert breakeven_fee(entry, blotter, market) == pytest.approx(-1.0)


def test_fee_grid_reaches_past_breakeven_by_default(market):
    profitable = pd.DataFrame([
        {"exec_ts": TS + pd.Timedelta(minutes=15), "product_delivery": DELIV,
         "side": "BUY", "qty_mw": 10.0, "price": 90.0},
        {"exec_ts": TS, "product_delivery": DELIV, "side": "SELL",
         "qty_mw": 10.0, "price": 100.0},
    ])
    e = {"submission": "t", "backtest_window": {"start": "2025-01-01",
                                                "end": "2025-01-01"}}
    sens = fee_sensitivity(e, profitable, market_fixture(), fill="mid")
    assert sens.index.max() > 5.0                 # breakeven is 5.00
    assert sens["total_pnl_eur"].iloc[-1] < 0


def test_fee_sweep_matches_charging_the_fee_directly(entry, market, blotter):
    """The linear shortcut must agree with actually repricing at fee=0.05."""
    b = attach_book(blotter, market)
    direct = fill_price(b, Scenario("c", cross=True, fee_eur_mwh=0.05))
    b["px"] = direct
    from repricer import daily_pnl
    expected = float(daily_pnl(b, "px").sum())
    sens = fee_sensitivity(entry, blotter, market, max_fee=0.05)
    assert sens.loc[0.05, "total_pnl_eur"] == pytest.approx(expected)


# ---- holding time and execution delay --------------------------------------

def _trade(ts, side, qty=10.0, deliv=None, price=100.0):
    return {"exec_ts": ts, "product_delivery": deliv or DELIV, "side": side,
            "qty_mw": qty, "price": price}


def test_round_trip_pairs_an_open_with_its_close():
    b = pd.DataFrame([_trade(TS, "BUY"),
                      _trade(TS + pd.Timedelta(minutes=30), "SELL")])
    rt = round_trips(b)
    assert len(rt) == 1
    assert rt.loc[0, "hold_s"] == pytest.approx(1800.0)
    assert rt.loc[0, "qty_mw"] == pytest.approx(10.0)
    assert rt.attrs["unmatched_mw"] == pytest.approx(0.0)


def test_round_trips_match_first_in_first_out():
    """Two opens then two closes: the earlier open pairs with the earlier close."""
    b = pd.DataFrame([
        _trade(TS, "BUY"),
        _trade(TS + pd.Timedelta(minutes=15), "BUY"),
        _trade(TS + pd.Timedelta(minutes=30), "SELL"),
        _trade(TS + pd.Timedelta(minutes=60), "SELL"),
    ])
    rt = round_trips(b).sort_values("open_ts").reset_index(drop=True)
    assert rt["hold_s"].tolist() == [1800.0, 2700.0]      # 30 min, 45 min


def test_round_trips_split_a_larger_close_across_lots():
    b = pd.DataFrame([
        _trade(TS, "BUY", qty=10.0),
        _trade(TS + pd.Timedelta(minutes=15), "BUY", qty=10.0),
        _trade(TS + pd.Timedelta(minutes=30), "SELL", qty=20.0),
    ])
    rt = round_trips(b)
    assert len(rt) == 2
    assert rt["qty_mw"].tolist() == [10.0, 10.0]
    assert sorted(rt["hold_s"].tolist()) == [900.0, 1800.0]


def test_round_trips_report_an_unclosed_position():
    b = pd.DataFrame([_trade(TS, "BUY", qty=10.0),
                      _trade(TS + pd.Timedelta(minutes=30), "SELL", qty=4.0)])
    rt = round_trips(b)
    assert rt["qty_mw"].sum() == pytest.approx(4.0)
    assert rt.attrs["unmatched_mw"] == pytest.approx(6.0)


def test_round_trips_do_not_pair_across_products():
    other = DELIV + pd.Timedelta(hours=1)
    b = pd.DataFrame([_trade(TS, "BUY"),
                      _trade(TS + pd.Timedelta(minutes=30), "SELL", deliv=other)])
    rt = round_trips(b)
    assert len(rt) == 0
    assert rt.attrs["unmatched_mw"] == pytest.approx(20.0)


def test_holding_time_weights_by_volume():
    b = pd.DataFrame([
        _trade(TS, "BUY", qty=30.0),
        _trade(TS + pd.Timedelta(minutes=60), "SELL", qty=30.0),
        _trade(TS, "BUY", qty=10.0, deliv=DELIV + pd.Timedelta(hours=1)),
        _trade(TS + pd.Timedelta(minutes=20), "SELL", qty=10.0,
               deliv=DELIV + pd.Timedelta(hours=1)),
    ])
    h = holding_time(b)
    assert h["mean_h"] == pytest.approx((30 * 1.0 + 10 * (1 / 3)) / 40)
    assert h["round_trips"] == 2
    assert h["unmatched_mw"] == pytest.approx(0.0)


def test_delay_sensitivity_starts_undelayed(market, blotter, entry):
    sens = delay_sensitivity(entry, blotter, market, fractions=(0.5,))
    assert sens.loc[0.0, "snapshots"] == 0
    assert sens.loc[0.0, "pnl_retained"] == pytest.approx(1.0)


def test_delay_sensitivity_reports_the_snapshot_rounding(market, entry):
    """A 30-minute hold delayed 10% wants 3 minutes, which the 15-minute book
    cannot represent - the row must say so rather than silently apply 15."""
    b = pd.DataFrame([_trade(TS, "BUY"),
                      _trade(TS + pd.Timedelta(minutes=30), "SELL")])
    sens = delay_sensitivity(entry, b, market, fractions=(0.10, 0.40))
    assert sens.loc[0.10, "target_min"] == pytest.approx(3.0)
    assert sens.loc[0.10, "snapshots"] == 0            # rounds away
    assert sens.loc[0.10, "realised_fraction"] == pytest.approx(0.0)
    assert sens.loc[0.40, "target_min"] == pytest.approx(12.0)
    assert sens.loc[0.40, "snapshots"] == 1            # rounds up to 15 min
    assert sens.loc[0.40, "applied_min"] == pytest.approx(15.0)
    assert sens.loc[0.40, "realised_fraction"] == pytest.approx(0.5)


def _wide_market(n_products=3, n_snaps=6):
    """A book long enough for hour-scale holds, several products wide."""
    rows = []
    for pid in range(n_products):
        for k in range(n_snaps):
            rows.append(_book_row(TS + k * pd.Timedelta(minutes=15), pid,
                                  99.0, 101.0))
    book = pd.DataFrame(rows)
    book["mid"] = (book["bid_px_1"] + book["ask_px_1"]) / 2.0
    products = pd.DataFrame([
        {"product_id": pid, "delivery_start": DELIV + pid * pd.Timedelta(hours=1)}
        for pid in range(n_products)])
    return {"book": book, "products": products}


def test_delay_clock_defaults_to_the_median_holding_time(entry):
    """Holds of 15, 15 and 75 minutes: median 15, volume-weighted mean 35.
    One long hold must not stretch the yardstick for the typical trade."""
    m = _wide_market()
    hours = pd.Timedelta(hours=1)
    b = pd.DataFrame([
        _trade(TS, "BUY", deliv=DELIV),
        _trade(TS + pd.Timedelta(minutes=15), "SELL", deliv=DELIV),
        _trade(TS, "BUY", deliv=DELIV + hours),
        _trade(TS + pd.Timedelta(minutes=15), "SELL", deliv=DELIV + hours),
        _trade(TS, "BUY", deliv=DELIV + 2 * hours),
        _trade(TS + pd.Timedelta(minutes=75), "SELL", deliv=DELIV + 2 * hours),
    ])
    h = holding_time(b)
    assert h["median_h"] == pytest.approx(0.25)       # 15 minutes
    assert h["mean_h"] == pytest.approx(35 / 60)      # 15, 15, 75 -> 35 minutes

    med = delay_sensitivity(entry, b, m, fractions=(1.0,))
    assert med.loc[1.0, "target_min"] == pytest.approx(15.0)
    mean = delay_sensitivity(entry, b, m, fractions=(1.0,), stat="mean")
    assert mean.loc[1.0, "target_min"] == pytest.approx(35.0)


def test_resolution_floor_is_half_a_snapshot_over_the_hold():
    """Half a snapshot is 7.5 min, so a 30-minute holder cannot resolve any
    delay below 25% of its holding time."""
    b = pd.DataFrame([_trade(TS, "BUY"),
                      _trade(TS + pd.Timedelta(minutes=30), "SELL")])
    assert resolution_floor(b) == pytest.approx(7.5 / 30.0)


def test_below_resolution_is_flagged_not_silently_zero(market, entry):
    b = pd.DataFrame([_trade(TS, "BUY"),
                      _trade(TS + pd.Timedelta(minutes=30), "SELL")])
    sens = delay_sensitivity(entry, b, market, fractions=(0.01, 0.10, 0.50))
    assert sens.loc[0.01, "below_resolution"] is True    # 0.3 min -> nothing
    assert sens.loc[0.10, "below_resolution"] is True    # 3.0 min -> nothing
    assert sens.loc[0.50, "below_resolution"] is False   # 15 min -> 1 snapshot
    # the undelayed baseline is never "below resolution"
    assert sens.loc[0.0, "below_resolution"] is False


def test_default_fractions_are_log_spaced():
    from friction import HOLD_FRACTIONS
    assert HOLD_FRACTIONS[0] == 0.01
    ratios = [b / a for a, b in zip(HOLD_FRACTIONS, HOLD_FRACTIONS[1:])]
    assert all(1.5 <= r <= 3.0 for r in ratios)          # roughly geometric


def test_delay_exactly_at_the_resolution_floor_resolves(market, entry):
    """Round half UP: a 30-min holder asked for 25% wants exactly 7.5 min,
    which must become one snapshot rather than rounding away to nothing."""
    b = pd.DataFrame([_trade(TS, "BUY"),
                      _trade(TS + pd.Timedelta(minutes=30), "SELL")])
    floor = resolution_floor(b)
    assert floor == pytest.approx(0.25)
    sens = delay_sensitivity(entry, b, market, fractions=(floor,))
    assert sens.loc[floor, "target_min"] == pytest.approx(7.5)
    assert sens.loc[floor, "snapshots"] == 1
    assert sens.loc[floor, "below_resolution"] is False


def test_default_fractions_stop_at_half_the_holding_time():
    from friction import HOLD_FRACTIONS
    assert max(HOLD_FRACTIONS) == 0.50


# ---- sweep vs BBO cross ----------------------------------------------------

def test_sweep_equals_cross_when_every_clip_fits_at_the_touch(entry, market, blotter):
    """10 MW clip into 10 MW of level-1 depth: nothing to sweep."""
    d = depth_cost(entry, blotter, market)
    assert d.loc["total_pnl_eur", "sweep_minus_cross"] == pytest.approx(0.0)
    assert d.loc["clips_over_level_1", "sweep"] == 0.0
    assert d.loc["max_depth_used_x_level_1", "sweep"] == pytest.approx(1.0)


def test_sweep_costs_more_than_cross_when_the_clip_overflows(entry, market):
    """25 MW into 10 MW levels: the touch cannot fill it, so BBO cross is
    an impossible price and sweeping must be worse."""
    big = pd.DataFrame([_trade(TS, "BUY", qty=25.0),
                        _trade(TS, "SELL", qty=25.0)])
    d = depth_cost(entry, big, market)
    assert d.loc["clips_over_level_1", "sweep"] == 2.0
    assert d.loc["clips_over_level_1_frac", "sweep"] == pytest.approx(1.0)
    assert d.loc["max_depth_used_x_level_1", "sweep"] == pytest.approx(2.5)
    # buy VWAP 101.8 vs touch 101.0, sell 98.2 vs 99.0 -> 0.8 worse each side
    assert d.loc["total_pnl_eur", "cross"] == pytest.approx(-50.0)   # 2 * 25
    assert d.loc["total_pnl_eur", "sweep"] == pytest.approx(-90.0)   # 3.6 * 25
    assert d.loc["total_pnl_eur", "sweep_minus_cross"] == pytest.approx(-40.0)


def test_depth_cost_counts_clips_the_book_cannot_fill(entry, market):
    """40 MW against 30 MW of visible depth has no sweep price at all."""
    huge = pd.DataFrame([_trade(TS, "BUY", qty=40.0),
                         _trade(TS, "SELL", qty=40.0)])
    d = depth_cost(entry, huge, market)
    assert d.loc["unfillable_clips", "sweep"] == 2.0


def test_sweep_is_a_first_class_fill(entry, market, blotter):
    p = performance(entry, blotter, market, fill="sweep")
    assert p["fill"] == "sweep"
    assert p["total_pnl_eur"] == pytest.approx(-20.0)   # fits at the touch


def test_fills_offers_mid_cross_and_sweep():
    from friction import FILLS
    assert sorted(FILLS) == ["cross", "mid", "sweep"]


def test_delay_grid_always_has_an_undelayed_baseline(market, entry, blotter):
    """`delay_summary` reports this row as `pnl_immediate`. Unlike the delayed
    columns it is never blanked out by the book's 15-minute resolution, so
    there is always something to read the others against."""
    sens = delay_sensitivity(entry, blotter, market, fractions=(0.01, 0.5))
    assert 0.0 in sens.index
    assert sens.loc[0.0, "snapshots"] == 0
    assert sens.loc[0.0, "below_resolution"] is False
    assert not np.isnan(sens.loc[0.0, "total_pnl_eur"])
    # the 1% request is below resolution and gets blanked; the baseline never is
    assert sens.loc[0.01, "below_resolution"] is True


# ---- size stress -----------------------------------------------------------

def test_sweep_partial_fills_what_the_book_holds():
    """30 MW of visible depth against a 50 MW clip fills 30, at the 3-level
    VWAP - a partial fill, not a blank."""
    px = np.array([[101.0, 102.0, 103.0]])
    sz = np.array([[10.0, 10.0, 10.0]])
    vwap, filled = _sweep_partial(px, sz, np.array([50.0]))
    assert filled[0] == pytest.approx(30.0)
    assert vwap[0] == pytest.approx((101 + 102 + 103) / 3)


def test_sweep_partial_matches_full_sweep_when_depth_suffices():
    px = np.array([[101.0, 102.0, 103.0]])
    sz = np.array([[10.0, 10.0, 10.0]])
    vwap, filled = _sweep_partial(px, sz, np.array([15.0]))
    assert filled[0] == pytest.approx(15.0)
    assert vwap[0] == pytest.approx(_sweep_vwap(px, sz, np.array([15.0]))[0])


def test_size_stress_reports_fill_and_exhaustion(entry, market, blotter):
    """Fixture book holds 10 MW on each of 3 levels = 30 MW a side."""
    t = size_stress(entry, blotter, market, multipliers=(1, 3, 5))
    assert t.loc[1, "mean_fill_frac"] == pytest.approx(1.0)
    assert t.loc[1, "pct_depth_exhausted"] == pytest.approx(0.0)
    assert t.loc[1, "pct_past_touch"] == pytest.approx(0.0)
    assert t.loc[3, "mean_fill_frac"] == pytest.approx(1.0)   # 30 of 30 MW
    assert t.loc[3, "pct_past_touch"] == pytest.approx(1.0)   # but past level 1
    assert t.loc[5, "mean_fill_frac"] == pytest.approx(30 / 50)
    assert t.loc[5, "pct_depth_exhausted"] == pytest.approx(1.0)


def test_doubling_the_clip_more_than_doubles_the_cost(entry, market, blotter):
    """Cost per MWh rises with size, so P&L does NOT scale linearly.

    At 1x the 10 MW clip fills at the touch: buy 101, sell 99, losing 20.
    At 2x it sweeps two levels each side - buy (101+102)/2, sell (99+98)/2 -
    losing 3/MWh over 20 MWh, so 60 rather than 40.
    """
    t = size_stress(entry, blotter, market, multipliers=(1, 2))
    assert t.loc[1, "total_pnl_eur"] == pytest.approx(-20.0)
    assert t.loc[2, "total_pnl_eur"] == pytest.approx(-60.0)
    assert t.loc[2, "total_pnl_eur"] < 2 * t.loc[1, "total_pnl_eur"]
    # the per-MWh cost is what grew: 1.0 -> 1.5 x the half-spread
    assert (t.loc[2, "eur_per_mwh"] / t.loc[1, "eur_per_mwh"]
            == pytest.approx(1.5))


def test_best_multiplier_can_require_a_full_fill(entry, market, blotter):
    t = size_stress(entry, blotter, market, multipliers=(1, 3, 5))
    loose = best_multiplier(t, require_full_fill=False)
    strict = best_multiplier(t, require_full_fill=True)
    assert strict["mean_fill_frac"] >= 0.999
    assert strict["multiplier"] in (1.0, 3.0)
    assert loose["multiplier"] in (1.0, 3.0, 5.0)
