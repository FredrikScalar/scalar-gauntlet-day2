"""Blotter repricer — MID ONLY.

Loads the market tables, joins a trade blotter to the order book, and rebuilds
per-day P&L with every trade valued at the book mid. Each submission's claimed
numbers should reproduce here to within rounding: that is the consistency
check that proves the records hold together. It is NOT validation — a leaked,
overfit, or friction-doomed strategy reproduces perfectly at mid.

What is deliberately not here: repricing at prices you would actually get —
crossing the spread (buy at ask, sell at bid) and walking the book beyond
level 1. The book table carries three price/size levels per side; that
plumbing is yours to build, and most of the Friction section stands on it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_market(data_dir: str | Path = "../data") -> dict[str, pd.DataFrame]:
    """Load all market tables. Adds a derived `mid` column to the book —
    mid is (bid_px_1 + ask_px_1) / 2, a convenience, not a quoted price."""
    d = Path(data_dir)
    book = pd.read_csv(d / "book_snapshots.csv", parse_dates=["ts"])
    book["mid"] = (book["bid_px_1"] + book["ask_px_1"]) / 2.0
    return {
        "products": pd.read_csv(d / "products.csv",
                                parse_dates=["delivery_start", "delivery_end"]),
        "da_auction": pd.read_csv(d / "da_auction.csv"),
        "forecasts": pd.read_csv(d / "forecasts.csv",
                                 parse_dates=["issue_ts", "target_ts"]),
        "actuals": pd.read_csv(d / "actuals.csv", parse_dates=["target_ts"]),
        "outages": pd.read_csv(d / "outages.csv", parse_dates=["publish_ts"]),
        "book": book,
    }


def load_blotter(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["exec_ts", "product_delivery"])


def reprice_mid(blotter: pd.DataFrame, market: dict) -> pd.DataFrame:
    """Attach `product_id` and the book mid (`mid_price`) to every trade.

    Every position in every blotter is opened and closed through trades, so a
    product's P&L is simply sells minus buys — no marking of open positions
    is needed at any price column you add.
    """
    prods = market["products"][["product_id", "delivery_start"]]
    b = blotter.merge(prods, left_on="product_delivery",
                      right_on="delivery_start", how="left")
    if b["product_id"].isna().any():
        raise ValueError("blotter contains deliveries not in the products table")
    b = b.merge(market["book"][["ts", "product_id", "mid"]],
                left_on=["exec_ts", "product_id"],
                right_on=["ts", "product_id"], how="left")
    if b["mid"].isna().any():
        raise ValueError("some trades have no book snapshot at exec_ts")
    return (b.rename(columns={"mid": "mid_price"})
             .drop(columns=["delivery_start", "ts"]))


def product_pnl(blotter: pd.DataFrame, price_col: str = "price") -> pd.Series:
    """P&L per product (indexed by delivery start), trades valued at price_col."""
    signed = np.where(blotter["side"].eq("SELL"), 1.0, -1.0) \
        * blotter["qty_mw"].values * blotter[price_col].values
    return pd.Series(signed, index=pd.DatetimeIndex(blotter["product_delivery"])) \
        .groupby(level=0).sum()


def daily_pnl(blotter: pd.DataFrame, price_col: str = "price",
              window: tuple[str, str] | None = None) -> pd.Series:
    """Daily P&L by delivery day. Pass the registry's declared backtest window
    to include flat days — Sharpe over active days only flatters everyone."""
    pp = product_pnl(blotter, price_col)
    daily = pp.groupby(pp.index.normalize()).sum()
    if window is not None:
        idx = pd.date_range(window[0], window[1], freq="D", tz="UTC")
        daily = daily.reindex(idx, fill_value=0.0)
    return daily


def sharpe(daily: pd.Series) -> float:
    s = daily.std()
    return float(daily.mean() / s * np.sqrt(365.0)) if s > 0 else 0.0
