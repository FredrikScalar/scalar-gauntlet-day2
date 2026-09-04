"""Core: load market, reprice a blotter along the execution ladder, compute metrics."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, yaml

D = Path(__file__).resolve().parent.parent
FEE = 0.12          # EUR/MWh, each side
SUBS = ["windfall","windfall2","sunspot","spikecatcher","pingpong","bounceback","blackbox"]

BIDP = ["bid_px_1","bid_px_2","bid_px_3"]; BIDS = ["bid_sz_1","bid_sz_2","bid_sz_3"]
ASKP = ["ask_px_1","ask_px_2","ask_px_3"]; ASKS = ["ask_sz_1","ask_sz_2","ask_sz_3"]

def load_market():
    book = pd.read_csv(D/"data/book_snapshots.csv", parse_dates=["ts"])
    book["mid"] = (book.bid_px_1 + book.ask_px_1)/2.0
    prods = pd.read_csv(D/"data/products.csv", parse_dates=["delivery_start","delivery_end"])
    return {
        "book": book,
        "products": prods,
        "da": pd.read_csv(D/"data/da_auction.csv"),
        "actuals": pd.read_csv(D/"data/actuals.csv", parse_dates=["target_ts"]),
        "outages": pd.read_csv(D/"data/outages.csv", parse_dates=["publish_ts"]),
    }

def load_registry():
    return {p.stem: yaml.safe_load(p.read_text()) for p in sorted((D/"registry").glob("*.yaml"))}

def load_blotter(key):
    return pd.read_csv(D/f"blotters/{key}-blotter.csv", parse_dates=["exec_ts","product_delivery"])


def reprice(blotter: pd.DataFrame, market: dict) -> pd.DataFrame:
    """Attach every rung of the execution ladder to each trade.

    px_mid      : (bid1+ask1)/2 -- the price nobody quoted
    px_cross    : buy at ask_1, sell at bid_1
    px_walk     : VWAP walking levels 1..3 for the *cumulative* size demanded at
                  that (snapshot, product, side) -- so two 10MW orders at the same
                  instant eat depth sequentially, as they would in the market.
    fee_signed  : +FEE for buys, -FEE for sells (always a cost)
    """
    b = blotter.merge(market["products"][["product_id","delivery_start"]],
                      left_on="product_delivery", right_on="delivery_start", how="left")
    assert b.product_id.notna().all(), "delivery not in products table"
    cols = ["ts","product_id","mid"] + BIDP + BIDS + ASKP + ASKS
    b = b.merge(market["book"][cols], left_on=["exec_ts","product_id"],
                right_on=["ts","product_id"], how="left")
    assert b.mid.notna().all(), "trade with no book snapshot at exec_ts"
    b = b.drop(columns=["delivery_start","ts"])

    is_buy = b.side.eq("BUY").values
    b["px_mid"]   = b.mid.values
    b["px_cross"] = np.where(is_buy, b.ask_px_1.values, b.bid_px_1.values)

    # walk the book on cumulative demand within (exec_ts, product_id, side)
    b = b.sort_values(["exec_ts","product_id","side"]).reset_index(drop=True)
    grp = b.groupby(["exec_ts","product_id","side"], sort=False)
    start = grp.qty_mw.cumsum().values - b.qty_mw.values   # size already consumed
    need  = b.qty_mw.values
    is_buy = b.side.eq("BUY").values
    px = np.where(is_buy[:,None], b[ASKP].values, b[BIDP].values)
    sz = np.where(is_buy[:,None], b[ASKS].values, b[BIDS].values)
    cum = np.cumsum(sz, axis=1)
    lo = start[:,None]; hi = (start+need)[:,None]
    prev = np.concatenate([np.zeros((len(b),1)), cum[:,:-1]], axis=1)
    filled = np.clip(np.minimum(cum, hi) - np.maximum(prev, lo), 0, None)
    got = filled.sum(axis=1)
    notional = (filled*px).sum(axis=1)
    # depth exhausted: remainder fills at worst visible level, penalised one more tick
    tick = np.abs(px[:,2]-px[:,1])
    short = need - got
    worst = px[:,2] + np.where(is_buy, tick, -tick)
    b["px_walk"] = (notional + short*worst) / need
    b["depth_short_mw"] = short
    b["fee_signed"] = np.where(is_buy, FEE, -FEE)
    b["half_spread"] = (b.ask_px_1 - b.bid_px_1).values/2.0
    return b

LADDER = {
    "mid":            ("px_mid",   0.0),
    "mid_fee":        ("px_mid",   1.0),
    "cross":          ("px_cross", 0.0),
    "cross_fee":      ("px_cross", 1.0),
    "walk_fee":       ("px_walk",  1.0),
}

def eff_price(b, rung):
    col, feem = LADDER[rung]
    return b[col].values + feem*b.fee_signed.values

def daily_pnl(b, rung, window=None):
    px = eff_price(b, rung)
    signed = np.where(b.side.eq("SELL").values, 1.0, -1.0) * b.qty_mw.values * px
    s = pd.Series(signed, index=pd.DatetimeIndex(b.product_delivery)).groupby(level=0).sum()
    d = s.groupby(s.index.normalize()).sum()
    if window is not None:
        idx = pd.date_range(window[0], window[1], freq="D", tz="UTC")
        d = d.reindex(idx, fill_value=0.0)
    return d

def sharpe(d, ann=365.0):
    s = d.std(ddof=1)
    return float(d.mean()/s*np.sqrt(ann)) if s > 0 else 0.0

def sortino(d, ann=365.0):
    dn = np.minimum(d.values, 0.0)
    dd = np.sqrt((dn**2).mean())
    return float(d.mean()/dd*np.sqrt(ann)) if dd > 0 else np.nan

def max_dd(d):
    c = d.cumsum().values
    return float((c - np.maximum.accumulate(c)).min())
