"""Baselines recomputed so the strategy leg is priced at the SAME rung as the
benchmark it is compared against, plus a matched-horizon passive long.

run_full.py calls L.baselines(t, ...) with the mid-priced trip frame, so the
"strategy" row it stores is mid-priced at every rung while always_long /
always_short / market_bh_long are rung-priced. That makes the cross_fee
comparison apples-to-oranges. This recomputes all of it consistently.
"""
import json, time, numpy as np, pandas as pd
import engine as E, trips as T, luck as L

t0 = time.time()
mk = E.load_market(); reg = E.load_registry(); bg = L.BookGrid(mk)
RUNGS = ["mid", "passive_in_fee", "cross_fee"]
QTY = 10.0
out = {"rungs": RUNGS, "subs": {}}

allr = np.arange(len(bg.pids))
LAST = bg.k - 1

for k in E.SUBS:
    e = reg[k]
    w = (str(e["backtest_window"]["start"]), str(e["backtest_window"]["end"]))
    b = E.reprice(E.load_blotter(k), mk)
    t = T.round_trips(b, "mid")
    days_i = L._naive(pd.date_range(w[0], w[1], freq="D"))
    day = L._naive(t.delivery)
    q = t.qty.values
    r_ = bg.rows_of(t.product_id.values)
    e_ = bg.slot_of(t.product_id.values, t.entry_ts.values)
    x_ = bg.slot_of(t.product_id.values, t.exit_ts.values)

    # matched-horizon passive: enter at the strategy's median entry snapshot and
    # hold for its median holding period, 10 MW long on EVERY product
    entry_slot = int(np.median(e_))
    hold_steps = int(round(np.median(x_ - e_)))
    ms_e = np.full(len(allr), entry_slot)
    ms_x = np.full(len(allr), min(entry_slot + hold_steps, LAST))

    rec = {"entry_slot": entry_slot, "hold_steps": hold_steps,
           "hold_min": float(t.hold_min.median()), "long_frac": float((t.dir > 0).mean()),
           "n_trips": int(len(t)), "vol_mwh": float(b.qty_mw.sum()), "rungs": {}}

    def ser(p, idx):
        return pd.Series(p, index=idx).groupby(level=0).sum().reindex(days_i, fill_value=0.0).values

    for rung in RUNGS:
        lp, sp = bg.leg_prices(r_, e_, x_, rung)
        tp = L.trip_pnl(t, bg, rung)
        d_strat = ser(tp, day)
        d_long = ser(q*lp, day)
        d_short = ser(q*sp, day)
        blp, _ = bg.leg_prices(allr, np.zeros(len(allr), int),
                               np.full(len(allr), LAST), rung)
        d_bh = ser(QTY*blp, bg.deliv_day)
        mlp, _ = bg.leg_prices(allr, ms_e, ms_x, rung)
        d_match = ser(QTY*mlp, bg.deliv_day)
        # per-MWh edge, and the market's per-MWh return over the SAME clock window
        edge = float(tp.sum()/(q.sum()))
        match_edge = float(mlp.mean())
        rec["rungs"][rung] = {
            "strategy":       {"pnl": float(d_strat.sum()),  "sharpe": L.sharpe(d_strat)},
            "always_long":    {"pnl": float(d_long.sum()),   "sharpe": L.sharpe(d_long)},
            "always_short":   {"pnl": float(d_short.sum()),  "sharpe": L.sharpe(d_short)},
            "market_bh_full": {"pnl": float(d_bh.sum()),     "sharpe": L.sharpe(d_bh)},
            "matched_passive":{"pnl": float(d_match.sum()),  "sharpe": L.sharpe(d_match)},
            "edge_eur_mwh": edge,
            "matched_edge_eur_mwh": match_edge,
            "edge_ratio": (edge/match_edge) if match_edge else None,
            "prods_per_day": float(len(t)/max((d_strat != 0).sum(), 1)),
        }
    out["subs"][k] = rec
    print(f"{k:13s} entry {entry_slot:2d} hold {hold_steps:2d} steps  {time.time()-t0:.0f}s", flush=True)

json.dump(out, open("base2.json", "w"), indent=1)
print("wrote base2.json")
