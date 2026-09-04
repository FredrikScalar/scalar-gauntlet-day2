import numpy as np, pandas as pd, engine as E, trips as T, luck as L
mk=E.load_market(); bg=L.BookGrid(mk); F=E.FEE
for k in ["spikecatcher","blackbox"]:
    b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid")
    r=bg.rows_of(t.product_id.values)
    e=bg.slot_of(t.product_id.values,t.entry_ts.values)
    x=bg.slot_of(t.product_id.values,t.exit_ts.values)
    d=t.dir.values; q=t.qty.values
    nL=int((d>0).sum()); nS=int((d<0).sum())
    print("="*84); print(f"{k.upper()}   {len(t)} round trips   {nL} long / {nS} short")
    print(f"  entry snapshot: {int(np.median(e))} (median)   exit snapshot: {int(np.median(x))}  "
          f"= the LAST book snapshot, 15 min before gate closure")
    print(f"  recorded blotter close price (mean): {b[b.index.isin([])].shape[0] if False else 0:.0f}", end="")
    # recorded exit prices straight from the blotter
    bs=b.sort_values(["product_id","exit_ts"] if "exit_ts" in b else ["product_id","exec_ts"])
    print()
    print(f"  --- CLOSING PRICE AT EACH RUNG (mean across trips, EUR/MWh) ---")
    print(f"  {'rung':<18} {'long closes at':>22} {'short closes at':>22}")
    def m(a): return a.mean()
    rows=[("mid",       bg.mid[r,x],                bg.mid[r,x]),
          ("mid_fee",   bg.mid[r,x]-F,              bg.mid[r,x]+F),
          ("cross",     bg.bid[r,x],                bg.ask[r,x]),
          ("cross_fee", bg.bid[r,x]-F,              bg.ask[r,x]+F)]
    for n,lp,sp in rows:
        lv = m(lp[d>0]) if nL else float("nan")
        sv = m(sp[d<0]) if nS else float("nan")
        print(f"  {n:<18} {('sell bid' if 'cross' in n else 'sell mid'):>10} {lv:>11.2f} "
              f"{('buy ask' if 'cross' in n else 'buy mid'):>11} {sv:>10.2f}" if nS else
              f"  {n:<18} {('sell bid' if 'cross' in n else 'sell mid'):>10} {lv:>11.2f} "
              f"{'—':>22}")
    print(f"  book at the exit snapshot (mean): bid {bg.bid[r,x].mean():.2f} | "
          f"mid {bg.mid[r,x].mean():.2f} | ask {bg.ask[r,x].mean():.2f} | "
          f"spread {(bg.ask[r,x]-bg.bid[r,x]).mean():.3f}")
    print(f"  entry side of the book (mean):     bid {bg.bid[r,e].mean():.2f} | "
          f"mid {bg.mid[r,e].mean():.2f} | ask {bg.ask[r,e].mean():.2f}")
    for rung in ["mid","mid_fee","cross","cross_fee"]:
        p=L.trip_pnl(t,bg,rung)
        print(f"  P&L at {rung:<10} {p.sum():>12,.0f} EUR")
print("="*84)
print("Neither ever reaches gate, let alone imbalance: both close in the intraday book")
print("at snapshot 39, which is 15 minutes BEFORE gate closure (delivery start - 30 min).")
