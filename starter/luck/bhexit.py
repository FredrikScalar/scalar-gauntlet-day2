import numpy as np, pandas as pd, engine as E, luck as L
mk=E.load_market(); bg=L.BookGrid(mk); F=E.FEE
N=len(bg.pids); LAST=bg.k-1
print(f"buy-and-hold: 10 MW long on every one of {N:,} products")
print(f"  ENTRY snapshot 0    = delivery start - 10h30m  (first row in the book)")
print(f"  EXIT  snapshot {LAST}   = delivery start - 45m      (LAST row in the book,")
print(f"                          i.e. 15 min BEFORE gate closure at start-30m)\n")
print("mean book level at each end (EUR/MWh, across all products):")
print(f"  entry  bid {bg.bid[:,0].mean():7.2f}   mid {bg.mid[:,0].mean():7.2f}   ask {bg.ask[:,0].mean():7.2f}   spread {(bg.ask[:,0]-bg.bid[:,0]).mean():.3f}")
print(f"  exit   bid {bg.bid[:,LAST].mean():7.2f}   mid {bg.mid[:,LAST].mean():7.2f}   ask {bg.ask[:,LAST].mean():7.2f}   spread {(bg.ask[:,LAST]-bg.bid[:,LAST]).mean():.3f}\n")
print("EXIT PRICE ACTUALLY USED, per rung (a long SELLS to close):")
rows=[("mid",        bg.mid[:,LAST],            bg.mid[:,0]),
      ("mid_fee",    bg.mid[:,LAST]-F,          bg.mid[:,0]+F),
      ("cross",      bg.bid[:,LAST],            bg.ask[:,0]),
      ("cross_fee",  bg.bid[:,LAST]-F,          bg.ask[:,0]+F)]
print(f"  {'rung':<12} {'exit price':>12} {'entry price':>12} {'gross/MWh':>11}   what the exit price is")
what={"mid":"the mid at snapshot 39 (a price nobody quoted)",
      "mid_fee":"mid at 39, minus the 0.12 fee",
      "cross":"the BID at snapshot 39 — you hit the bid to get out",
      "cross_fee":"bid at 39, minus the 0.12 fee"}
for n,ex,en in rows:
    print(f"  {n:<12} {ex.mean():12.3f} {en.mean():12.3f} {(ex-en).mean():11.3f}   {what[n]}")
print()
day=bg.deliv_day
for n,ex,en in rows:
    p=10.0*(ex-en)
    d=pd.Series(p,index=day).groupby(level=0).sum()
    d=d.reindex(L._naive(pd.date_range("2025-01-01","2026-06-30",freq="D")),fill_value=0.0).values
    print(f"  {n:<12} full 546-day window: {d.sum():>12,.0f} EUR   Sharpe {L.sharpe(d):>6.2f}")
print()
print("Same exit snapshot as SPIKECATCHER and BLACKBOX use (both exit at 39 on 100% of")
print("trips), so the comparison in section 08 is apples-to-apples on the exit side.")
print("Buy-and-hold crosses the spread ONCE over 9h45; a 15-min strategy crosses it")
print("every round trip. That asymmetry, not any signal, is what section 08 measures.")
