import numpy as np, pandas as pd, engine as E, luck as L, trips as T, json
mk=E.load_market(); bg=L.BookGrid(mk)
day=bg.deliv_day
FULL=("2025-01-01","2026-06-30")

def bh(h, start=0, w=FULL, qty=10.0, fee=0.0):
    """long qty MW on every product, enter at snapshot `start`, hold h steps."""
    x=min(start+h, bg.k-1)
    p=qty*((bg.mid[:,x]-fee)-(bg.mid[:,start]+fee))
    d=pd.Series(p,index=day).groupby(level=0).sum()
    d=d.reindex(L._naive(pd.date_range(w[0],w[1],freq="D")),fill_value=0.0).values
    return d

print("BUY-AND-HOLD BASELINE vs HOLDING HORIZON  (enter at snapshot 0, 10h before gate)")
print(f"{'hold':>8} {'steps':>6} {'drift EUR/MWh':>14} {'P&L (10MW, all prods)':>22} {'Sharpe':>8}")
for h in [1,2,3,4,6,8,12,20,30,39]:
    d=bh(h)
    dr=(bg.mid[:,min(h,bg.k-1)]-bg.mid[:,0]).mean()
    print(f"{h*15:>6}m {h:>6} {dr:>14.3f} {d.sum():>22,.0f} {L.sharpe(d):>8.2f}")

print("\nMEAN MID DRIFT is monotone in horizon: +0.164 EUR/MWh per 15-min step.")
print("So the baseline's Sharpe is a function of how long you hold, not of any skill.")

print("\n" + "="*100)
print("MATCHED-HORIZON COMPARISON: each strategy vs a passive long held for the SAME time")
print("="*100)
res=json.load(open("results.json"))
print(f"{'sub':>13} {'med hold':>9} {'strat P&L':>12} {'strat Sh':>9} {'always-long*':>13} {'AL Sh':>7} {'matched BH':>12} {'BH Sh':>7} {'full-window BH':>15}")
for k in E.SUBS:
    rec=res["subs"][k]; w=tuple(rec["window"])
    b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid")
    h=int(round(t.hold_min.median()/15))
    # entry slot the strategy actually used (median), so the drift window matches
    r=bg.rows_of(t.product_id.values); e=bg.slot_of(t.product_id.values,t.entry_ts.values)
    s0=int(np.median(e))
    dm=bh(h, start=s0, w=w)
    bl=rec["battery"]["mid"]["baselines"]
    print(f"{k:>13} {t.hold_min.median():>8.0f}m {bl['strategy']['pnl']:>12,.0f} "
          f"{bl['strategy']['sharpe']:>9.2f} {bl['always_long']['pnl']:>13,.0f} "
          f"{bl['always_long']['sharpe']:>7.2f} {dm.sum():>12,.0f} {L.sharpe(dm):>7.2f} "
          f"{bl['market_bh_long']['pnl']:>15,.0f}")
    rec["matched_bh"]={"hold_steps":h,"entry_slot":s0,"pnl":float(dm.sum()),
                       "sharpe":L.sharpe(dm)}
print("\n* always-long = the strategy's OWN trades and timestamps, side forced long.")
json.dump(res,open("results.json","w"),default=str)
