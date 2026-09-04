import numpy as np, pandas as pd, engine as E, trips as T, luck as L, json
mk=E.load_market(); bg=L.BookGrid(mk)
da=mk["da"].set_index("product_id").loc[bg.pids,"da_price_eur_mwh"].values
F=E.FEE
print("What is and is not in the pack:")
print("  da_auction.csv  -> day-ahead price per product        (a price)")
print("  actuals.csv     -> wind/solar/load outturn in MW      (a volume, not a price)")
print("  no imbalance / balancing price series exists anywhere in data/\n")
print("mean final-snapshot mid minus DA price: +%.2f EUR/MWh  (intraday closes ABOVE the auction)"%((bg.mid[:,-1]-da).mean()))
print("sd of that gap: %.2f\n"%((bg.mid[:,-1]-da).std()))

rows=[]
for k in E.SUBS:
    b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid")
    w=("%s"%b.product_delivery.min().date(),)
    reg=json.load(open("payload.json"))["subs"][k]; w=tuple(reg["window"])
    r=bg.rows_of(t.product_id.values)
    e=bg.slot_of(t.product_id.values,t.entry_ts.values)
    x=bg.slot_of(t.product_id.values,t.exit_ts.values)
    d=t.dir.values; q=t.qty.values
    LAST=bg.k-1
    # entry always crosses + fee
    entry=np.where(d>0, bg.ask[r,e]+F, bg.bid[r,e]-F)
    def pnl(exit_px, exit_fee=True):
        f=F if exit_fee else 0.0
        ex=np.where(d>0, exit_px-f, exit_px+f)
        return d*q*(ex-entry)
    v={}
    # A: as recorded, crossed both legs
    v["as_recorded"]=pnl(np.where(d>0,bg.bid[r,x],bg.ask[r,x]))
    # B: ignore their exit, liquidate at the LAST snapshot before gate (still crossing)
    v["hold_to_gate_cross"]=pnl(np.where(d>0,bg.bid[r,LAST],bg.ask[r,LAST]))
    # C: hold to gate, marked at final mid (no exit spread, exit fee still paid)
    v["hold_to_gate_mid"]=pnl(bg.mid[r,LAST])
    # D: run into delivery, settled at the day-ahead price, no exit spread
    v["settle_at_da"]=pnl(da[r])
    o={"sub":k,"trips":len(t),"hold":f"{t.hold_min.median():.0f}m","long":f"{(d>0).mean():.0%}"}
    for n,p in v.items():
        dd=L.daily_from_trips(t,p,w)
        o[n+"_pnl"]=round(p.sum()); o[n+"_sh"]=round(L.sharpe(dd),2)
    rows.append(o)
df=pd.DataFrame(rows); pd.set_option("display.width",260)
print("ENTRY ALWAYS CROSSES THE BOOK + FEE.  Only the EXIT convention changes:\n")
print(df[["sub","hold","long","as_recorded_pnl","as_recorded_sh",
          "hold_to_gate_cross_pnl","hold_to_gate_cross_sh",
          "hold_to_gate_mid_pnl","hold_to_gate_mid_sh",
          "settle_at_da_pnl","settle_at_da_sh"]].to_string(index=False))
df.to_json("exitvariants.json",orient="records")
