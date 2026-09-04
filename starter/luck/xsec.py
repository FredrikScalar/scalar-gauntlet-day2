import json, numpy as np, pandas as pd, engine as E, luck as L, trips as T
mk=E.load_market(); bg=L.BookGrid(mk); pay=json.load(open("payload.json"))
day=bg.deliv_day
def bh_cross(w, start=0, h=None, qty=10.0, fee=E.FEE):
    x=bg.k-1 if h is None else min(start+h,bg.k-1)
    # buy the ask at entry, sell the bid at exit, fee both legs
    p=qty*((bg.bid[:,x]-fee)-(bg.ask[:,start]+fee))
    d=pd.Series(p,index=day).groupby(level=0).sum()
    return d.reindex(L._naive(pd.date_range(w[0],w[1],freq="D")),fill_value=0.0).values
rows=[]
for k in E.SUBS:
    s=pay["subs"][k]; w=tuple(s["window"]); r=s["rungs"]["cross_fee"]
    hs=int(round(s["hold_med"]/15)); s0=s["entry_slot_med"]
    dm=bh_cross(w,start=s0,h=hs); df=bh_cross(w)
    s["cross_cmp"]={"matched_pnl":float(dm.sum()),"matched_sharpe":L.sharpe(dm),
                    "full_pnl":float(df.sum()),"full_sharpe":L.sharpe(df),
                    "hold_steps":hs,"entry_slot":s0}
    rows.append(dict(sub=k,strat_pnl=round(r["pnl"]),strat_sh=round(r["sharpe"],2),
        matched_pnl=round(dm.sum()),matched_sh=round(L.sharpe(dm),2),
        full_pnl=round(df.sum()),full_sh=round(L.sharpe(df),2),
        hold=f"{s['hold_med']:.0f}m", days=s["days"]))
json.dump(pay,open("payload.json","w"),separators=(",",":"),default=str)
open("out/_data.js","w").write("window.GAUNTLET="+open("payload.json").read()+";")
pd.set_option("display.width",220)
print("ALL CROSSING THE BOOK (buy ask / sell bid, 0.12 EUR/MWh each side)")
print(pd.DataFrame(rows).to_string(index=False))
