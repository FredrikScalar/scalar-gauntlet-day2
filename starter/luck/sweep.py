import json, numpy as np, pandas as pd, engine as E, trips as T, luck as L
mk=E.load_market(); bg=L.BookGrid(mk); res=json.load(open("results.json"))
CAPS=[0.0,0.25,0.5,0.75,1.0]
for k in E.SUBS:
    rec=res["subs"][k]; w=tuple(rec["window"])
    b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid")
    r=bg.rows_of(t.product_id.values)
    e=bg.slot_of(t.product_id.values,t.entry_ts.values)
    x=bg.slot_of(t.product_id.values,t.exit_ts.values)
    mv=bg.mid[r,x]-bg.mid[r,e]
    edge=t.dir.values*mv                                  # gross EUR/MWh at mid
    hs=(bg.ask[r,e]-bg.bid[r,e])/2 + (bg.ask[r,x]-bg.bid[r,x])/2   # both half-spreads
    sweep={}
    for c in CAPS:
        net=(edge-((1-c)*hs+2*E.FEE))*t.qty.values
        d=L.daily_from_trips(t,net,w)
        sweep[str(c)]={"pnl":float(net.sum()),"sharpe":L.sharpe(d),
                       "sortino":L.sortino_claimed(d)}
    # breakeven capture fraction (P&L = 0), and breakeven fee at full crossing
    lo,hi=0.0,1.0
    f=lambda c:((edge-((1-c)*hs+2*E.FEE))*t.qty.values).sum()
    be=None
    if f(0)<0<f(1):
        for _ in range(60):
            m=(lo+hi)/2
            if f(m)<0: lo=m
            else: hi=m
        be=float((lo+hi)/2)
    gross=float((edge*t.qty.values).sum())
    vol=float(t.qty.values.sum())
    rec["capture_sweep"]={"caps":CAPS,"sweep":sweep,"breakeven_capture":be,
        "gross_edge_eur_mwh":float(edge.mean()),
        "gross_edge_med":float(np.median(edge)),
        "rt_cost_full_cross":float((hs+2*E.FEE).mean()),
        "rt_cost_one_leg":float((hs/2+2*E.FEE).mean()),
        "fee_rt":2*E.FEE,
        "edge_over_fee":float(edge.mean()/(2*E.FEE)),
        "frac_trips_beat_full":float((edge>(hs+2*E.FEE)).mean()),
        "frac_trips_beat_oneleg":float((edge>(hs/2+2*E.FEE)).mean()),
        "frac_trips_beat_fee":float((edge>2*E.FEE).mean()),
        "gross_pnl_mid":gross,"vol_mwh":vol,
        "px_at_mid_frac":float(((b.price-b.bid_px_1)/(b.ask_px_1-b.bid_px_1)
                                 -0.5).abs().lt(0.02).mean())}
json.dump(res,open("results.json","w"),default=str)
rows=[]
for k in E.SUBS:
    c=res["subs"][k]["capture_sweep"]
    rows.append({"sub":k,"gross_edge":round(c["gross_edge_eur_mwh"],3),
      "fee_rt":c["fee_rt"],"edge/fee":round(c["edge_over_fee"],2),
      "cost_full":round(c["rt_cost_full_cross"],3),
      "be_capture":None if c["breakeven_capture"] is None else round(c["breakeven_capture"]*100,1),
      **{f"cap{int(float(q)*100)}_sh":round(v["sharpe"],2) for q,v in c["sweep"].items()}})
pd.set_option("display.width",250)
print(pd.DataFrame(rows).to_string(index=False))
