import numpy as np, pandas as pd, engine as E, trips as T, luck as L, json
mk=E.load_market(); bg=L.BookGrid(mk); F=E.FEE
pay=json.load(open("payload.json"))

# where do the snapshots sit relative to gate?
p0=mk["products"].iloc[0]
print("product 0 delivery start:", p0.delivery_start, " gate = start - 30min")
print("snapshot 0 ts:", bg.ts[0,0], " snapshot 39 ts:", bg.ts[0,-1])
print("snapshot 39 is", (p0.delivery_start.tz_localize(None)-pd.Timestamp(bg.ts[0,-1])).total_seconds()/60,
      "min before delivery start =",
      (p0.delivery_start.tz_localize(None)-pd.Timestamp(bg.ts[0,-1])).total_seconds()/60-30,
      "min before the LAST TRADEABLE snapshot is gate\n")

print("DOES ANY BLOTTER LEAVE A POSITION OPEN AT GATE?")
for k in E.SUBS:
    b=E.reprice(E.load_blotter(k),mk)
    net=b.groupby("product_id").apply(
        lambda x:(np.where(x.side.eq("BUY"),1,-1)*x.qty_mw).sum(), include_groups=False)
    print(f"  {k:13s} products with a non-zero net position at gate: {(net.abs()>1e-9).sum()}")
print()

print("EXIT SNAPSHOT INDEX (0 = 10h before gate, 39 = last snapshot before gate)")
rows=[]
for k in E.SUBS:
    b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid")
    r=bg.rows_of(t.product_id.values)
    e=bg.slot_of(t.product_id.values,t.entry_ts.values)
    x=bg.slot_of(t.product_id.values,t.exit_ts.values)
    d=t.dir.values
    at39=(x==bg.k-1).mean()
    # the actual close prices used, at cross_fee
    close_px=np.where(d>0, bg.bid[r,x]-F, bg.ask[r,x]+F)
    close_mid=bg.mid[r,x]
    rows.append(dict(sub=k, entry_slot_med=int(np.median(e)), exit_slot_med=int(np.median(x)),
      exit_slot_min=int(x.min()), exit_slot_max=int(x.max()),
      frac_exit_at_last=f"{at39:.1%}",
      mean_close_mid=round(close_mid.mean(),2),
      mean_close_used=round(close_px.mean(),2),
      mean_bid_at_exit=round(bg.bid[r,x].mean(),2),
      mean_ask_at_exit=round(bg.ask[r,x].mean(),2)))
df=pd.DataFrame(rows); pd.set_option("display.width",250)
print(df.to_string(index=False))
