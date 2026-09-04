import numpy as np, pandas as pd, engine as E, trips as T, luck as L, json
mk=E.load_market(); bg=L.BookGrid(mk); F=E.FEE; K=bg.k
spr=bg.ask-bg.bid; mid=bg.mid
pay=json.load(open("payload.json")); sts=json.load(open("spreadts.json"))
print("EACH STRATEGY AGAINST THE BREAKEVEN HOLD")
print("Does the market's own drift, over the horizon it actually holds, cover its friction?")
print(f"{'sub':>13} {'entry snap':>10} {'hold':>7} {'drift avail':>12} {'friction':>9} "
      f"{'drift covers?':>14} {'gross edge':>11}")
rows=[]
for k in E.SUBS:
    b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid")
    r=bg.rows_of(t.product_id.values)
    e=bg.slot_of(t.product_id.values,t.entry_ts.values)
    x=bg.slot_of(t.product_id.values,t.exit_ts.values)
    # drift the market handed anyone holding long over their exact windows
    drift=(mid[r,x]-mid[r,e]).mean()
    fric=((spr[r,e]/2)+(spr[r,x]/2)).mean()+2*F
    edge=(t.dir.values*(mid[r,x]-mid[r,e])).mean()
    hold=t.hold_min.median()
    covers = drift>=fric
    rows.append(dict(sub=k, entry=int(np.median(e)), hold_min=float(hold),
        drift=float(drift), friction=float(fric), covers=bool(covers), edge=float(edge),
        edge_over_fric=float(edge/fric)))
    print(f"{k:>13} {int(np.median(e)):>10} {hold:>6.0f}m {drift:>12.3f} {fric:>9.3f} "
          f"{('YES' if covers else 'no'):>14} {edge:>11.3f}")
print("\nbreakeven hold entering ~10h before gate: 105 min (7 steps of 15 min)")
print("Only SPIKECATCHER (225m) and BLACKBOX (585m) hold long enough for the market's")
print("own drift to pay their friction bill. The other five must earn it from signal.")
pay["spread_ts"]=sts
pay["breakeven_rows"]=rows
json.dump(pay,open("payload.json","w"),separators=(",",":"),default=str)
open("out/_data.js","w").write("window.GAUNTLET="+open("payload.json").read()+";")
print("\npayload updated")
