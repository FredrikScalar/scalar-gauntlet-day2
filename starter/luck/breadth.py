import numpy as np, pandas as pd, engine as E, trips as T, luck as L, json
mk=E.load_market(); bg=L.BookGrid(mk); F=E.FEE; K=bg.k
spr=bg.ask-bg.bid; mid=bg.mid; pay=json.load(open("payload.json"))
print("IS THE PASSIVE BENCHMARK'S HIGHER SHARPE AN EDGE ADVANTAGE, OR JUST BREADTH?")
print("The passive long holds all 24 products every day. A concentrated strategy holds")
print("one or two. Sharpe rewards that diversification even at an identical per-MWh edge.\n")
print(f"{'sub':>13} {'prods/day':>10} {'edge/MWh':>9} {'passive same':>13} {'edge ratio':>11} "
      f"{'Sharpe':>7} {'passive Sh':>11} {'breadth x':>10}")
rows=[]
for k in E.SUBS:
    b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid"); s=pay["subs"][k]
    r=bg.rows_of(t.product_id.values)
    e=bg.slot_of(t.product_id.values,t.entry_ts.values)
    x=bg.slot_of(t.product_id.values,t.exit_ts.values)
    edge=(t.dir.values*(mid[r,x]-mid[r,e])).mean()          # EUR per position-MWh
    # the SAME window, but averaged over the whole market (no hour selection)
    s0=int(np.median(e)); hh=int(round(s.get("hold_med",0)/15)); x0=min(s0+hh,K-1)
    passive_edge=(mid[:,x0]-mid[:,s0]).mean()
    nprod=len(t.product_id.unique()); days=s["days"]
    breadth=24.0/(nprod/days)
    rows.append(dict(sub=k, prods_day=nprod/days, edge=float(edge),
        passive_edge=float(passive_edge), ratio=float(edge/passive_edge) if passive_edge else None,
        sharpe=s["rungs"]["mid"]["sharpe"], passive_sharpe=s["matched_bh"]["sharpe"],
        breadth=float(breadth)))
    print(f"{k:>13} {nprod/days:>10.2f} {edge:>9.3f} {passive_edge:>13.3f} "
          f"{(edge/passive_edge if passive_edge else 0):>11.2f}x {s['rungs']['mid']['sharpe']:>7.2f} "
          f"{s['matched_bh']['sharpe']:>11.2f} {breadth:>9.1f}x")
print("\nedge ratio > 1 means the strategy's hour/product selection genuinely beats holding")
print("the whole market over the same clock window, per MWh of position.")
print("breadth x = how many times more diversified the passive benchmark is.")
pay["breadth_rows"]=rows
json.dump(pay,open("payload.json","w"),separators=(",",":"),default=str)
open("out/_data.js","w").write("window.GAUNTLET="+open("payload.json").read()+";")
