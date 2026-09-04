"""How seed-stable is 'beats the luckiest of 10,000 runs'? The max of a null is
the least stable statistic you can choose. Quantiles are stable; the max is not."""
import numpy as np, pandas as pd, engine as E, trips as T, luck as L, json
mk=E.load_market(); bg=L.BookGrid(mk); pay=json.load(open("payload.json"))
SEEDS=[101,202,303,404,505]
rows=[]
for k in E.SUBS:
    b=E.reprice(E.load_blotter(k),mk); w=tuple(pay["subs"][k]["window"])
    t=T.round_trips(b,"mid")
    bl=L.baselines(t,bg,w,rung="mid"); bl.pop("_prod_move"); leg=bl.pop("_leg")
    obs=L.sharpe(E.daily_pnl(b,"mid",w).values)
    mx=[]; q999=[]; q99=[]; pv=[]
    for s in SEEDS:
        sh,_=L.mc_null_a(t,leg,w,B=10000,rng=np.random.default_rng(s))
        mx.append(sh.max()); q999.append(np.percentile(sh,99.9))
        q99.append(np.percentile(sh,99)); pv.append((sh>=obs).mean())
    beats=[obs>m for m in mx]
    rows.append(dict(sub=k, obs=round(obs,2),
      max_lo=round(min(mx),2), max_hi=round(max(mx),2), max_range=round(max(mx)-min(mx),2),
      beats_max="always" if all(beats) else ("never" if not any(beats) else "*** FLIPS ***"),
      q999_lo=round(min(q999),2), q999_hi=round(max(q999),2),
      q999_range=round(max(q999)-min(q999),2),
      above_q999="yes" if all(obs>q for q in q999) else ("no" if not any(obs>q for q in q999) else "FLIPS"),
      p_lo=f"{min(pv):.4f}", p_hi=f"{max(pv):.4f}"))
df=pd.DataFrame(rows); pd.set_option("display.width",260)
print("NULL A, 10,000 runs, five independent seeds, priced at mid\n")
print(df.to_string(index=False))
df.to_json("seedstab.json",orient="records")
