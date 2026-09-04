"""Recompute the MC nulls storing STABLE quantiles, not just the max."""
import json, time, numpy as np, pandas as pd, engine as E, trips as T, luck as L
t0=time.time(); mk=E.load_market(); bg=L.BookGrid(mk); pay=json.load(open("payload.json"))
stab={r["sub"]:r for r in json.load(open("seedstab.json"))}
for k in E.SUBS:
    s=pay["subs"][k]; w=tuple(s["window"])
    b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid")
    for rung in ["mid","passive_in_fee","cross_fee"]:
        tp=L.trip_pnl(t,bg,rung); tt=t.copy(); tt["pnl"]=tp
        bl=L.baselines(tt,bg,w,rung=rung); bl.pop("_prod_move"); leg=bl.pop("_leg")
        obs=s["rungs"][rung]["sharpe"]
        for tag,fn in (("mc_a",lambda: L.mc_null_a(tt,leg,w,B=10000,rng=np.random.default_rng(7))),
                       ("mc_b",lambda: L.mc_null_b(tt,bg,w,rung=rung,B=10000,rng=np.random.default_rng(7)))):
            sh,tot=fn()
            m=s["rungs"][rung][tag]
            m["q99"]=float(np.percentile(sh,99))
            m["q999"]=float(np.percentile(sh,99.9))
            m["above_q999"]=bool(obs>m["q999"])
            m["sharpe_null_max"]=float(sh.max())
            m["sharpe_beats_best_run"]=bool(obs>sh.max())
            m["sharpe_p_value"]=float((sh>=obs).mean())
            m["sharpe_pctile"]=float((sh<obs).mean()*100)
        print(f"  {k:13s} {rung:16s} {time.time()-t0:.0f}s",flush=True)
    st=stab[k]
    s["seed_stability"]={"max_lo":st["max_lo"],"max_hi":st["max_hi"],
        "max_range":st["max_range"],"beats_max":st["beats_max"],
        "q999_range":st["q999_range"],"above_q999":st["above_q999"],
        "p_lo":st["p_lo"],"p_hi":st["p_hi"],"seeds":5}
json.dump(pay,open("payload.json","w"),separators=(",",":"),default=str)
open("out/_data.js","w").write("window.GAUNTLET="+open("payload.json").read()+";")
print("done",round(time.time()-t0),"s")
