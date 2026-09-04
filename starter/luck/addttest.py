import numpy as np, json, engine as E, trips as T, luck as L
from scipy import stats
mk=E.load_market(); bg=L.BookGrid(mk); pay=json.load(open("payload.json")); Z=1.96
for k in E.SUBS:
    s=pay["subs"][k]; w=tuple(s["window"])
    b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid")
    for rung in ["mid","passive_in_fee","cross_fee"]:
        d=L.daily_from_trips(t,L.trip_pnl(t,bg,rung),w)
        n=len(d); m=d.mean(); sd=d.std(ddof=1); yrs=n/365.0
        o={"years":yrs,"n":n}
        if sd>0:
            srd=m/sd; sra=srd*np.sqrt(365)
            g3=float(stats.skew(d)); g4=float(stats.kurtosis(d,fisher=False))
            o["t_conv"]=float(sra*np.sqrt(yrs))
            o["t_plain"]=float(m/(sd/np.sqrt(n)))
            if srd>0:
                o["mintrl"]=float(1+(1-g3*srd+(g4-1)/4*srd**2)*(Z/srd)**2)
                o["mintrl_ok"]=bool(n>=o["mintrl"])
            else:
                o["mintrl"]=None; o["mintrl_ok"]=False
        else:
            o["t_conv"]=None; o["t_plain"]=None; o["mintrl"]=None; o["mintrl_ok"]=False
        # per-trip conventional t
        tp=L.trip_pnl(t,bg,rung)/t.qty.values
        o["t_trip_plain"]=float(tp.mean()/(tp.std(ddof=1)/np.sqrt(len(tp)))) if tp.std(ddof=1)>0 else None
        s["rungs"][rung]["conv"]=o
json.dump(pay,open("payload.json","w"),separators=(",",":"),default=str)
open("out/_data.js","w").write("window.GAUNTLET="+open("payload.json").read()+";")
print("conventional t + MinTRL added for all seven x 3 bases")
