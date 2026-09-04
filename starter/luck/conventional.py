import numpy as np, pandas as pd, engine as E, trips as T, luck as L, json
from scipy import stats
mk=E.load_market(); bg=L.BookGrid(mk); pay=json.load(open("payload.json"))
Z=1.96
print("THE CONVENTIONAL TEST: t ON THE MEAN RETURN")
print("The identity every desk uses:   t  =  Sharpe(annual) x sqrt(years)")
print("Rule of thumb: t>2 to take it seriously, t>3 before sizing it.\n")
print(f"{'sub':>13} {'days':>5} {'years':>6} {'Sharpe':>7} {'t = SR*sqrt(yr)':>16} "
      f"{'NW t':>6} {'MinTRL days':>12} {'have enough?':>13}")
rows=[]
for k in E.SUBS:
    w=tuple(pay["subs"][k]["window"])
    b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid")
    d=L.daily_from_trips(t,L.trip_pnl(t,bg,"mid"),w)
    n=len(d); m=d.mean(); sd=d.std(ddof=1); srd=m/sd
    sra=srd*np.sqrt(365); yrs=n/365
    g3=float(stats.skew(d)); g4=float(stats.kurtosis(d,fisher=False))
    # Bailey & Lopez de Prado minimum track record length, in periods
    mintrl=1+(1-g3*srd+(g4-1)/4*srd**2)*(Z/srd)**2
    rows.append(dict(sub=k,days=n,years=yrs,sharpe=sra,t=sra*np.sqrt(yrs),
        nw=L.newey_west_t(d),mintrl=mintrl,ok=n>=mintrl))
    print(f"{k:>13} {n:>5} {yrs:>6.2f} {sra:>7.2f} {sra*np.sqrt(yrs):>16.2f} "
          f"{L.newey_west_t(d):>6.2f} {mintrl:>12.0f} {('yes' if n>=mintrl else 'NO'):>13}")
print("\nMinTRL = Bailey & Lopez de Prado minimum track record length: the number of days")
print("you need before a Sharpe THIS size is distinguishable from zero at 95%, given the")
print("strategy's own skew and kurtosis. All seven clear it comfortably at mid.\n")
print("Now the same thing at the honest execution basis, which is where it bites:")
print(f"{'sub':>13} {'Sharpe':>7} {'t = SR*sqrt(yr)':>16} {'MinTRL days':>12} {'have?':>7}")
for k in E.SUBS:
    w=tuple(pay["subs"][k]["window"])
    b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid")
    d=L.daily_from_trips(t,L.trip_pnl(t,bg,"cross_fee"),w)
    n=len(d); m=d.mean(); sd=d.std(ddof=1)
    if sd==0 or m<=0:
        print(f"{k:>13} {L.sharpe(d):>7.2f} {'--':>16} {'n/a (no edge)':>12} {'--':>7}"); continue
    srd=m/sd; sra=srd*np.sqrt(365); yrs=n/365
    g3=float(stats.skew(d)); g4=float(stats.kurtosis(d,fisher=False))
    mintrl=1+(1-g3*srd+(g4-1)/4*srd**2)*(Z/srd)**2
    print(f"{k:>13} {sra:>7.2f} {sra*np.sqrt(yrs):>16.2f} {mintrl:>12.0f} "
          f"{('yes' if n>=mintrl else 'NO'):>7}")
