import numpy as np, pandas as pd, engine as E, trips as T, luck as L, json
from scipy import stats
mk=E.load_market(); bg=L.BookGrid(mk); pay=json.load(open("payload.json"))
k="windfall"; w=tuple(pay["subs"][k]["window"])
b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid")
d=L.daily_from_trips(t,L.trip_pnl(t,bg,"mid"),w)
n=len(d); m=d.mean(); sd=d.std(ddof=1)
g3=float(stats.skew(d)); g4=float(stats.kurtosis(d,fisher=False))
print(f"WINDFALL at mid: n={n} days, mean daily P&L={m:.2f}, SD={sd:.2f}")
print(f"                 skew={g3:.2f}, kurtosis={g4:.1f}\n")

print("1) PLAIN t ON THE MEAN  =  mean / (SD/sqrt(n))")
se_m=sd/np.sqrt(n)
print(f"   SE of the mean = {sd:.2f}/sqrt({n}) = {se_m:.3f}")
print(f"   t = {m:.2f} / {se_m:.3f} = {m/se_m:.2f}")
sr_d=m/sd
print(f"   identity check: t = daily Sharpe x sqrt(n) = {sr_d:.4f} x {np.sqrt(n):.2f} = {sr_d*np.sqrt(n):.2f}\n")

print("2) NEWEY-WEST t ON THE MEAN  (same thing, SE widened for autocorrelation)")
print(f"   reported = {L.newey_west_t(d):.2f}   (vs {m/se_m:.2f} plain)")
print(f"   AR(1) of daily P&L = {L.ar1(d):.3f}  -> mild positive autocorrelation widens the SE\n")

print("3) BOOTSTRAP t ON THE SHARPE  =  Sharpe / SD of the bootstrap Sharpes")
rng=np.random.default_rng(7)
idx=rng.integers(0,n,size=(20000,n)); s=d[idx]
mu=s.mean(axis=1); sdb=s.std(axis=1,ddof=1); shb=mu/sdb*np.sqrt(365)
print(f"   observed Sharpe = {L.sharpe(d):.3f}")
print(f"   bootstrap SE    = {shb.std(ddof=1):.4f}")
print(f"   t = {L.sharpe(d):.3f} / {shb.std(ddof=1):.4f} = {L.sharpe(d)/shb.std(ddof=1):.2f}\n")

print("   WHY IS THAT ~2x THE t ON THE MEAN?")
va=(1 - g3*sr_d + (g4-1)/4*sr_d**2)/(n-1)
print(f"   analytic Var(SR) with skew/kurtosis = (1 - g3*SR + (g4-1)/4*SR^2)/(n-1)")
print(f"     = (1 - {g3:.2f}*{sr_d:.4f} + {(g4-1)/4:.3f}*{sr_d**2:.4f})/{n-1}")
print(f"     = {(1 - g3*sr_d + (g4-1)/4*sr_d**2):.4f}/{n-1} = {va:.8f}")
print(f"   analytic SE annualised = {np.sqrt(va)*np.sqrt(365):.4f}  <- matches the bootstrap {shb.std(ddof=1):.4f}")
va_norm=(1+0.5*sr_d**2)/n
print(f"   if returns were NORMAL it would be {np.sqrt(va_norm)*np.sqrt(365):.4f} -- twice as wide")
print(f"   the -g3*SR term (={-g3*sr_d:.3f}) is what shrinks it: positive skew makes the")
print(f"   RATIO easier to pin down than the mean alone.\n")
print(f"   corr(bootstrap mean, bootstrap SD) = {np.corrcoef(mu,sdb)[0,1]:.3f}")
print("   -> a resample that catches a big up-day gets BOTH a higher mean and a higher")
print("      SD, so the two errors partly cancel in the ratio. That is the mechanism.\n")

print("4) PER-TRIP t (section 05) = Newey-West t on the per-round-trip margin")
tp=L.trip_pnl(t,bg,"mid")/t.qty.values
print(f"   {len(tp)} trips, mean {tp.mean():.4f} EUR/MWh, SD {tp.std(ddof=1):.3f}")
print(f"   t = {L.newey_west_t(tp):.2f}   (one observation per DECISION, no calendar)\n")
print("READING THRESHOLDS (two-sided, large sample)")
for tv,p in [(1.65,"10%"),(1.96,"5%"),(2.58,"1%"),(3.29,"0.1%")]:
    print(f"   |t| > {tv:<5} ~ p < {p}")
