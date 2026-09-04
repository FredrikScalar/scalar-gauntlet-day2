import numpy as np, pandas as pd, engine as E, luck as L, json
mk=E.load_market(); bg=L.BookGrid(mk); F=E.FEE; K=bg.k
spr=(bg.ask-bg.bid)                      # (n_products, 40)
mean_spr=spr.mean(axis=0)
mid=bg.mid
cum_drift=(mid-mid[:,[0]]).mean(axis=0)
print("SPREAD TERM STRUCTURE  (snapshot 0 = delivery start - 10h30m, 39 = start - 45m)")
print(f"{'snap':>5} {'mins to gate':>13} {'mean spread':>12} {'half-spread':>12} {'cum drift':>10}")
for i in list(range(0,40,4))+[39]:
    mins = (K-1-i)*15 + 15          # snapshot 39 is 15 min before gate
    print(f"{i:>5} {mins:>13} {mean_spr[i]:>12.3f} {mean_spr[i]/2:>12.3f} {cum_drift[i]:>10.3f}")
print(f"\nspread at snapshot 0:  {mean_spr[0]:.3f}   at 39: {mean_spr[-1]:.3f}   "
      f"tightening: {(mean_spr[-1]/mean_spr[0]-1)*100:.1f}%")
print(f"widest {mean_spr.max():.3f} at snap {int(mean_spr.argmax())}, "
      f"tightest {mean_spr.min():.3f} at snap {int(mean_spr.argmin())}")
# monotone?
d=np.diff(mean_spr); print(f"steps that tighten: {(d<0).sum()} of {len(d)}  (monotone: {bool((d<0).all())})")

# BREAKEVEN HOLD: how long must you hold before drift pays for one round trip?
print("\nBREAKEVEN HOLD — entering at snapshot s, how many 15-min steps until the")
print("accumulated drift covers one round trip of friction (both half-spreads + 2 fees)?")
print(f"{'entry snap':>11} {'mins to gate':>13} {'breakeven hold':>15} {'cost EUR/MWh':>13}")
rows=[]
for s in [0,4,8,12,16,20,24,28,32,36]:
    best=None
    for hsteps in range(1,K-s):
        drift=(mid[:,s+hsteps]-mid[:,s]).mean()
        cost=(spr[:,s]/2).mean()+(spr[:,s+hsteps]/2).mean()+2*F
        if drift>=cost: best=hsteps; break
    cost0=(spr[:,s]/2).mean()+(spr[:,min(s+(best or 1),K-1)]/2).mean()+2*F
    mins=(K-1-s)*15+15
    rows.append([s,mins,best,round(cost0,3)])
    print(f"{s:>11} {mins:>13} {(str(best*15)+' min') if best else 'never':>15} {cost0:>13.3f}")
json.dump({"mean_spread":[round(float(v),4) for v in mean_spr],
           "cum_drift":[round(float(v),4) for v in cum_drift],
           "breakeven":rows,
           "spread_0":float(mean_spr[0]),"spread_39":float(mean_spr[-1])},
          open("spreadts.json","w"))
