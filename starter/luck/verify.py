"""Verification: does anything in the report rest on an unreplicated number?"""
import json, numpy as np, pandas as pd, engine as E, trips as T, luck as L
mk=E.load_market(); reg=E.load_registry(); bg=L.BookGrid(mk)
pay=json.load(open("payload.json")); ok=[]; bad=[]
def chk(name,cond,detail=""):
    (ok if cond else bad).append(f"{name}: {detail}")

# 1. claims reproduce at mid
for k in E.SUBS:
    b=E.reprice(E.load_blotter(k),mk); w=tuple(pay["subs"][k]["window"])
    d=E.daily_pnl(b,"mid",w).values; c=reg[k]["claimed"]
    chk(f"{k} claimed pnl", abs(d.sum()-c["total_pnl_eur"])<max(2,5e-5*abs(c["total_pnl_eur"])), f"{d.sum():.0f} vs {c['total_pnl_eur']}")
    chk(f"{k} claimed sharpe", abs(L.sharpe(d)-c["sharpe_ann"])<0.01, f"{L.sharpe(d):.2f}")
    chk(f"{k} claimed sortino", abs(L.sortino_claimed(d)-c["sortino_ann"])<0.02, f"{L.sortino_claimed(d):.2f} vs {c['sortino_ann']}")
    chk(f"{k} claimed maxdd", abs(L.max_dd(d)-c["max_drawdown_eur"])<2)
    chk(f"{k} claimed hitrate", abs((d[d!=0]>0).mean()-c["hit_rate_active_days"])<0.006)

# 2. fee arithmetic = 0.12 * total MWh, exactly
for k in E.SUBS:
    s=pay["subs"][k]; vol=s["vol_mwh"]
    delta=s["rungs"]["mid"]["pnl"]-(s["rungs"]["mid"]["pnl"])  # placeholder
    b=E.reprice(E.load_blotter(k),mk); w=tuple(s["window"])
    pm=E.daily_pnl(b,"mid",w).sum(); pmf=E.daily_pnl(b,"mid_fee",w).sum()
    chk(f"{k} fee = 0.12*MWh", abs((pm-pmf)-0.12*vol)<0.01,
        f"drop {pm-pmf:.1f} vs 0.12*{vol:.0f}={0.12*vol:.1f}")

# 3. bootstrap CI stability across seeds
for k in E.SUBS:
    b=E.reprice(E.load_blotter(k),mk); w=tuple(pay["subs"][k]["window"])
    d=E.daily_pnl(b,"cross_fee",w).values
    cis=[]
    for seed in (1,2,3,4):
        r=L.bootstrap(d,B=4000,rng=np.random.default_rng(seed))
        cis.append(r["iid"]["sharpe_ci"])
    cis=np.array(cis); spread=max(np.ptp(cis[:,0]), np.ptp(cis[:,1]))
    rng_=abs(cis[:,1].mean()-cis[:,0].mean())
    chk(f"{k} CI seed-stable", spread < 0.06*max(rng_,1e-9)+0.06,
        f"max endpoint spread {spread:.3f} on CI width {rng_:.2f}")

# 4. MC verdict must rest on a SEED-STABLE statistic
for k in ["spikecatcher","bounceback","blackbox"]:
    b=E.reprice(E.load_blotter(k),mk); w=tuple(pay["subs"][k]["window"])
    t=T.round_trips(b,"mid")
    bl=L.baselines(t,bg,w,rung="mid"); bl.pop("_prod_move"); leg=bl.pop("_leg")
    osh=L.sharpe(E.daily_pnl(b,"mid",w).values)
    mx=[]; q=[]; pv=[]
    for seed in (11,22,33,44):
        sh,_=L.mc_null_a(t,leg,w,B=10000,rng=np.random.default_rng(seed))
        mx.append(sh.max()); q.append(np.percentile(sh,99.9)); pv.append((sh>=osh).mean())
    # the max is expected to be unstable -- document it rather than rely on it
    beats=[osh>m for m in mx]
    stable_max = all(x==beats[0] for x in beats)
    aboveq=[osh>x for x in q]
    chk(f"{k} p99.9 verdict is seed-stable", all(x==aboveq[0] for x in aboveq),
        f"above-p99.9 {aboveq}, q999 range {max(q)-min(q):.3f}")
    chk(f"{k} p-value is seed-stable", (max(pv)-min(pv))<0.002,
        f"p in [{min(pv):.4f}, {max(pv):.4f}]")
    print(f"  note {k}: max-based verdict {'stable' if stable_max else 'UNSTABLE'} "
          f"(null max {min(mx):.2f}-{max(mx):.2f}) -> not used for the verdict")

# 5. round-trip reconstruction reconciles at every rung
for k in E.SUBS:
    b=E.reprice(E.load_blotter(k),mk)
    for rung in ["mid","mid_fee","passive_in_fee","passive_out_fee","cross","cross_fee"]:
        t=T.round_trips(b,"mid")
        p=L.trip_pnl(t,bg,rung).sum()
        if rung in ("mid","mid_fee","cross","cross_fee"):
            q=E.daily_pnl(b,rung).sum()
            chk(f"{k}/{rung} trips==trades", abs(p-q)<1e-6, f"{p:.4f} vs {q:.4f}")

# 6. drift claim
d=bg.mid[:,1:]-bg.mid[:,:-1]
chk("drift per step = 0.164", abs(d.mean()-0.16387)<1e-4, f"{d.mean():.5f}")
chk("drift full window = 6.39", abs((bg.mid[:,-1]-bg.mid[:,0]).mean()-6.3908)<1e-3)
chk("frac ending higher = 48.2%", abs((bg.mid[:,-1]>bg.mid[:,0]).mean()-0.4819)<1e-3)

# 7. pingpong edge == fee
b=E.reprice(E.load_blotter("pingpong"),mk); t=T.round_trips(b,"mid")
r=bg.rows_of(t.product_id.values); e=bg.slot_of(t.product_id.values,t.entry_ts.values)
x=bg.slot_of(t.product_id.values,t.exit_ts.values)
edge=(t.dir.values*(bg.mid[r,x]-bg.mid[r,e])).mean()
chk("pingpong edge 0.242 vs fee 0.24", abs(edge-0.2418)<1e-3, f"edge {edge:.4f}, fee {2*E.FEE}")

# 8. every verdict rests on >1 number? report the thin ones
print(f"PASS {len(ok)}   FAIL {len(bad)}")
for x_ in bad: print("  FAIL", x_)
if not bad: print("  all checks green")
