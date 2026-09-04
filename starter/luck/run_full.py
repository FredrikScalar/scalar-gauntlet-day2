import json, time, numpy as np, pandas as pd
from scipy import stats
import engine as E, trips as T, luck as L

t0=time.time()
mk = E.load_market(); reg = E.load_registry(); bg = L.BookGrid(mk)
print(f"loaded {time.time()-t0:.0f}s", flush=True)

RUNGS = ["mid","mid_fee","passive_in_fee","passive_out_fee","cross","cross_fee"]
BATTERY = ["mid","passive_in_fee","cross_fee"]      # rungs the full battery runs on
HONEST  = "cross_fee"

def metrics(d, vol):
    return {"pnl":float(d.sum()), "sharpe":L.sharpe(d),
            "sortino_claimed":L.sortino_claimed(d), "sortino_strict":L.sortino_strict(d),
            "max_dd":L.max_dd(d), "margin":float(d.sum()/vol),
            "active_frac":float((d!=0).mean()),
            "hit_rate":float((d[d!=0]>0).mean()) if (d!=0).any() else float("nan")}

# ---- pass 1: ladder ------------------------------------------------------
store, dailies, tripstore = {}, {}, {}
for k in E.SUBS:
    e = reg[k]; w = (str(e["backtest_window"]["start"]), str(e["backtest_window"]["end"]))
    b = E.reprice(E.load_blotter(k), mk); vol = float(b.qty_mw.sum())
    t = T.round_trips(b, "mid")            # geometry only; P&L re-priced per rung
    tripstore[k] = t
    rec = {"key":k, "name":e["name"], "class":e["class"], "window":w,
           "days":e["backtest_window"]["days"], "declared_trials":e["declared_trials"],
           "dev_window":str(e.get("dev_window")), "params":e["params"],
           "claimed":e["claimed"], "n_trades":int(len(b)), "vol_mwh":vol,
           "n_trips":int(len(t)), "long_frac":float((t.dir>0).mean()),
           "hold_med_min":float(t.hold_min.median()),
           "half_spread_mean":float(b.half_spread.mean()),
           "half_spread_med":float(b.half_spread.median()),
           "depth_short_mw":float(b.depth_short_mw.sum()), "ladder":{}}
    for r in RUNGS:
        d = L.daily_from_trips(t, L.trip_pnl(t, bg, r), w)
        rec["ladder"][r] = metrics(d, vol); dailies[(k,r)] = d
    # walk-the-book rung still needs trade-level depth walking
    dw = E.daily_pnl(b, "walk_fee", w).values
    rec["ladder"]["walk_fee"] = metrics(dw, vol); dailies[(k,"walk_fee")] = dw
    store[k] = rec
print(f"ladder done {time.time()-t0:.0f}s", flush=True)

V_SEARCH = float(np.var([dailies[(k,"mid")].mean()/dailies[(k,"mid")].std(ddof=1)
                         for k in E.SUBS], ddof=1))
print("V_search =", round(V_SEARCH,6), flush=True)

# ---- pass 2: the battery -------------------------------------------------
for k in E.SUBS:
    e = reg[k]; w = tuple(store[k]["window"]); N = e["declared_trials"]
    t = tripstore[k]; rec = store[k]; rec["battery"] = {}
    for rung in BATTERY:
        d = dailies[(k,rung)]; tp = L.trip_pnl(t, bg, rung)
        r = {}
        r["boot"] = L.bootstrap(d)
        r["nw_t_mean_daily"] = L.newey_west_t(d)
        r["ar1"] = L.ar1(d)
        sd = d.std(ddof=1); sr = d.mean()/sd if sd>0 else 0.0
        g3 = float(stats.skew(d)); g4 = float(stats.kurtosis(d, fisher=False))
        v_an = max((1 - g3*sr + (g4-1)/4*sr**2)/(len(d)-1), 1e-12)
        r["v_analytic"] = v_an
        r["dsr_analytic"] = L.deflated_sharpe(d, N, v_an)
        r["dsr_search"]   = L.deflated_sharpe(d, N, V_SEARCH)
        r["trials_to_break_analytic"] = L.trials_to_break(d, v_an)
        r["trials_to_break_search"]   = L.trials_to_break(d, V_SEARCH)
        r["dsr_curve"] = {str(n): L.deflated_sharpe(d, n, v_an)["dsr"]
                          for n in (1,3,6,8,10,12,15,50,100,371,1000)}
        r["robust"] = L.drop_best(d, tp)
        # baselines, sizing, market-vs-skill, MC nulls at this rung
        bl = L.baselines(t, bg, w, rung=rung)
        mv = bl.pop("_prod_move"); leg = bl.pop("_leg")
        r["baselines"] = {n:{"pnl":float(v.sum()),"sharpe":L.sharpe(v)} for n,v in bl.items()}
        tt = t.copy(); tt["pnl"] = tp
        r["sizing"] = L.sizing_stripped(tt, w, mv)
        r["market_vs_skill"] = L.market_vs_skill(tt, bg, mv, w)
        obs_sh = rec["ladder"][rung]["sharpe"]; obs_pnl = rec["ladder"][rung]["pnl"]
        shA, totA = L.mc_null_a(tt, leg, w)
        r["mc_a"] = L.mc_summary(obs_sh, obs_pnl, shA, totA)
        shB, totB = L.mc_null_b(tt, bg, w, rung=rung)
        r["mc_b"] = L.mc_summary(obs_sh, obs_pnl, shB, totB)
        hA = np.histogram(shA, bins=50); hB = np.histogram(shB, bins=50)
        r["mc_hist"] = {"a":hA[0].tolist(),"a_edges":[round(v,4) for v in hA[1]],
                        "b":hB[0].tolist(),"b_edges":[round(v,4) for v in hB[1]]}
        rec["battery"][rung] = r
        print(f"  {k:13s} {rung:16s} {time.time()-t0:.0f}s  mcA={r['mc_a']['sharpe_pctile']:.2f} mcB={r['mc_b']['sharpe_pctile']:.2f}", flush=True)
    rec["equity"] = {rr: np.round(np.cumsum(dailies[(k,rr)]),1).tolist()
                     for rr in ["mid","passive_in_fee","cross_fee"]}
    rec["dates"] = [str(x.date()) for x in pd.date_range(w[0], w[1], freq="D")]

json.dump({"v_search":V_SEARCH,"fee":E.FEE,"rungs":RUNGS+["walk_fee"],
           "battery_rungs":BATTERY,"subs":store},
          open("results.json","w"), default=str)
print(f"done {time.time()-t0:.0f}s  -> results.json")
