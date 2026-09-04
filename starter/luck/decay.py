"""Exact daily P&L series + full drop-best-days decay curves for all 7 submissions.

Writes decay.json for the focused HTML report. Nothing here re-derives a claim:
it re-uses engine/trips/luck exactly as run_full.py does, then extends the
drop-best-days curve from the 4 points the first pass stored to every k up to
the point where cumulative P&L crosses zero.
"""
import json, time, numpy as np, pandas as pd
from scipy import stats
import engine as E, trips as T, luck as L

t0 = time.time()
mk = E.load_market(); reg = E.load_registry(); bg = L.BookGrid(mk)
print(f"market loaded {time.time()-t0:.0f}s", flush=True)

RUNGS = ["mid", "passive_in_fee", "cross_fee"]
ANN = 365.0

def sharpe(x):
    s = np.std(x, ddof=1)
    return float(np.mean(x)/s*np.sqrt(ANN)) if s > 0 else 0.0

def sortino_claimed(x):
    neg = x[x < 0]
    if neg.size < 2: return None
    s = np.std(neg, ddof=1)
    return float(np.mean(x)/s*np.sqrt(ANN)) if s > 0 else None

def decay_curve(daily):
    """Remove the k best days, k = 0,1,2,... until total P&L <= 0.

    Returns per-k: P&L kept, fraction of original P&L kept, Sharpe, Sortino,
    and the same three as a fraction of their k=0 value.
    """
    d = np.asarray(daily, float)
    order = np.argsort(d)[::-1]          # best day first
    tot0 = d.sum(); sh0 = sharpe(d); so0 = sortino_claimed(d)
    n = len(d)
    out = []
    kmax = n - 3
    for k in range(0, kmax + 1):
        kept = d[order[k:]]
        tot = kept.sum()
        sh = sharpe(kept); so = sortino_claimed(kept)
        out.append(dict(k=k,
                        pnl=round(float(tot), 2),
                        pnl_frac=round(float(tot/tot0), 6) if tot0 else None,
                        sharpe=round(sh, 4),
                        sharpe_frac=round(sh/sh0, 6) if sh0 else None,
                        sortino=None if so is None else round(so, 4),
                        sortino_frac=None if (so is None or not so0) else round(so/so0, 6)))
        if tot <= 0:
            break
    return out, dict(pnl=float(tot0), sharpe=sh0, sortino=so0)

res = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
       "ann": ANN, "fee": E.FEE, "subs": {}}

V_daily = {}
for k in E.SUBS:
    e = reg[k]
    w = (str(e["backtest_window"]["start"]), str(e["backtest_window"]["end"]))
    b = E.reprice(E.load_blotter(k), mk)
    t = T.round_trips(b, "mid")
    rec = {"name": e["name"], "declared_trials": e["declared_trials"],
           "n_days": None, "window": list(w), "rungs": {}}
    for rung in RUNGS:
        tp = L.trip_pnl(t, bg, rung)
        d = L.daily_from_trips(t, tp, w)
        V_daily[(k, rung)] = d
        curve, base = decay_curve(d)
        # trade/trip-level decay too: remove the best trips
        tps = np.sort(np.asarray(tp, float))[::-1]
        tt = tps.sum()
        cum = np.cumsum(tps)
        trips_to_zero = int(np.searchsorted(cum, tt) + 1) if tt > 0 else None
        act = int((d != 0).sum())
        rec["n_days"] = int(len(d))
        rec["rungs"][rung] = {
            "pnl": round(base["pnl"], 2),
            "sharpe": round(base["sharpe"], 4),
            "sortino": None if base["sortino"] is None else round(base["sortino"], 4),
            "active_days": act,
            "days_to_zero": curve[-1]["k"] if curve[-1]["pnl"] <= 0 else None,
            "n_trips": int(len(tp)),
            "trips_to_zero": trips_to_zero,
            "pos_trip_frac": round(float((tp > 0).mean()), 4),
            "top1_share": round(float(np.sort(d)[::-1][0]/base["pnl"]), 4) if base["pnl"] else None,
            "top5_share": round(float(np.sort(d)[::-1][:5].sum()/base["pnl"]), 4) if base["pnl"] else None,
            "curve": curve,
        }
    res["subs"][k] = rec
    print(f"{k:13s} done {time.time()-t0:.0f}s", flush=True)

# ---- deflated Sharpe: full continuous curve, both variance assumptions ----
V_SEARCH = float(np.var([V_daily[(k, "mid")].mean()/V_daily[(k, "mid")].std(ddof=1)
                         for k in E.SUBS], ddof=1))
res["v_search"] = V_SEARCH
TRIALS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100,
          150, 200, 300, 371, 500, 750, 1000, 2000, 5000, 10000]
for k in E.SUBS:
    for rung in RUNGS:
        d = V_daily[(k, rung)]
        sd = d.std(ddof=1); sr = d.mean()/sd if sd > 0 else 0.0
        g3 = float(stats.skew(d)); g4 = float(stats.kurtosis(d, fisher=False))
        v_an = max((1 - g3*sr + (g4-1)/4*sr**2)/(len(d)-1), 1e-12)
        N = res["subs"][k]["declared_trials"]
        blk = res["subs"][k]["rungs"][rung]
        blk["dsr"] = {
            "sr_ann": round(float(sr*np.sqrt(ANN)), 4),
            "skew": round(g3, 3), "kurt": round(g4, 3),
            "v_analytic": v_an, "v_search": V_SEARCH,
            "declared": N,
            "at_declared_analytic": round(L.deflated_sharpe(d, N, v_an)["dsr"], 6),
            "at_declared_search": round(L.deflated_sharpe(d, N, V_SEARCH)["dsr"], 6),
            "break_analytic": L.trials_to_break(d, v_an),
            "break_search": L.trials_to_break(d, V_SEARCH),
            "sr0_ann_analytic": round(L.deflated_sharpe(d, N, v_an)["sr0_ann"], 4),
            "sr0_ann_search": round(L.deflated_sharpe(d, N, V_SEARCH)["sr0_ann"], 4),
            "curve_analytic": [[n, round(L.deflated_sharpe(d, n, v_an)["dsr"], 6)] for n in TRIALS],
            "curve_search": [[n, round(L.deflated_sharpe(d, n, V_SEARCH)["dsr"], 6)] for n in TRIALS],
        }

# daily series themselves, rounded to the cent, for the equity/day charts
for k in E.SUBS:
    for rung in RUNGS:
        res["subs"][k]["rungs"][rung]["daily"] = [round(float(x), 2) for x in V_daily[(k, rung)]]
res["dates"] = [str(x.date()) for x in L._naive(pd.date_range(
    res["subs"]["windfall"]["window"][0], res["subs"]["windfall"]["window"][1], freq="D"))]

json.dump(res, open("decay.json", "w"))
print("wrote decay.json", round(len(json.dumps(res))/1e6, 2), "MB",
      f"{time.time()-t0:.0f}s")
