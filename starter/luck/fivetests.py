"""Compute the five-tests record for ONE submission (or several), from raw data.

Self-contained: nothing here reads a previous payload. Round trips are recovered
from the blotter by FIFO matching (trips.py) and repriced against
book_snapshots.csv (luck.BookGrid), exactly as run_full.py does.

Public API
----------
    ctx = Context()                       # loads market + registry + BookGrid once
    rec = compute(ctx, "windfall")        # everything the report needs for one sub
    ctx.cache_write("windfall", rec)      # cache/windfall.json

CLI
---
    python3 fivetests.py windfall [windfall2 ...]      compute and cache
    python3 fivetests.py all                           compute and cache all seven
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import engine as E, trips as T, luck as L

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"


def repo_root() -> Path:
    """The directory that holds data/ and blotters/, wherever this file sits.

    engine.py hard-codes its root two levels up from itself; that breaks if the
    luck/ folder is nested differently from the checkout it was written in, so
    resolve it by looking for the data and repoint engine at what we find.
    """
    for d in [HERE, *HERE.parents]:
        if (d/"data"/"book_snapshots.csv").exists() and (d/"blotters").is_dir():
            return d
    raise FileNotFoundError(
        "cannot find data/book_snapshots.csv above " + str(HERE) +
        " — run the pipeline from inside the repo checkout")


ROOT = repo_root()
if E.D != ROOT:
    E.D = ROOT          # engine.load_market / load_blotter / load_registry follow this
RUNGS = ["mid", "passive_in_fee", "cross_fee"]
QTY = 10.0
SEEDS = (20260903, 11, 2718, 31337, 987654)
DSR_TRIALS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100,
              150, 200, 300, 371, 500, 750, 1000, 2000, 5000, 10000]
SCHEMA = 4          # bump when the record shape changes so caches invalidate


# ---------------------------------------------------------------- context
class Context:
    """Market, registry and book grid, loaded once and shared across strategies."""

    def __init__(self, b_boot=10_000, b_mc=10_000, seeds=SEEDS, quiet=False):
        t0 = time.time()
        self.b_boot, self.b_mc, self.seeds, self.quiet = b_boot, b_mc, tuple(seeds), quiet
        self.mk = E.load_market()
        self.reg = E.load_registry()
        self.bg = L.BookGrid(self.mk)
        self._trips = {}
        self._mid_daily = {}
        self._v_search = None
        # market term structure: what the book itself hands anyone holding long
        d = np.diff(self.bg.mid, axis=1)
        self.drift_per_step = float(d.mean())
        self.drift_full = float((self.bg.mid[:, -1] - self.bg.mid[:, 0]).mean())
        spr = self.bg.ask - self.bg.bid
        self.spread_by_slot = [float(x) for x in spr.mean(axis=0)]
        self.mid_by_slot = [float(x) for x in (self.bg.mid - self.bg.mid[:, [0]]).mean(axis=0)]
        self.log(f"market loaded in {time.time()-t0:.0f}s "
                 f"({len(self.bg.pids):,} products x {self.bg.k} snapshots)")

    def log(self, msg):
        if not self.quiet:
            print(msg, flush=True)

    # ---- per-strategy primitives, memoised
    def window(self, key):
        w = self.reg[key]["backtest_window"]
        return (str(w["start"]), str(w["end"]))

    def trips(self, key):
        if key not in self._trips:
            b = E.reprice(E.load_blotter(key), self.mk)
            self._trips[key] = (b, T.round_trips(b, "mid"))
        return self._trips[key]

    def mid_daily(self, key):
        """Daily P&L at mid — needed for every strategy to set the
        cross-sectional trial variance used by the harsh deflated Sharpe."""
        if key not in self._mid_daily:
            _b, t = self.trips(key)
            self._mid_daily[key] = L.daily_from_trips(
                t, L.trip_pnl(t, self.bg, "mid"), self.window(key))
        return self._mid_daily[key]

    def v_search(self):
        """Var of daily Sharpe across the seven submissions (the harsh
        deflated-Sharpe assumption). Needs all seven, so it is computed once."""
        if self._v_search is None:
            srs = []
            for k in E.SUBS:
                d = self.mid_daily(k)
                srs.append(d.mean()/d.std(ddof=1))
            self._v_search = float(np.var(srs, ddof=1))
            self.log(f"cross-sectional trial variance v_search = {self._v_search:.6f}")
        return self._v_search

    # ---- cache
    def cache_path(self, key):
        return CACHE / f"{key}.json"

    def cache_write(self, key, rec):
        CACHE.mkdir(exist_ok=True)
        self.cache_path(key).write_text(json.dumps(rec, separators=(",", ":")))
        return self.cache_path(key)


def cache_read(key):
    p = CACHE / f"{key}.json"
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text())
    except Exception:
        return None
    return rec if rec.get("schema") == SCHEMA else None


def cache_stale(key):
    """True when the cached record predates the data or code it came from."""
    p = CACHE / f"{key}.json"
    if not p.exists():
        return True
    srcs = [ROOT/"data"/"book_snapshots.csv", ROOT/"data"/"products.csv",
            ROOT/"blotters"/f"{key}-blotter.csv", ROOT/"registry"/f"{key}.yaml",
            HERE/"fivetests.py", HERE/"luck.py", HERE/"engine.py", HERE/"trips.py",
            HERE/"verdicts.py"]
    mt = p.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > mt for s in srcs)


# ---------------------------------------------------------------- metrics
ANN = L.ANN


def _sharpe(x):
    return L.sharpe(np.asarray(x, float))


def _sortino(x):
    v = L.sortino_claimed(np.asarray(x, float))
    return None if v is None or not np.isfinite(v) else float(v)


def decay_curve(daily):
    """Delete the best day, the two best, ... until the P&L crosses zero.

    Returns the full curve plus the breakeven day count. Fractions are against
    the untruncated value so every series starts at exactly 1.0.
    """
    d = np.asarray(daily, float)
    order = np.argsort(d)[::-1]
    tot0, sh0, so0 = d.sum(), _sharpe(d), _sortino(d)
    curve, z = [], None
    for k in range(0, max(len(d) - 3, 1)):
        kept = d[order[k:]]
        tot, sh, so = kept.sum(), _sharpe(kept), _sortino(kept)
        curve.append([k,
                      None if not tot0 else round(float(tot/tot0), 6),
                      None if not sh0 else round(sh/sh0, 6),
                      None if (so is None or not so0) else round(so/so0, 6)])
        if tot <= 0:
            z = k
            break
    return curve, z, dict(pnl=float(tot0), sharpe=sh0, sortino=so0)


def thin(curve):
    """Every k up to 40, then every 4th, always keeping the last point."""
    return [p for i, p in enumerate(curve)
            if p[0] <= 40 or p[0] % 4 == 0 or i == len(curve) - 1]


# ---------------------------------------------------------------- compute
def compute(ctx: Context, key: str) -> dict:
    """Everything the report needs for one submission, at all three rungs."""
    t0 = time.time()
    e = ctx.reg[key]
    w = ctx.window(key)
    b, t = ctx.trips(key)
    bg = ctx.bg
    N = e["declared_trials"]
    v_search = ctx.v_search()

    r_ = bg.rows_of(t.product_id.values)
    e_ = bg.slot_of(t.product_id.values, t.entry_ts.values)
    x_ = bg.slot_of(t.product_id.values, t.exit_ts.values)
    day = L._naive(t.delivery)
    days_i = L._naive(pd.date_range(w[0], w[1], freq="D"))
    q = t.qty.values
    allr = np.arange(len(bg.pids))
    LAST = bg.k - 1

    def ser(p, idx):
        return (pd.Series(p, index=idx).groupby(level=0).sum()
                .reindex(days_i, fill_value=0.0).values)

    # matched-horizon passive: the strategy's own median entry snapshot and
    # median hold, applied to 10 MW long on EVERY product in the window
    entry_slot = int(np.median(e_))
    hold_steps = int(round(np.median(x_ - e_)))
    ms_e = np.full(len(allr), entry_slot)
    ms_x = np.full(len(allr), min(entry_slot + hold_steps, LAST))

    # market term structure over the horizon this strategy actually holds
    spr = bg.ask - bg.bid
    drift_avail = float((bg.mid[r_, x_] - bg.mid[r_, e_]).mean())
    friction = float(((spr[r_, e_] + spr[r_, x_])/2).mean() + 2*E.FEE)
    edge_mid = float((t.dir.values*(bg.mid[r_, x_] - bg.mid[r_, e_])).mean())
    passive_edge_mid = float((bg.mid[:, ms_x[0]] - bg.mid[:, entry_slot]).mean())
    n_prod = int(t.product_id.nunique())
    days_n = int(e["backtest_window"]["days"])

    rec = {
        "schema": SCHEMA, "key": key, "name": e["name"], "class": e["class"],
        "declared": {"params": e.get("params", {}),
                     "declared_trials": N,
                     "dev_window": (None if e.get("dev_window") in (None, "None")
                                    else str(e.get("dev_window"))),
                     "position_cap_mw": e.get("position_cap_mw"),
                     "data_dependencies": list(e.get("data_dependencies") or []),
                     "window_days": days_n},
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "fee": E.FEE, "ann": ANN, "rungs_available": RUNGS,
        "window": list(w), "days": days_n, "declared_trials": N,
        "dev_window": str(e.get("dev_window")), "params": e.get("params", {}),
        "claimed": e["claimed"], "b_boot": ctx.b_boot, "b_mc": ctx.b_mc,
        "n_trades": int(len(b)), "n_trips": int(len(t)),
        "vol_mwh": float(b.qty_mw.sum()),
        "long_frac": round(float((t.dir > 0).mean()), 4),
        "hold_med": float(t.hold_min.median()),
        "entry_slot": entry_slot, "hold_steps": hold_steps,
        "half_spread": float(b.half_spread.mean()),
        "v_search": v_search,
        "drift_per_step": ctx.drift_per_step, "drift_full": ctx.drift_full,
        "spread_by_slot": ctx.spread_by_slot, "mid_by_slot": ctx.mid_by_slot,
        "prods_day": round(n_prod/days_n, 4),
        "breadth": round(24.0/(n_prod/days_n), 3),
        "edge_mid": round(edge_mid, 4),
        "passive_edge_mid": round(passive_edge_mid, 4),
        "edge_ratio_mid": round(edge_mid/passive_edge_mid, 4) if passive_edge_mid else None,
        "be_hold": {"entry": entry_slot, "hold_min": float(t.hold_min.median()),
                    "drift": round(drift_avail, 4), "friction": round(friction, 4),
                    "covers": bool(drift_avail >= friction)},
        "rungs": {}, "equity": {}, "verdicts": {}, "why": {},
    }

    # ---- the battery, per rung
    for rung in RUNGS:
        tp = L.trip_pnl(t, bg, rung)
        d = L.daily_from_trips(t, tp, w)
        vol = float(b.qty_mw.sum())
        curve, z, base = decay_curve(d)
        tps = np.sort(np.asarray(tp, float))[::-1]
        tt = tps.sum()
        trips_to_zero = int(np.searchsorted(np.cumsum(tps), tt) + 1) if tt > 0 else None
        srt = np.sort(d)[::-1]

        # bootstrap
        bt = L.bootstrap(d, B=ctx.b_boot)
        boot = {"ci": [round(x, 4) for x in bt["iid"]["sharpe_ci"]],
                "blk_ci": [round(x, 4) for x in bt["block"]["sharpe_ci"]],
                "se": round(bt["iid"]["sharpe_se"], 4),
                "t": round(bt["iid"]["sharpe_t"], 3),
                "blk_t": round(bt["block"]["sharpe_t"], 3),
                "blk_len": bt["block_len"],
                "p_le0": bt["iid"]["sharpe_p_le0"],
                "so_ci": [round(x, 3) for x in bt["iid"].get("sortino_ci", [float("nan")]*2)],
                "so_t": (round(bt["iid"]["sortino_t"], 3) if "sortino_t" in bt["iid"] else None),
                "nw_t": round(L.newey_west_t(d), 3), "ar1": round(L.ar1(d), 4)}

        # deflated Sharpe, both variance assumptions
        sd = d.std(ddof=1)
        sr = d.mean()/sd if sd > 0 else 0.0
        g3 = float(stats.skew(d)); g4 = float(stats.kurtosis(d, fisher=False))
        v_an = max((1 - g3*sr + (g4-1)/4*sr**2)/(len(d)-1), 1e-12)
        da = L.deflated_sharpe(d, N, v_an)
        ds = L.deflated_sharpe(d, N, v_search)
        dsr = {"sr_ann": round(float(sr*np.sqrt(ANN)), 4),
               "skew": round(g3, 3), "kurt": round(g4, 3),
               "declared": N, "v_analytic": v_an, "v_search": v_search,
               "at_declared_analytic": round(da["dsr"], 6),
               "at_declared_search": round(ds["dsr"], 6),
               "sr0_ann_analytic": round(da["sr0_ann"], 4),
               "sr0_ann_search": round(ds["sr0_ann"], 4),
               "break_analytic": L.trials_to_break(d, v_an),
               "break_search": L.trials_to_break(d, v_search),
               "curve_analytic": [[n, round(L.deflated_sharpe(d, n, v_an)["dsr"], 6)]
                                  for n in DSR_TRIALS],
               "curve_search": [[n, round(L.deflated_sharpe(d, n, v_search)["dsr"], 6)]
                                for n in DSR_TRIALS]}

        # baselines, priced on BOTH sides at this rung
        lp, sp = bg.leg_prices(r_, e_, x_, rung)
        blp, _ = bg.leg_prices(allr, np.zeros(len(allr), int),
                               np.full(len(allr), LAST), rung)
        mlp, _ = bg.leg_prices(allr, ms_e, ms_x, rung)
        bser = {"strategy": ser(tp, day), "always_long": ser(q*lp, day),
                "always_short": ser(q*sp, day),
                "market_bh_full": ser(QTY*blp, bg.deliv_day),
                "matched_passive": ser(QTY*mlp, bg.deliv_day)}
        base_out = {n: {"pnl": round(float(v.sum())), "sharpe": round(L.sharpe(v), 4)}
                    for n, v in bser.items()}

        # sizing stripped, market vs skill
        tt_ = t.copy(); tt_["pnl"] = tp
        mv = bg.mid[r_, x_] - bg.mid[r_, e_]
        sz = L.sizing_stripped(tt_, w, mv)
        mvs = L.market_vs_skill(tt_, bg, mv, w)

        # Monte Carlo nulls
        rng = np.random.default_rng(ctx.seeds[0])
        shA, totA = L.mc_null_a(tt_, (lp, sp, r_, e_, x_), w, B=ctx.b_mc, rng=rng)
        shB, totB = L.mc_null_b(tt_, bg, w, rung=rung, B=ctx.b_mc, rng=rng)
        obs_sh, obs_pnl = L.sharpe(d), float(d.sum())

        def mc(sh, tot):
            m = L.mc_summary(obs_sh, obs_pnl, sh, tot)
            m["q99"] = float(np.quantile(sh, 0.99))
            m["q999"] = float(np.quantile(sh, 0.999))
            m["above_q999"] = bool(obs_sh > m["q999"])
            return m
        mc_a, mc_b = mc(shA, totA), mc(shB, totB)
        cA, eA = np.histogram(shA, bins=50)
        cB, eB = np.histogram(shB, bins=50)

        rec["rungs"][rung] = {
            "pnl": round(float(d.sum())), "sharpe": round(L.sharpe(d), 4),
            "sortino": _sortino(d) and round(_sortino(d), 4),
            "sortino_strict": round(float(L.sortino_strict(d)), 4),
            "max_dd": round(L.max_dd(d)), "margin": round(float(d.sum()/vol), 5),
            "active_days": int((d != 0).sum()),
            "hit": round(float((d[d != 0] > 0).mean()), 4) if (d != 0).any() else None,
            "n_trips": int(len(tp)),
            "days_to_zero": z, "trips_to_zero": trips_to_zero,
            "top1_share": round(float(srt[0]/d.sum()), 4) if d.sum() else None,
            "top5_share": round(float(srt[:5].sum()/d.sum()), 4) if d.sum() else None,
            "pos_trip_frac": round(float((tp > 0).mean()), 4),
            "decay": thin(curve),
            "boot": boot, "dsr": dsr, "base": base_out,
            "edge": round(float(tp.sum()/q.sum()), 4),
            "matched_edge": round(float(mlp.mean()), 4),
            "sizing": {"unitclip_sharpe": round(sz["unitclip_sharpe"], 4),
                       "unitclip_pnl": round(sz["unitclip_pnl"]),
                       "margin_sharpe": round(sz["margin_sharpe"], 4),
                       "trip_mean": round(sz["trip_mean_eur_mwh"], 4),
                       "trip_t": round(sz["trip_t"], 3),
                       "trip_sd": round(sz["trip_sd_eur_mwh"], 4)},
            "mvs": {"market_share": round(mvs["market_share"], 4),
                    "market_sharpe": round(mvs["market_sharpe"], 4),
                    "skill_sharpe": round(mvs["skill_sharpe"], 4),
                    "market_component": round(mvs["market_component"]),
                    "skill_component": round(mvs["skill_component"])},
            "mc_a": mc_a, "mc_b": mc_b,
            "hist_a": {"c": [int(x) for x in cA], "e": [round(float(x), 3) for x in eA]},
            "hist_b": {"c": [int(x) for x in cB], "e": [round(float(x), 3) for x in eB]},
        }
        rec["equity"][rung] = [int(round(x)) for x in np.cumsum(d)]
        ctx.log(f"  {key:13s} {rung:15s} pnl {d.sum():>10,.0f}  Sharpe {L.sharpe(d):>6.2f}"
                f"  breakeven days {z}   {time.time()-t0:.0f}s")

    # ---- seed stability of the coin-flip verdict, at mid
    tp_mid = L.trip_pnl(t, bg, "mid")
    tt_mid = t.copy(); tt_mid["pnl"] = tp_mid
    lp0, sp0 = bg.leg_prices(r_, e_, x_, "mid")
    d_mid = L.daily_from_trips(t, tp_mid, w)
    obs = L.sharpe(d_mid)
    mx, q999, pv = [], [], []
    for s in ctx.seeds:
        sh, _tot = L.mc_null_a(tt_mid, (lp0, sp0, r_, e_, x_), w,
                               B=ctx.b_mc, rng=np.random.default_rng(s))
        mx.append(float(sh.max()))
        q999.append(float(np.quantile(sh, 0.999)))
        pv.append(float((sh >= obs).mean()))
    beats = [obs > m for m in mx]
    above = [obs > q for q in q999]
    rec["seed_stability"] = {
        "seeds": len(ctx.seeds),
        "max_lo": round(min(mx), 3), "max_hi": round(max(mx), 3),
        "max_range": round(max(mx)-min(mx), 3),
        "beats_max": "always" if all(beats) else ("never" if not any(beats) else "FLIPS"),
        "q999_range": round(max(q999)-min(q999), 3),
        "above_q999": "yes" if all(above) else ("no" if not any(above) else "FLIPS"),
        "p_lo": round(min(pv), 5), "p_hi": round(max(pv), 5),
    }

    # ---- verdicts
    import verdicts as VD
    for rung in RUNGS:
        v, why = VD.score(rec, rung)
        rec["verdicts"][rung] = v
        rec["why"][rung] = why
    rec["dates"] = [str(x.date()) for x in days_i]
    ctx.log(f"  {key:13s} verdict on cross_fee: {rec['verdicts']['cross_fee']['overall']}"
            f"   ({time.time()-t0:.0f}s)")
    return rec


def load_or_compute(ctx: Context, key: str, refresh=False) -> dict:
    """Cached record for one strategy, recomputed when stale or asked."""
    if not refresh and not cache_stale(key):
        rec = cache_read(key)
        if rec is not None:
            ctx.log(f"  {key:13s} from cache")
            return rec
    rec = compute(ctx, key)
    ctx.cache_write(key, rec)
    return rec


def main(argv):
    keys = argv[1:] or ["all"]
    if keys == ["all"]:
        keys = list(E.SUBS)
    bad = [k for k in keys if k not in E.SUBS]
    if bad:
        sys.exit(f"unknown strategy {bad}; choose from {', '.join(E.SUBS)} or 'all'")
    ctx = Context()
    for k in keys:
        p = ctx.cache_write(k, compute(ctx, k))
        print(f"wrote {p} ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv)
