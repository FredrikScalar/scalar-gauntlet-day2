"""The Luck battery: friction ladder, bootstrap, deflated Sharpe, robustness,
baselines, sizing-stripped, market-vs-skill, and two Monte Carlo nulls."""
from __future__ import annotations
import json, numpy as np, pandas as pd
from scipy import stats, sparse
import engine as E, trips as T

RNG = np.random.default_rng(20260903)
B_BOOT = 10000
B_MC   = 10000
ANN    = 365.0
EULER  = 0.5772156649015329

# ---------- metrics -------------------------------------------------------
def sharpe(x):
    s = np.std(x, ddof=1)
    return float(np.mean(x)/s*np.sqrt(ANN)) if s > 0 else 0.0

def sortino_claimed(x):
    """The submitters' definition: sample std of negative days only."""
    neg = x[x < 0]
    if neg.size < 2: return np.nan
    s = np.std(neg, ddof=1)
    return float(np.mean(x)/s*np.sqrt(ANN)) if s > 0 else np.nan

def sortino_strict(x):
    """RMS of negative days over all negative observations (no zero-day credit)."""
    neg = x[x < 0]
    if neg.size < 1: return np.nan
    dd = np.sqrt((neg**2).mean())
    return float(np.mean(x)/dd*np.sqrt(ANN)) if dd > 0 else np.nan

def max_dd(x):
    c = np.cumsum(x); return float((c - np.maximum.accumulate(c)).min())

def newey_west_t(x, lags=None):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 3: return np.nan
    if lags is None: lags = int(np.floor(4*(n/100)**(2/9)))
    e = x - m; g0 = (e@e)/n; v = g0
    for l in range(1, max(lags,0)+1):
        gl = (e[l:]@e[:-l])/n
        v += 2*(1 - l/(lags+1))*gl
    se = np.sqrt(max(v,1e-300)/n)
    return float(m/se)

def ar1(x):
    x = np.asarray(x,float)
    if len(x) < 3 or x.std()==0: return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0,1])

# ---------- bootstrap -----------------------------------------------------
def _boot_idx_iid(n, B, rng): return rng.integers(0, n, size=(B, n))

def _boot_idx_block(n, B, rng, L):
    nb = int(np.ceil(n/L))
    starts = rng.integers(0, n, size=(B, nb))
    off = np.arange(L)
    idx = (starts[:,:,None] + off[None,None,:]) % n
    return idx.reshape(B, -1)[:, :n]

def bootstrap(x, B=B_BOOT, rng=RNG):
    x = np.asarray(x, float); n = len(x)
    L = max(2, int(round(n**(1/3))))
    out = {}
    for tag, idx in (("iid", _boot_idx_iid(n,B,rng)), ("block", _boot_idx_block(n,B,rng,L))):
        s = x[idx]
        mu = s.mean(axis=1); sd = s.std(axis=1, ddof=1)
        sh = np.where(sd>0, mu/np.where(sd>0,sd,1)*np.sqrt(ANN), 0.0)
        so = np.array([sortino_claimed(r) for r in s]) if tag=="iid" else None
        d = {"sharpe_mean": float(np.mean(sh)), "sharpe_se": float(np.std(sh, ddof=1)),
             "sharpe_ci": [float(np.percentile(sh,2.5)), float(np.percentile(sh,97.5))],
             "sharpe_p_le0": float((sh<=0).mean())}
        d["sharpe_t"] = float(sharpe(x)/d["sharpe_se"]) if d["sharpe_se"]>0 else np.nan
        if so is not None:
            so = so[np.isfinite(so)]
            if so.size > 10:
                d |= {"sortino_mean": float(so.mean()), "sortino_se": float(so.std(ddof=1)),
                      "sortino_ci": [float(np.percentile(so,2.5)), float(np.percentile(so,97.5))],
                      "sortino_p_le0": float((so<=0).mean())}
                d["sortino_t"] = float(sortino_claimed(x)/d["sortino_se"]) if d["sortino_se"]>0 else np.nan
        out[tag] = d
    out["block_len"] = L
    return out

# ---------- deflated Sharpe ----------------------------------------------
def expected_max_sr(N, v_sr):
    """E[max SR] over N independent trials whose SRs have variance v_sr."""
    if N <= 1: return 0.0
    sd = np.sqrt(max(v_sr, 0.0))
    z1 = stats.norm.ppf(1 - 1.0/N)
    z2 = stats.norm.ppf(1 - 1.0/(N*np.e))
    return float(sd*((1-EULER)*z1 + EULER*z2))

def deflated_sharpe(x, N, v_sr):
    """Bailey & Lopez de Prado DSR. x = per-period P&L, N = trials."""
    x = np.asarray(x, float); Tn = len(x)
    sd = x.std(ddof=1)
    if sd == 0 or Tn < 3: return dict(dsr=np.nan)
    sr = x.mean()/sd                                    # per-period SR
    g3 = float(stats.skew(x)); g4 = float(stats.kurtosis(x, fisher=False))
    sr0 = expected_max_sr(N, v_sr)
    denom = np.sqrt(max(1 - g3*sr + (g4-1)/4*sr**2, 1e-12))
    z = (sr - sr0)*np.sqrt(Tn-1)/denom
    return dict(sr_period=float(sr), sr_ann=float(sr*np.sqrt(ANN)), skew=g3, kurt=g4,
                sr0_period=sr0, sr0_ann=float(sr0*np.sqrt(ANN)),
                dsr=float(stats.norm.cdf(z)), z=float(z), T=int(Tn), N=int(N),
                v_sr=float(v_sr))

def trials_to_break(x, v_sr, thresh=0.95, nmax=200000):
    """Smallest declared-trial count N at which DSR falls below thresh."""
    lo, hi = 1, nmax
    if deflated_sharpe(x, nmax, v_sr).get("dsr", 0) >= thresh: return None
    if deflated_sharpe(x, 1, v_sr).get("dsr", 0) < thresh: return 1
    while lo < hi-1:
        mid = (lo+hi)//2
        if deflated_sharpe(x, mid, v_sr)["dsr"] >= thresh: lo = mid
        else: hi = mid
    return hi

# ---------- robustness ----------------------------------------------------
def drop_best(daily, trips_pnl):
    d = np.sort(np.asarray(daily,float))[::-1]; tot = d.sum()
    res = {"total": float(tot), "n_days": len(d)}
    for k in (1,3,5,10):
        if len(d) > k:
            kept = np.sort(np.asarray(daily,float))[::-1][k:]
            res[f"drop{k}_pnl"] = float(kept.sum())
            res[f"drop{k}_sharpe"] = sharpe(kept)
            res[f"drop{k}_share_lost"] = float(1 - kept.sum()/tot) if tot else np.nan
    res["top5_share"] = float(d[:5].sum()/tot) if tot else np.nan
    res["top1day_share"] = float(d[0]/tot) if tot else np.nan
    cum = np.cumsum(d); res["days_to_zero"] = int(np.searchsorted(cum, tot)+1) if tot>0 else None
    n = 0; running = tot
    while n < len(d) and running > 0: running -= d[n]; n += 1
    res["days_to_zero"] = n
    tp = np.sort(np.asarray(trips_pnl,float))[::-1]; tt = tp.sum()
    for q in (0.001, 0.01, 0.05):
        k = max(1,int(round(q*len(tp))))
        res[f"droptrips_{q}_pnl"] = float(tt - tp[:k].sum())
        res[f"droptrips_{q}_share_lost"] = float(tp[:k].sum()/tt) if tt else np.nan
    res["n_trips"] = int(len(tp)); res["trips_pnl"] = float(tt)
    res["pos_trip_frac"] = float((tp>0).mean())
    return res

# ---------- book as (product, snapshot) grids ----------------------------
class BookGrid:
    """book_snapshots as dense (n_products, 40) grids + ts lookup."""
    def __init__(self, market):
        bk = market["book"].sort_values(["product_id","ts"], kind="stable")
        n = bk.product_id.nunique(); k = len(bk)//n
        assert len(bk) == n*k, "ragged book"
        self.k = k; self.pids = np.sort(bk.product_id.unique())
        self.mid = bk["mid"].values.reshape(n,k)
        self.bid = bk["bid_px_1"].values.reshape(n,k)
        self.ask = bk["ask_px_1"].values.reshape(n,k)
        self.ts  = bk["ts"].values.reshape(n,k)
        # map exec_ts -> snapshot index (per product the offsets are identical)
        self.slot = {}
        for pid, row in zip(self.pids, self.ts):
            pass
        # global ts -> index within product, built from the first product's offsets
        self.ts_flat = bk["ts"].values
        self.pid_flat = bk["product_id"].values
        idx = np.tile(np.arange(k), n)
        self.lookup = pd.Series(idx, index=pd.MultiIndex.from_arrays(
            [self.pid_flat, self.ts_flat]))
        self.prod_row = {p:i for i,p in enumerate(self.pids)}
        # market index: cross-sectional mean mid across products live at each ts
        s = pd.Series(bk["mid"].values, index=self.ts_flat)
        self.mkt = s.groupby(level=0).mean()
        # true delivery day per product row (NOT the snapshot day: the last
        # snapshot of an 00:00 product falls on the previous calendar day)
        pr = market["products"].set_index("product_id").loc[self.pids]
        self.deliv_day = _naive(pr["delivery_start"].values)

    def slot_of(self, pid, ts):
        return self.lookup.loc[list(zip(pid, ts))].values

    def rows_of(self, pid):
        return np.array([self.prod_row[p] for p in pid])

    def leg_prices(self, rows, e, x, rung):
        """(P&L per MW long, P&L per MW short) for entering at e, exiting at x.

        Rungs:
          mid             both legs at the price nobody quoted, no fee
          mid_fee         both legs at mid, 12c/MWh each side
          passive_in_fee  ENTRY rests at mid (order filled without crossing),
                          EXIT crosses -- half the spread, both fees
          passive_out_fee ENTRY crosses, EXIT rests at mid -- half the spread
          cross           both legs cross (buy ask / sell bid), no fee
          cross_fee       both legs cross, both fees  <- the honest rung
        A long buys and later sells; a short sells and later buys. Both pay, so
        a coin-flip null cannot have mean zero at any rung that charges friction.
        """
        f = E.FEE if rung.endswith("fee") else 0.0
        mE, mX = self.mid[rows, e], self.mid[rows, x]
        bE, aE = self.bid[rows, e], self.ask[rows, e]
        bX, aX = self.bid[rows, x], self.ask[rows, x]
        if rung in ("mid", "mid_fee"):
            buyE, sellE, buyX, sellX = mE, mE, mX, mX
        elif rung == "passive_in_fee":
            buyE, sellE, buyX, sellX = mE, mE, aX, bX
        elif rung == "passive_out_fee":
            buyE, sellE, buyX, sellX = aE, bE, mX, mX
        elif rung in ("cross", "cross_fee"):
            buyE, sellE, buyX, sellX = aE, bE, aX, bX
        else:
            raise ValueError(f"unknown rung {rung}")
        long_pnl  = (sellX - f) - (buyE + f)
        short_pnl = (sellE - f) - (buyX + f)
        return long_pnl, short_pnl


def trip_pnl(t, bg, rung):
    """Per-trip P&L at any rung, using each trip's ACTUAL recorded direction."""
    r = bg.rows_of(t.product_id.values)
    e = bg.slot_of(t.product_id.values, t.entry_ts.values)
    x = bg.slot_of(t.product_id.values, t.exit_ts.values)
    lp, sp = bg.leg_prices(r, e, x, rung)
    return t.qty.values*np.where(t.dir.values > 0, lp, sp)


def daily_from_trips(t, pnl, window):
    days = _naive(pd.date_range(window[0], window[1], freq="D"))
    return pd.Series(pnl, index=_naive(t.delivery)).groupby(level=0).sum() \
             .reindex(days, fill_value=0.0).values


# ---------- baselines: straight long / short ----------------------------
def baselines(t: pd.DataFrame, bg: BookGrid, window, rung="cross_fee", qty=10.0):
    """Directional benchmarks that need no signal at all, priced at `rung`."""
    r = bg.rows_of(t.product_id.values)
    e = bg.slot_of(t.product_id.values, t.entry_ts.values)
    x = bg.slot_of(t.product_id.values, t.exit_ts.values)
    lp, sp = bg.leg_prices(r, e, x, rung)
    mv = bg.mid[r, x] - bg.mid[r, e]
    day = _naive(t.delivery); q = t.qty.values
    days_i = _naive(pd.date_range(window[0], window[1], freq="D"))
    def ser(p):
        return pd.Series(p, index=day).groupby(level=0).sum().reindex(days_i, fill_value=0.0).values
    out = {}
    out["strategy"]     = ser(t.pnl.values)
    out["always_long"]  = ser(q*lp)     # same trades and timing, always long
    out["always_short"] = ser(q*sp)     # same trades and timing, always short
    # market buy-and-hold: qty MW long every product in the window, first -> last snapshot
    allr = np.arange(len(bg.pids))
    z = np.zeros(len(allr), int); last = np.full(len(allr), bg.k-1)
    blp, _ = bg.leg_prices(allr, z, last, rung)
    out["market_bh_long"] = pd.Series(qty*blp, index=bg.deliv_day) \
        .groupby(level=0).sum().reindex(days_i, fill_value=0.0).values
    out["_prod_move"] = mv
    out["_leg"] = (lp, sp, r, e, x)
    return out

# ---------- sizing stripped ---------------------------------------------
def sizing_stripped(t, window, mv):
    """(a) unit clip on every trip; (b) equal-weight days by per-MWh margin;
       (c) equal-weight trips."""
    day = _naive(t.delivery)
    days = _naive(pd.date_range(window[0], window[1], freq="D"))
    pnl = t.pnl.values
    unit = pnl/t.qty.values*10.0            # every clip forced to 10 MW
    def ser(p):
        return pd.Series(p, index=day).groupby(level=0).sum().reindex(days, fill_value=0.0).values
    d_act = ser(pnl); d_unit = ser(unit)
    vol = pd.Series(t.qty.values, index=day).groupby(level=0).sum().reindex(days, fill_value=0.0).values
    with np.errstate(invalid="ignore", divide="ignore"):
        margin = np.where(vol>0, d_act/np.where(vol>0,vol,1), 0.0)   # EUR per MWh per day
    per_trip = pnl/t.qty.values                                       # EUR per MWh per trip
    return {
        "actual_sharpe":  sharpe(d_act),
        "unitclip_sharpe": sharpe(d_unit),
        "unitclip_pnl": float(d_unit.sum()),
        "margin_sharpe": sharpe(margin),
        "margin_mean": float(margin[vol>0].mean()) if (vol>0).any() else np.nan,
        "trip_mean_eur_mwh": float(per_trip.mean()),
        "trip_t": newey_west_t(per_trip),
        "trip_sd_eur_mwh": float(per_trip.std(ddof=1)),
        "n_trips": int(len(per_trip)),
    }

# ---------- market vs skill ---------------------------------------------
def market_vs_skill(t, bg, mv, window):
    """Split each trip's move into the market's move over the same clock
    interval and the product-specific residual."""
    pairs = pd.DataFrame({"e": t.entry_ts.values, "x": t.exit_ts.values})
    uniq = pairs.drop_duplicates()
    m = bg.mkt
    dm = (m.reindex(pd.DatetimeIndex(uniq.x)).values -
          m.reindex(pd.DatetimeIndex(uniq.e)).values)
    key = pd.Series(dm, index=pd.MultiIndex.from_arrays([uniq.e, uniq.x]))
    mkt_move = key.loc[list(zip(pairs.e, pairs.x))].values
    d = t.dir.values; q = t.qty.values
    pnl   = d*q*mv
    beta  = d*q*mkt_move                # what the market handed them, given their side
    alpha = d*q*(mv - mkt_move)         # product selection residual
    day = _naive(t.delivery)
    days = _naive(pd.date_range(window[0], window[1], freq="D"))
    def ser(p): return pd.Series(p, index=day).groupby(level=0).sum().reindex(days, fill_value=0.0).values
    dp, db, da = ser(pnl), ser(beta), ser(alpha)
    # daily regression of strategy P&L on the market factor
    mf = pd.Series(m.values, index=pd.DatetimeIndex(m.index)).groupby(
            _naive(m.index)).agg(lambda s: s.iloc[-1]-s.iloc[0])
    mf = mf.reindex(days, fill_value=0.0).values
    ok = np.isfinite(dp) & np.isfinite(mf)
    if ok.sum() > 5 and np.std(mf[ok]) > 0:
        sl, ic, rv, pv, se = stats.linregress(mf[ok], dp[ok])
        reg = dict(beta=float(sl), alpha_per_day=float(ic), r2=float(rv**2), p=float(pv))
    else:
        reg = dict(beta=np.nan, alpha_per_day=np.nan, r2=np.nan, p=np.nan)
    tot = pnl.sum()
    return {"pnl": float(tot), "market_component": float(beta.sum()),
            "skill_component": float(alpha.sum()),
            "market_share": float(beta.sum()/tot) if tot else np.nan,
            "market_sharpe": sharpe(db), "skill_sharpe": sharpe(da),
            "regression": reg}

# ---------- Monte Carlo nulls -------------------------------------------
def _naive(x):
    i = pd.DatetimeIndex(x)
    return (i.tz_convert("UTC").tz_localize(None) if i.tz is not None else i).normalize()

def _day_matrix(day_index, days):
    """sparse (n_days x n_trips) incidence, so day-aggregation is a matmul."""
    pos = pd.Index(_naive(days)).get_indexer(_naive(day_index))
    assert (pos >= 0).all(), "trip outside declared window"
    n = len(day_index)
    return sparse.csr_matrix((np.ones(n), (pos, np.arange(n))), shape=(len(days), n)), pos

def _sharpe_cols(M):
    mu = M.mean(axis=0); sd = M.std(axis=0, ddof=1)
    return np.where(sd>0, mu/np.where(sd>0,sd,1)*np.sqrt(ANN), 0.0)

def mc_null_a(t, leg, window, B=B_MC, rng=RNG, chunk=500):
    """Same timestamps, same products, same clips -- side randomised per trip.
    Both sides pay the round-trip friction, so this null is not antisymmetric."""
    lp, sp = leg[0], leg[1]
    days = pd.date_range(window[0], window[1], freq="D")
    S, _ = _day_matrix(t.delivery.values, days)
    q = t.qty.values
    L_ = q*lp; S_ = q*sp
    sh, tot = [], []
    for i in range(0, B, chunk):
        c = min(chunk, B-i)
        take_long = rng.random((len(q), c)) < 0.5
        D = S @ np.where(take_long, L_[:,None], S_[:,None])
        sh.append(_sharpe_cols(D)); tot.append(D.sum(axis=0))
    return np.concatenate(sh), np.concatenate(tot)

def mc_null_b(t, bg, window, rung="cross_fee", B=B_MC, rng=RNG, chunk=250):
    """Same delivery days and same trip count per day, same holding periods --
    but product and entry snapshot drawn at random from that day, side random."""
    days = pd.date_range(window[0], window[1], freq="D")
    S, _ = _day_matrix(t.delivery.values, days)
    n = len(t)
    hold = np.rint(t.hold_min.values/15).astype(int)
    hold = np.clip(hold, 1, bg.k-1)
    # the 24 product rows belonging to each trip's delivery day
    dday = _naive(t.delivery)
    udays = pd.Index(np.sort(bg.deliv_day.unique()))
    order = np.lexsort((np.arange(len(bg.pids)), udays.get_indexer(bg.deliv_day)))
    per = np.bincount(udays.get_indexer(bg.deliv_day))
    assert per.min() == per.max(), f"ragged delivery days: {per.min()}..{per.max()}"
    day_tab = order.reshape(len(udays), per[0])             # (n_days, 24) product rows
    trip_day = udays.get_indexer(dday)
    assert (trip_day >= 0).all(), "trip delivery day not in the product table"
    q = t.qty.values
    sh, tot = [], []
    for i in range(0, B, chunk):
        c = min(chunk, B-i)
        pick = rng.integers(0, day_tab.shape[1], size=(n, c))
        rows = day_tab[trip_day[:,None], pick]
        emax = (bg.k - 1 - hold)[:,None]
        e = (rng.random((n,c))*(emax+1)).astype(int)
        x = e + hold[:,None]
        lp, sp = bg.leg_prices(rows.ravel(), e.ravel(), x.ravel(), rung)
        lp = lp.reshape(rows.shape); sp = sp.reshape(rows.shape)
        take_long = rng.random((n, c)) < 0.5
        D = S @ (q[:,None]*np.where(take_long, lp, sp))
        sh.append(_sharpe_cols(D)); tot.append(D.sum(axis=0))
    return np.concatenate(sh), np.concatenate(tot)

def mc_summary(obs_sharpe, obs_pnl, sh, tot):
    return {
        "n": int(len(sh)),
        "sharpe_null_mean": float(sh.mean()), "sharpe_null_sd": float(sh.std(ddof=1)),
        "sharpe_null_p95": float(np.percentile(sh,95)),
        "sharpe_null_p99": float(np.percentile(sh,99)),
        "sharpe_null_max": float(sh.max()),
        "sharpe_pctile": float((sh < obs_sharpe).mean()*100),
        "sharpe_p_value": float((sh >= obs_sharpe).mean()),
        "sharpe_beats_best_run": bool(obs_sharpe > sh.max()),
        "sharpe_z": float((obs_sharpe - sh.mean())/sh.std(ddof=1)) if sh.std(ddof=1)>0 else np.nan,
        "pnl_null_max": float(tot.max()), "pnl_null_p95": float(np.percentile(tot,95)),
        "pnl_pctile": float((tot < obs_pnl).mean()*100),
        "pnl_p_value": float((tot >= obs_pnl).mean()),
    }
