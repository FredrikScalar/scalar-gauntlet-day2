"""Assemble the focused five-tests report data + verdicts. Writes five_tests.json."""
import json, numpy as np

P = json.load(open("/home/claude/work/payload.json"))
D = json.load(open("decay.json"))
B = json.load(open("base2.json"))

RUNGS = ["mid", "passive_in_fee", "cross_fee"]
SUBS = ["windfall", "windfall2", "sunspot", "spikecatcher", "pingpong", "bounceback", "blackbox"]
BR = {r["sub"]: r for r in P["breadth_rows"]}
BE = {r["sub"]: r for r in P["breakeven_rows"]}

def thin(curve):
    """Every k up to 40, then every 4th, always keeping the last point."""
    out = []
    for i, p in enumerate(curve):
        if p["k"] <= 40 or p["k"] % 4 == 0 or i == len(curve)-1:
            out.append([p["k"], p["pnl_frac"], p["sharpe_frac"], p["sortino_frac"]])
    return out

# ---------- verdict rules -------------------------------------------------
def v_coinflip(k, rung):
    """Null A at MID: friction removed from both sides, so this is the pure
    directional-skill test. z = (observed SR - null mean)/null sd."""
    a = P["subs"][k]["rungs"]["mid"]["mc_a"]
    ss = P["subs"][k]["seed_stability"]
    z = a["sharpe_z"]; p = a["sharpe_p_value"]
    stable_best = ss["beats_max"] == "always"
    flips = "FLIP" in str(ss["beats_max"]) or ss["above_q999"] == "FLIPS"
    if p > 0.01 or z < 2.5: return "FAIL"
    if z < 4.0 or flips or not stable_best: return "SUSPECT"
    return "PASS"

def v_bootstrap(k, rung):
    b = P["subs"][k]["rungs"][rung]["boot"]
    lo = min(b["ci"][0], b["blk_ci"][0])
    if lo <= 0: return "FAIL"
    if lo < 1.0: return "SUSPECT"
    return "PASS"

def v_dsr(k, rung):
    d = D["subs"][k]["rungs"][rung]["dsr"]
    N = d["declared"]; brk = d["break_analytic"]
    if brk is None: return "PASS"
    if brk < N: return "FAIL"
    if brk < 10*N: return "SUSPECT"
    return "PASS"

def v_days(k, rung):
    r = D["subs"][k]["rungs"][rung]
    z = r["days_to_zero"]; act = r["active_days"]
    if z is None or act == 0: return "FAIL"
    pct = 100.0*z/act
    if pct < 5: return "FAIL"
    if pct < 15: return "SUSPECT"
    return "PASS"

def v_base(k, rung):
    bb = B["subs"][k]["rungs"][rung]
    ratio = BR[k]["ratio"]                       # breadth-neutral, priced at mid
    beaten_by_long = bb["always_long"]["sharpe"] > bb["strategy"]["sharpe"] + 0.05
    if ratio < 1.0 or beaten_by_long: return "FAIL"
    if ratio < 3.0: return "SUSPECT"
    return "PASS"

TESTS = [
    ("coinflip",  "Coin-flip trader",        v_coinflip),
    ("bootstrap", "Bootstrapped Sharpe",     v_bootstrap),
    ("dsr",       "Deflated Sharpe",         v_dsr),
    ("days",      "Best-days removal",       v_days),
    ("baseline",  "Baseline comparison",     v_base),
]
ORD = {"PASS": 0, "SUSPECT": 1, "FAIL": 2}

out = {
    "generated": D["generated"], "fee": D["fee"], "ann": D["ann"],
    "v_search": D["v_search"], "rungs": RUNGS, "order": SUBS,
    "drift_per_step": P["drift_per_step"], "spread_ts": P["spread_ts"],
    "tests": [{"id": t[0], "name": t[1]} for t in TESTS],
    "subs": {},
}

for k in SUBS:
    p = P["subs"][k]; d = D["subs"][k]; b = B["subs"][k]
    rec = {
        "name": p["name"], "class": p["class"], "note": p["note"],
        "declared_trials": p["trials"], "days": p["days"],
        "n_trips": p["n_trips"], "vol_mwh": p["vol_mwh"],
        "long_frac": round(p["long_frac"], 3), "hold_med": p["hold_med"],
        "entry_slot": b["entry_slot"], "hold_steps": b["hold_steps"],
        "claimed": p["claimed"],
        "edge_ratio_mid": round(BR[k]["ratio"], 3),
        "prods_day": round(BR[k]["prods_day"], 2),
        "breadth": round(BR[k]["breadth"], 1),
        "passive_edge_mid": round(BR[k]["passive_edge"], 3),
        "edge_mid": round(BR[k]["edge"], 3),
        "be_hold": {"entry": BE[k]["entry"], "hold_min": BE[k]["hold_min"],
                    "drift": round(BE[k]["drift"], 3), "friction": round(BE[k]["friction"], 3),
                    "covers": BE[k]["covers"]},
        "seed": p["seed_stability"],
        "mvs": {kk: (round(vv, 4) if isinstance(vv, (int, float)) else vv)
                for kk, vv in p["rungs"]["mid"]["mvs"].items() if kk != "regression"},
        "rungs": {}, "verdicts": {},
    }
    for rung in RUNGS:
        pr = p["rungs"][rung]; dr = d["rungs"][rung]; br = b["rungs"][rung]
        rec["rungs"][rung] = {
            "pnl": round(dr["pnl"]), "sharpe": round(dr["sharpe"], 3),
            "sortino": None if dr["sortino"] is None else round(dr["sortino"], 3),
            "hit": round(pr["hit"], 4), "max_dd": round(pr["max_dd"]),
            "active_days": dr["active_days"], "n_trips": dr["n_trips"],
            "boot": {"ci": [round(x, 3) for x in pr["boot"]["ci"]],
                     "blk_ci": [round(x, 3) for x in pr["boot"]["blk_ci"]],
                     "t": round(pr["boot"]["t"], 2), "blk_t": round(pr["boot"]["blk_t"], 2),
                     "blk_len": pr["boot"]["blk_len"],
                     "so_ci": [round(x, 2) for x in pr["boot"]["so_ci"]],
                     "so_t": None if pr["boot"]["so_t"] is None else round(pr["boot"]["so_t"], 2),
                     "nw_t": round(pr["nw_t"], 2)},
            "mc_a": {kk: pr["mc_a"][kk] for kk in
                     ("sharpe_null_mean", "sharpe_null_sd", "q99", "q999",
                      "sharpe_null_max", "sharpe_p_value", "sharpe_z",
                      "above_q999", "sharpe_beats_best_run", "n")},
            "mc_b": {kk: pr["mc_b"][kk] for kk in
                     ("sharpe_null_mean", "q999", "sharpe_null_max",
                      "sharpe_p_value", "sharpe_z", "above_q999")},
            "hist_a": {"c": pr["hist"]["a"], "e": [round(x, 3) for x in pr["hist"]["a_edges"]]},
            "dsr": {kk: dr["dsr"][kk] for kk in
                    ("sr_ann", "skew", "kurt", "declared", "at_declared_analytic",
                     "at_declared_search", "break_analytic", "break_search",
                     "sr0_ann_analytic", "sr0_ann_search", "curve_analytic", "curve_search")},
            "days_to_zero": dr["days_to_zero"], "trips_to_zero": dr["trips_to_zero"],
            "top1_share": dr["top1_share"], "top5_share": dr["top5_share"],
            "pos_trip_frac": dr["pos_trip_frac"],
            "decay": thin(dr["curve"]),
            "base": {kk: {"pnl": round(br[kk]["pnl"]), "sharpe": round(br[kk]["sharpe"], 3)}
                     for kk in ("strategy", "always_long", "always_short",
                                "market_bh_full", "matched_passive")},
            "edge": round(br["edge_eur_mwh"], 4),
            "matched_edge": round(br["matched_edge_eur_mwh"], 4),
            "sizing": {"unitclip_sharpe": round(pr["sizing"]["unitclip_sharpe"], 3),
                       "trip_mean": round(pr["sizing"]["trip_mean_eur_mwh"], 4),
                       "trip_t": round(pr["sizing"]["trip_t"], 2),
                       "margin_sharpe": round(pr["sizing"]["margin_sharpe"], 3)},
        }
        v = {tid: fn(k, rung) for tid, _, fn in TESTS}
        worst = max(v.values(), key=lambda s: ORD[s])
        v["overall"] = worst
        rec["verdicts"][rung] = v
    rec["equity"] = {r: p["equity"][r] for r in RUNGS}
    out["subs"][k] = rec

out["dates"] = P["subs"]["windfall"]["dates"]
json.dump(out, open("five_tests.json", "w"), separators=(",", ":"))
import os
print("five_tests.json", round(os.path.getsize("five_tests.json")/1024, 1), "KB")
print(f"{'sub':14s}" + "".join(f"{t[1][:11]:>13s}" for t in TESTS) + f"{'OVERALL':>10s}")
for k in SUBS:
    v = out["subs"][k]["verdicts"]["cross_fee"]
    print(f"{out['subs'][k]['name']:14s}" + "".join(f"{v[t[0]]:>13s}" for t in TESTS) + f"{v['overall']:>10s}")
