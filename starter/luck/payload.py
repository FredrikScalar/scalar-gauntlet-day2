import json, numpy as np, pandas as pd, engine as E, luck as L, trips as T
res=json.load(open("results.json")); mk=E.load_market(); bg=L.BookGrid(mk)
SUBS=E.SUBS; RUNGS=["mid","passive_in_fee","cross_fee"]
NOTE={
 "windfall":"The intraday market underreacts to wind forecast revisions; the price response completes over 15-45 minutes rather than instantly. Edge per trade comfortably exceeds typical spreads. The edge has softened somewhat in recent months.",
 "windfall2":"Windfall's economics, upgraded data. A provider trial gives materially fuller vintage coverage - we see revisions earlier and more completely than the public feed. The trading framework is identical; the improvement is entirely in the input.",
 "sunspot":"The solar application of the Windfall framework: cloud-cover updates are priced sluggishly, particularly in shoulder hours. Daylight products only. H1 2026 has been flat - an unusually stable forecast regime.",
 "spikecatcher":"When margin is thin the intraday market underprices the right tail. We go long a fixed 10 MW in the top 5% of exposed delivery hours, four hours before gate, held to close. Most active days we bleed the premium; the spike days pay for everything.",
 "pingpong":"Short-horizon liquidity provision. When a product's mid dislocates by more than the noise floor but less than a news-sized move, we fade it for one 15-minute interval. We execute passively, resting orders inside the spread. High frequency, small per-trade edge, large N.",
 "bounceback":"Downward overreactions revert; upward moves are physical. When oversupply knocks a product more than 5.5 EUR down inside an hour, the move systematically overshoots and partially retraces. We fade the downside only. Entry on the trigger, hold 45 minutes.",
 "blackbox":"A machine-learned ensemble over 42 intraday features, trained to emit one direction per product per session, entered at open and held to gate. The features are individually weak; the edge is emergent. Model 371 is the production candidate from our runs registry.",
}
DROPS=[0,1,3,5,10]
out={"fee":E.FEE,"rungs":RUNGS,"drift_per_step":0.16387,
     "drift_curve":[[1,3.24],[2,3.98],[3,4.76],[4,4.21],[6,4.50],[8,5.48],[12,6.45],[20,8.31],[30,10.24],[39,11.65]],
     "subs":{}}
for k in SUBS:
    r=res["subs"][k]
    b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid")
    e=bg.slot_of(t.product_id.values,t.entry_ts.values)
    o={"name":r["name"],"class":r["class"],"note":NOTE[k],"window":r["window"],
       "days":r["days"],"trials":r["declared_trials"],"dev_window":r["dev_window"],
       "params":r["params"],"claimed":r["claimed"],"n_trades":r["n_trades"],
       "n_trips":r["n_trips"],"vol_mwh":r["vol_mwh"],"long_frac":r["long_frac"],
       "hold_med":r["hold_med_min"],"half_spread":r["half_spread_mean"],
       "matched_bh":r["matched_bh"],"entry_slot_med":int(np.median(e)),
       "capture":r["capture_sweep"],"dates":r["dates"],"rungs":{}}
    for rung in RUNGS:
        L_=r["battery"][rung]; lad=r["ladder"][rung]
        bt=L_["boot"]; da=L_["dsr_analytic"]; ds=L_["dsr_search"]; rb=L_["robust"]
        o["rungs"][rung]={
          "pnl":lad["pnl"],"sharpe":lad["sharpe"],"sortino":lad["sortino_claimed"],
          "sortino_strict":lad["sortino_strict"],"max_dd":lad["max_dd"],
          "margin":lad["margin"],"active":lad["active_frac"],"hit":lad["hit_rate"],
          "boot":{"ci":bt["iid"]["sharpe_ci"],"se":bt["iid"]["sharpe_se"],
                  "t":bt["iid"]["sharpe_t"],"p":bt["iid"]["sharpe_p_le0"],
                  "blk_ci":bt["block"]["sharpe_ci"],"blk_t":bt["block"]["sharpe_t"],
                  "blk_len":bt["block_len"],
                  "so_ci":bt["iid"].get("sortino_ci"),"so_t":bt["iid"].get("sortino_t"),
                  "so_se":bt["iid"].get("sortino_se")},
          "nw_t":L_["nw_t_mean_daily"],"ar1":L_["ar1"],
          "dsr":{"sr_ann":da["sr_ann"],"skew":da["skew"],"kurt":da["kurt"],
                 "sr0_ann":da["sr0_ann"],"val":da["dsr"],
                 "sr0_ann_search":ds["sr0_ann"],"val_search":ds["dsr"],
                 "break":L_["trials_to_break_analytic"],"curve":L_["dsr_curve"]},
          "robust":{"top1":rb["top1day_share"],"top5":rb["top5_share"],
                    "d2z":rb["days_to_zero"],"n_days":rb["n_days"],
                    "pos_trip":rb["pos_trip_frac"],
                    "top1pct_trips":rb["droptrips_0.01_share_lost"],
                    "curve":[[d, rb["total"] if d==0 else rb[f"drop{d}_pnl"],
                              lad["sharpe"] if d==0 else rb[f"drop{d}_sharpe"]] for d in DROPS]},
          "baselines":L_["baselines"],"sizing":L_["sizing"],
          "mvs":L_["market_vs_skill"],
          "mc_a":L_["mc_a"],"mc_b":L_["mc_b"],"hist":L_["mc_hist"],
        }
    o["equity"]={rr:[round(v) for v in r["equity"][rr]] for rr in RUNGS}
    out["subs"][k]=o
json.dump(out,open("payload.json","w"),separators=(",",":"),default=str)
import os; print("payload.json", round(os.path.getsize("payload.json")/1024,1),"KB")
