"""Extract round trips from a flat blotter (every product nets to zero)."""
import numpy as np, pandas as pd, engine as E

def round_trips(b: pd.DataFrame, rung: str = "mid") -> pd.DataFrame:
    """FIFO-match each product's trades into round trips.

    Returns one row per closed lot: entry/exit timestamps and prices, signed
    direction (+1 long, -1 short) and qty. P&L for any rung is then
    dir*qty*(exit_px - entry_px), which is exactly the blotter's sells-minus-buys.
    """
    px = E.eff_price(b, rung)
    d = pd.DataFrame({
        "product_id": b.product_id.values, "delivery": b.product_delivery.values,
        "ts": b.exec_ts.values, "qty": b.qty_mw.values,
        "sgn": np.where(b.side.eq("BUY").values, 1.0, -1.0), "px": px,
    }).sort_values(["product_id", "ts"], kind="stable")

    out = []
    for pid, g in d.groupby("product_id", sort=False):
        open_lots = []          # [qty, sgn, px, ts]
        for ts, qty, sgn, p in zip(g.ts.values, g.qty.values, g.sgn.values, g.px.values):
            while qty > 1e-9:
                if open_lots and open_lots[0][1] != sgn:      # closes existing
                    lot = open_lots[0]
                    m = min(qty, lot[0])
                    entry_sgn = lot[1]
                    out.append((pid, g.delivery.values[0], lot[3], ts, m,
                                entry_sgn, lot[2], p))
                    lot[0] -= m; qty -= m
                    if lot[0] <= 1e-9: open_lots.pop(0)
                else:
                    open_lots.append([qty, sgn, p, ts]); qty = 0.0
        assert not open_lots, f"product {pid} left open"
    t = pd.DataFrame(out, columns=["product_id","delivery","entry_ts","exit_ts",
                                   "qty","dir","entry_px","exit_px"])
    t["pnl"] = t["dir"]*t["qty"]*(t["exit_px"]-t["entry_px"])
    t["hold_min"] = (t.exit_ts - t.entry_ts).dt.total_seconds()/60
    t["day"] = pd.DatetimeIndex(t.delivery).normalize()
    return t

if __name__ == "__main__":
    mk = E.load_market()
    for k in E.SUBS:
        b = E.reprice(E.load_blotter(k), mk)
        t = round_trips(b, "mid")
        chk = abs(t.pnl.sum() - E.daily_pnl(b, "mid").sum())
        print(f"{k:13s} trips={len(t):6d} long={int((t.dir>0).sum()):6d} short={int((t.dir<0).sum()):6d} "
              f"pnl={t.pnl.sum():10.1f} recon_err={chk:.6f} hold_med={t.hold_min.median():.0f}m")
