# Market data

One market, one zone (ZONE1), hourly delivery products, 546 delivery days
starting 2025-01-01. All timestamps are UTC. Gate closure is delivery start
minus 30 minutes. The order book is snapshotted every 15 minutes over the
last 10 hours before gate (40 snapshots per product).

## Tables

**products.csv** — one row per delivery product.
`product_id, zone, delivery_start, delivery_end, resolution`

**da_auction.csv** — the day-ahead auction price for each product.
`product_id, da_price_eur_mwh`

**book_snapshots.csv** — the intraday order book, three levels per side.
`ts, product_id, bid_px_1..3, bid_sz_1..3, ask_px_1..3, ask_sz_1..3`
Prices in EUR/MWh, sizes in MW. There is no mid column: mid is a derived
convenience, `(bid_px_1 + ask_px_1) / 2` — it is not a price anyone quoted,
and nothing in these files says you can trade at it.

**forecasts.csv** — every forecast vintage as it was issued.
`issue_ts, target_ts, variable, value_mw` with `variable` in
{wind, solar, load}. `target_ts` is the delivery hour the forecast is for;
`issue_ts` is when that number became known. Vintages update hourly. What
was knowable at any decision time is exactly the set of rows with
`issue_ts <= that time` — treat this table point-in-time or not at all.

**actuals.csv** — outturn per delivery hour.
`target_ts, variable, value_mw`

**outages.csv** — announced planned unavailability, aggregated across the
fleet, published day-ahead (REMIT-style).
`publish_ts, delivery_date, planned_out_mw`

## Conventions

Trades in the blotters reference products by `product_delivery`
(= `delivery_start`) and timestamps by `exec_ts`, which always lands on a
book snapshot `ts`. Every blotter position is opened and closed through
trades — no open positions remain at gate, so P&L is sells minus buys per
product at whatever price column you value the trades at.
