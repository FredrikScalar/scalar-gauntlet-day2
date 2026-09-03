# The market data — full reference

One market, one zone (`ZONE1`). Hourly delivery products over 546 delivery
days starting **2025-01-01**, so 546 × 24 = 13,104 products. All timestamps
in every file are **UTC** and every price is **EUR/MWh**, every quantity
**MW**. Prices are rounded to 2 dp, forecast/outage MW to 1 dp.

The clock around each product:

- **Delivery**: one hour, `delivery_start` to `delivery_end` (= start + 1h).
- **Gate closure**: `delivery_start − 30 min`. Nothing trades after gate.
- **Trading grid**: the order book is snapshotted every **15 minutes** over
  the last **10 hours** before gate — 40 snapshots per product, from
  `gate − 10h` up to `gate − 15min`. All trading happens on this grid.
- **Forecast updates**: hourly, aligned with every 4th snapshot, from
  `gate − 10h` to `gate − 1h` — 10 vintages per product per variable.

So products are *hourly*; it is the trading clock that is 15-minutely.

## Files at a glance

| file | rows | one line |
|---|---|---|
| `products.csv` | 13,104 | the product master: what each `product_id` delivers |
| `da_auction.csv` | 13,104 | the day-ahead auction price per product |
| `book_snapshots.csv` | 524,160 | the intraday order book, 3 levels per side |
| `forecasts.csv` | 393,120 | every wind/solar/load forecast vintage, with issue time |
| `actuals.csv` | 39,312 | outturn per delivery hour per variable |
| `outages.csv` | 546 | announced planned unavailability, published day-ahead |

## products.csv

`product_id, zone, delivery_start, delivery_end, resolution`

`product_id` runs 0…13,103 in delivery order: `day_index * 24 + hour`, so
consecutive ids are consecutive hours. `resolution` is `"H"` throughout —
the field exists so finer products could be added without a schema change.
This is the join hub: blotters reference products by `product_delivery`
(= `delivery_start`); the book references them by `product_id`.

## da_auction.csv

`product_id, da_price_eur_mwh`

The day-ahead auction clears once, the day before delivery. One price per
product, known before any intraday snapshot of that product exists. The
first intraday mids sit near — but not at — this price; the difference and
how it decays toward gate is real structure, not noise.

## book_snapshots.csv

`ts, product_id, bid_px_1..3, bid_sz_1..3, ask_px_1..3, ask_sz_1..3`

One row per (snapshot, product). Level 1 is best bid/ask; levels 2 and 3
sit behind it at fixed price steps with 10 MW displayed at every level.

Three things to internalise:

1. **There is no mid column.** Mid is a derived convenience,
   `(bid_px_1 + ask_px_1) / 2`. It is not a price anyone quoted, and
   nothing in these files says you can trade at it. The submissions'
   claimed numbers only reproduce at mid — treat that as a finding.
2. The **spread** (`ask_px_1 − bid_px_1`) varies by product and by time to
   gate. It is not a constant to be assumed once; it is a column to be
   joined per trade.
3. **Displayed depth** is what you get per level. A 10 MW order consumes
   level 1; anything bigger walks the book. Every submission trades a
   10 MW clip — but the question "at what size does this die?" is still
   answerable from the size columns.

## forecasts.csv

`issue_ts, target_ts, variable, value_mw` — `variable` ∈ {`wind`, `solar`, `load`}

Every vintage as it was issued. `target_ts` says which delivery hour the
number is for (= that product's `delivery_start`); `issue_ts` says when the
number **became known**. For one (product, variable) there are 10 rows: the
hourly updates from `gate − 10h` to `gate − 1h`.

**The discipline: this table is point-in-time or it is nothing.** What was
knowable at decision time `t` is exactly the rows with `issue_ts <= t` —
the latest such row per (target, variable) is the live forecast, the
difference between it and the previous vintage is the revision the market
is digesting. Any calculation that touches a row with `issue_ts > t` while
reasoning about time `t` has time-travelled. Whether a *submission* has
time-travelled is, of course, one of the questions on the table.

```python
# live wind forecast at each book snapshot, per product — the honest join
wind = forecasts[forecasts.variable == "wind"].sort_values("issue_ts")
live = pd.merge_asof(
    book.sort_values("ts"), wind,
    left_on="ts", right_on="issue_ts",
    left_by="delivery_start", right_by="target_ts")  # after joining product_id -> delivery_start
```

## actuals.csv

`target_ts, variable, value_mw`

Outturn per delivery hour. Actuals are settlement data: they do not exist
until after delivery, so they may explain P&L after the fact but can never
sit inside a decision.

## outages.csv

`publish_ts, delivery_date, planned_out_mw`

Announced planned unavailability aggregated across the generation fleet
(REMIT-style), one row per delivery day, published at **10:00 UTC the day
before**. From its `publish_ts` onward the number is public — day-ahead and
intraday participants alike know how much capacity has declared itself out.
Join to products on `delivery_date = delivery_start.date()`.

## The blotters (../blotters/)

`exec_ts, product_delivery, side, qty_mw, price`

One CSV per submission — its every trade, and the same schema for all
seven. `exec_ts` always lands on a book snapshot `ts`; `product_delivery`
equals a product's `delivery_start`; `side` is `BUY`/`SELL` with `qty_mw`
positive; `price` is the fill the submitter reports.

Every position is opened and closed through trades — nothing is left open
at gate. So a product's P&L needs no marking of open positions at any
price column you choose:

```python
signed = np.where(b.side.eq("SELL"), 1, -1) * b.qty_mw * b[price_col]
pnl_by_product = signed.groupby(b.product_delivery).sum()
```

Group product P&L by delivery day for daily P&L. Convention for
comparability: annualise Sharpe with √365 over the submission's **declared
backtest window** (in its registry entry), flat days included — Sharpe over
active days only flatters everyone.

## Gotchas worth 20 minutes of someone's day

- Parse timestamps as UTC (`pd.read_csv(..., parse_dates=[...])` keeps the
  `+00:00` offset). Mixing tz-aware and tz-naive datetimes is the classic
  silent join-killer: an equality join that matches nothing, or matches
  everything a timezone off.
- Join the blotter to the book on **both** `exec_ts` and `product_id`
  (via products) — joining on time alone matches 24 products at once.
- `book_snapshots` is half a million rows; load it once, not per blotter.
- Forecast *levels* and forecast *revisions* are different objects. Most
  stories in the book are about revisions.

`starter/repricer.py` implements the loading and the mid repricing;
`starter/00_start_here.ipynb` proves the whole thing round-trips.
