# Luck section — analysis pipeline

Everything the Luck audit computes, runnable from `starter/`. No strategy source
code is used: round trips are recovered from the blotters by FIFO matching and
repriced against `book_snapshots.csv`.

## Run order

```
python3 run_full.py     # ladder + full battery over 3 execution bases -> results.json  (~6 min)
python3 sweep.py        # passive-capture sweep, breakeven spread capture
python3 drift.py        # intraday drift by horizon + matched-horizon passive baseline
python3 xsec.py         # crossed-book comparison vs buy-and-hold
python3 imb.py          # exit rule stripped: hold-to-gate variants
python3 payload.py      # trim everything into payload.json for the HTML report
```

## Modules

| file | what it does |
|---|---|
| `engine.py` | market loading, trade-level repricing across the execution ladder, metrics |
| `trips.py` | FIFO round-trip matching per delivery product (reconciles to the cent) |
| `luck.py` | the battery: bootstrap, deflated Sharpe, drop-best, baselines, sizing, market-vs-skill, Monte Carlo nulls, `BookGrid` |

## Execution bases (`engine.LADDER`, `BookGrid.leg_prices`)

- `mid` — both legs at (bid+ask)/2. **All seven claims reproduce here exactly.**
- `mid_fee` — mid both legs, 0.12 EUR/MWh each side
- `passive_in_fee` — entry rests at mid, exit crosses, fee both legs
- `passive_out_fee` — entry crosses, exit rests at mid, fee both legs
- `cross`, `cross_fee` — buy ask / sell bid
- `walk_fee` — walks levels 1-3 on cumulative demand at each (snapshot, product, side)

Book depth is a uniform 10 MW at each of 3 levels and every blotter clip is 10 MW,
so `walk_fee` differs from `cross_fee` only for PINGPONG's 20 MW clips.

## Metric definitions worth knowing

- **Sortino** is the submitters' definition: sample SD of *negative days only*
  (`sortino_claimed`). This is what reproduces all seven claimed figures exactly.
  `sortino_strict` (RMS over negative days) is also computed.
- **Deflated Sharpe** (Bailey & López de Prado) needs a trial variance that the
  records do not contain. Two assumptions are reported: the analytic sampling
  variance of the Sharpe estimator (generous — trials differ only by noise), and
  the cross-sectional dispersion of daily Sharpe across the seven (harsh).
  `trials_to_break` bisects for the trial count at which DSR crosses 0.95.
- **Monte Carlo Null A** — same timestamps, products, clips; side randomised.
- **Monte Carlo Null B** — same delivery day, trade count and holding periods;
  product, entry snapshot and side all drawn at random from that day.
  Both nulls price each side through `leg_prices`, so at any basis charging
  friction the null is *not* antisymmetric — a coin flip pays the cost either way.

## Known findings this pipeline produces

- Mid repricing reproduces every claimed P&L, Sharpe, Sortino, drawdown,
  active-day fraction and hit rate.
- The book drifts up **+0.164 EUR/MWh per 15-minute step**, +6.39 over the full
  window, but only 48.2% of products end higher — a right-skewed drift. Passive-long
  Sharpe is therefore a function of holding period (3.24 at 15 min, 11.65 at 585 min),
  which is why the matched-horizon baseline is the fair one.
- PINGPONG's mean gross edge is 0.242 EUR/MWh against a 0.24 EUR/MWh round-trip fee.
- BLACKBOX declares 3 trials but names `model_id: 371`; DSR at 3 is 0.998, at 371 it
  is 0.791, and it breaks below 0.95 at 35 trials.
- SPIKECATCHER, BOUNCEBACK and BLACKBOX have headline Sharpes *below* the luckiest of
  10,000 coin-flip runs on their own timestamps, at mid.

## Not in scope

Execution modelling (fill probability, adverse selection, market impact),
Lineage, Shelf-life, Evidence and Warranty. There is **no imbalance price** in
`data/` — `actuals.csv` is volumes in MW, `da_auction.csv` clears before the
intraday session — so imbalance settlement is not computable from these records.

## Verification

`verify.py` runs 87 checks and must come back green before any number is published:

- every claimed statistic reproduces at mid (P&L, Sharpe, Sortino, drawdown, hit rate)
- the fee drop equals exactly `0.12 x total MWh` for every submission
- round-trip reconstruction reconciles with trade-level arithmetic at every rung
- bootstrap CI endpoints are stable across four seeds
- the drift constants (+0.164/step, +6.39 full window, 48.2% ending higher) replicate
- Monte Carlo verdicts rest on **seed-stable** statistics

### Why the Monte Carlo verdict does not use the null maximum

`seedstab.py` reruns Null A with five independent seeds. The maximum of the null is
the noisiest statistic in the whole section:

| | observed | null max across seeds | "beats best run?" |
|---|---|---|---|
| BOUNCEBACK | 4.81 | 4.77 – 6.27 | **flips with the seed** |
| SPIKECATCHER | 2.42 | 2.58 – 3.10 | stable (never) |
| BLACKBOX | 5.83 | 6.07 – 7.06 | stable (never) |

The 99.9th percentile of the same distributions moves only 0.15–0.43 across those
seeds and the p-values are stable to four decimals, so the verdict uses the
**p-value and p99.9**. The best-of-10,000 figure is reported as context only.

### Why the drop-best-days verdict does not use Sharpe

`skew.py` decomposes it. Sharpe = mean/SD; removing a positive outlier cuts the
mean linearly and the SD quadratically, so on right-skewed P&L Sharpe *rises* on
truncation. WINDFALL 2.0 loses 22.8% of its mean but 42.4% of its SD when its best
five days are removed, so its Sharpe climbs 6.87 → 9.20 while P&L falls 24%. The
verdict therefore uses the two monotone measures: **P&L retained** and
**days-to-erase**.

## Market term structure (added after the first pass)

`spread_ts.py`, `breakeven.py`, `breadth.py` — run after `run_full.py`, before `payload.py`.

### The clock does two things at once

| time to gate | 10h | 8h | 6h | 4h | 2h | 15m |
|---|---|---|---|---|---|---|
| mean L1 spread | 1.171 | 1.075 | 0.979 | 0.883 | 0.787 | **0.703** |
| cumulative drift | 0.000 | 1.699 | 3.301 | 4.473 | 5.616 | **6.391** |

The spread tightens on **all 39 of 39 steps**, linearly at −0.012 EUR/MWh per
15 minutes — a 40.0% narrowing end to end. Drift accumulates at +0.164/step.
Together they imply a **breakeven hold of 105 minutes**: entering at the 10-hour
snapshot, that is how long you must hold before the market's own drift covers one
round trip of friction. Enter inside 2h of gate and there is never enough time.

Only SPIKECATCHER (225m) and BLACKBOX (585m) hold past breakeven. The other five
must earn their entire friction bill from signal.

### Breadth confounds the Sharpe-vs-passive comparison

The passive benchmark holds all 24 products every day; SUNSPOT holds 0.48. Sharpe
pays for that diversification at an identical per-MWh edge, so the baseline verdict
uses the **breadth-neutral edge ratio** — edge per MWh of position divided by what
holding the whole market over the same clock window returned:

| | products/day | edge/MWh | market, same window | edge ratio | breadth against it |
|---|---|---|---|---|---|
| WINDFALL 2.0 | 2.88 | 8.760 | 1.049 | 8.35× | 8.3× |
| SUNSPOT | 0.48 | 4.699 | 0.587 | 8.00× | 50.2× |
| WINDFALL | 2.15 | 2.905 | 0.411 | 7.07× | 11.2× |
| PINGPONG | 23.99 | 0.242 | 0.101 | 2.40× | 1.0× |
| SPIKECATCHER | 1.20 | 4.589 | 1.918 | 2.39× | 20.0× |
| BOUNCEBACK | 6.86 | 1.683 | 1.040 | 1.62× | 3.5× |
| BLACKBOX | 24.00 | 1.414 | 6.391 | **0.22×** | 1.0× |

This **corrected two verdicts**. SUNSPOT and SPIKECATCHER appeared to lose to a
passive long on Sharpe; per MWh their selection is 8.00× and 2.39× the market, and
the gap is a capacity limitation rather than an absent edge. BLACKBOX has no breadth
confound at all — 24 products a day over the identical window — and captures 0.22×
what a flat long collected on the same products at the same times.

## Exit-price conventions (`gateclose.py`, `bhexit.py`, `twoclose.py`)

- The book's last row is snapshot 39 = `delivery start − 45 min`, i.e. **15 minutes
  before gate closure**. Nothing in the pack reaches gate.
- **No blotter carries a position into delivery** — zero open positions at gate for
  all seven, verified. SPIKECATCHER and BLACKBOX exit at snapshot 39 on 100% of trips.
- Buy-and-hold enters by **lifting the ask** at snapshot 0 (mean 38.036, +fee → 38.156)
  and exits by **hitting the bid** at snapshot 39 (mean 43.489, −fee → 43.369). It
  crosses on **both** legs: 0.586 + 0.351 spread + 0.240 fees = **1.177 EUR/MWh**, the
  same round trip anyone pays. Its advantage is paying it once over 9h45 against 6.391
  of drift, not paying less.
