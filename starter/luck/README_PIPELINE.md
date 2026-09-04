# Five-tests pipeline — one report per strategy

Produces the interactive HTML report for one submission. Everything is
recomputed from `data/` and `blotters/` through `engine.py`, `trips.py` and
`luck.py`. No earlier payload is read.

## Install

Python 3.9 or newer, plus four packages:

```
cd starter/luck
python3 -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 pipeline.py --check     # confirms deps, data, fonts and cache
```

`--check` computes nothing. It prints the Python and package versions it found,
whether the book, blotters and registry are reachable, whether the IBM Plex
faces are present (they make the reports work offline), and how much of the
cache is usable. Fix anything it marks MISSING before running a report.

## Run

```
python3 pipeline.py windfall                 # reports/windfall-five-tests.html
python3 pipeline.py windfall windfall2       # two individual reports
python3 pipeline.py all                      # one individual report per strategy
python3 pipeline.py blackbox --rung mid      # open the page on a different basis
python3 pipeline.py sunspot --quick          # 1,000 draws — a fast look, not publishable
```

**Every report is about one strategy only.** No other submission appears in it,
or in its embedded payload. Rather than comparing strategies with each other, a
report compares its strategy against **itself across the three execution
bases** — which is where the friction story lives, and the only comparison that
can be made from one submission's own records. Two figures are unavoidable
exceptions, flagged in place on the page: the cross-sectional trial variance
behind the harsh deflated Sharpe, and the market denominator of the
breadth-neutral edge ratio. Both come from the cohort by construction.

The combined seven-strategy page is **opt-in** and is the only output where
other submissions appear:

```
python3 pipeline.py all --combined           # seven individual + the combined page
python3 pipeline.py all --combined-only      # only the combined page
```

Run everything from `starter/luck/`. The repo root is found by looking for
`data/book_snapshots.csv` above this folder, so a differently nested checkout
still works (this also repoints `engine.D`, which hard-codes its own root).

## Files

| file | role |
|---|---|
| `pipeline.py` | CLI. Preflight, compute (or load), render, write. Start here. |
| `requirements.txt` | numpy, pandas, scipy, PyYAML, with version floors. |
| `fivetests.py` | the compute layer: one record per submission, all three rungs. Also a CLI: `python3 fivetests.py windfall`. |
| `verdicts.py` | **every Keep / Suspect / Fail threshold, in one place.** Change a number here and the whole report set moves with it. |
| `render.py` | payload assembly + template fill. Font embedding lives here. |
| `report.css` | the design system, shared by both templates. Edit colours and type here once. |
| `template_one.html` | the single-strategy page. Series are the three execution bases. |
| `template_all.html` | the combined seven-strategy page. Series are the strategies. |
| `fonts/` | nine IBM Plex woff2 faces, inlined as data URIs so a report opens offline. |
| `cache/<sub>.json` | the computed record. Safe to commit; safe to delete. |

## What a run costs

Loading the book takes ~3 s. Then, per submission at 10,000 draws:

| | trips | compute |
|---|---|---|
| windfall, windfall2, sunspot, spikecatcher, bounceback, blackbox | 276 – 2,880 | 4 – 12 s |
| **pingpong** | 47,022 | **~4½ min** |

Monte Carlo Null B dominates, and it scales with trip count — PINGPONG is
almost all of a full run. So:

* one strategy from cold is ~10–20 s (PINGPONG ~4½ min);
* `pipeline.py all` from cold is ~5½ min, almost all of it PINGPONG;
* naming one strategy computes only that strategy — the other six are never
  touched, because its report does not contain them;
* every later run on any subject is **under a second** — records come from
  `cache/`.

The cache invalidates itself when `book_snapshots.csv`, `products.csv`, that
strategy's blotter or registry entry, or any of `fivetests.py`, `luck.py`,
`engine.py`, `trips.py`, `verdicts.py` is newer than the record. `--refresh`
forces a recompute. Bump `fivetests.SCHEMA` when you change the record shape.

## What the report contains

Numbered to match the review's own bullets:

0. **What friction does to the verdict** — the five tests scored on each of the
   three execution bases. Nothing changes between the rows but what you pay to
   get in and out, so this is where a strategy that only works at mid shows up.
   (On the combined page this section is the seven-strategy verdict grid instead.)
1. **Breakeven decay** — delete the best day, the two best, … until the P&L
   crosses zero, with one line per execution basis so you can watch friction
   shorten the fuse. Toggles: P&L / Sharpe / Sortino, and days / % of active days.
2. **Coin-flip trader** — both nulls drawn full size: Null A (side randomised on
   the same trades) and Null B (product, entry snapshot and side all random),
   with the strategy's position marked in each, z-distance per basis, and the
   five-seed stability table showing which statistics the verdict may rest on.
3. **Bootstrapped Sharpe** — 10,000 iid and 8-day block resamples, one interval
   per basis, bootstrap t, Newey–West t and AR(1).
4. **Deflated Sharpe** — the DSR-against-trials curve under both variance
   assumptions on one chart, the declared trial count drawn as a rule, and the
   breakeven trial count labelled on each curve.
5. **Baseline comparison** — always long / always short on the strategy's own
   trades, matched-horizon passive, full-window buy-and-hold, and the
   breadth-neutral edge ratio.
6. **Everything else on the record** — cumulative P&L on all three bases,
   claimed against reproduced, sizing stripped out, market-versus-skill, the
   breakeven hold, and what the submitter declared but nobody verified.

Interactive throughout: the execution-basis switch rewires the whole page —
headline numbers, verdicts, prose and which series is drawn solid; every chart
has a table view underneath covering all three bases, and hover readouts.

## Conventions worth knowing

* **Fee** 0.12 EUR/MWh per side. **Slippage** is modelled as spread crossing,
  not market impact. `cross_fee` (buy the ask, sell the bid, fee both sides) is
  the honest rung; `mid` is the price nobody quoted.
* **Buy-and-hold** runs snapshot 0 (delivery start − 10h30m) to snapshot 39
  (start − 45m). Snapshot 39 is the last tradeable row — fifteen minutes before
  gate closure at start − 30m. Nothing in the pack reaches gate.
* **Baselines are priced on both legs at the same rung.** `run_full.py` priced
  the strategy leg at mid while pricing the benchmarks at the selected rung,
  which made the `cross_fee` comparison apples-to-oranges.
* **Sortino** is the submitters' definition — sample SD of negative days only —
  because that is what reproduces their claimed figures.
* **The coin-flip verdict is always scored at mid**, whatever basis the page is
  showing. At mid the null sits on zero, so the distance measures direction
  skill; at a rung charging friction the null goes deeply negative and beating
  it only proves the strategy churned less than a random trader.

## Superseded

`decay.py`, `base2.py`, `report_data.py`, `five_tests.json` and `template.html`
were the one-off scripts and the earlier single template behind the first
combined report. `pipeline.py` with `template_one.html` / `template_all.html`
replaces all of them and recomputes every number from raw data; they can be
deleted.
