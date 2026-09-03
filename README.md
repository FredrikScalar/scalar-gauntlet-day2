# Gauntlet day — the working pack

Seven strategies have been proposed for capital. Your job, as one team, is an
automated validation report: a pipeline that takes any submission's records
and returns PASS, SUSPECT with conditions attached, or FAIL — with the
evidence behind every line. The same pipeline runs over all seven.

Read first: `docs/submissions-book.pdf` (the seven, as they reached the
desk), then `docs/six-sections-handout.pdf` (what a validation report is
made of).

## What's here

- `data/` — the market as it traded: order book (3 levels per side),
  forecast vintages with issue timestamps, day-ahead prices, actuals, the
  outage feed. Schema and conventions: `data/DATA.md`.
- `blotters/` — one CSV per submission, every trade: `exec_ts,
  product_delivery, side, qty_mw, price`. Same schema for all seven, so
  everything you build runs on all of them unchanged.
- `registry/` — one YAML per submission: params, declared trial counts, dev
  windows, data dependencies, claimed stats — the submitter's own
  statements, unverified — plus empty `validation:` fields your pipeline
  fills. `loader.py` loads them all.
- `starter/` — `00_start_here.ipynb` (load everything, reproduce the book's
  numbers from the records), `repricer.py` (mid-only blotter repricing —
  repricing at prices you'd actually get is yours to build), and
  `report.py` (the Report contract: six sections, verdicts, the aggregation
  rule — the build target).
- `docs/` — the two handouts.

## What's not here

No strategy source code and no callables — records are all a validator
gets. The truth about the seven, and six further months of market, are
sealed until the reveal.
