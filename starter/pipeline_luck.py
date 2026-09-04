#!/usr/bin/env python3
"""One command, one strategy: compute the five tests and write its report.

    python3 pipeline.py windfall
    python3 pipeline.py windfall windfall2
    python3 pipeline.py all                    # seven reports + the combined page
    python3 pipeline.py all --combined-only    # just the combined page

What it does, in order:
  1. loads the market once (book_snapshots.csv, products, registry, blotters)
  2. computes the record for each strategy asked for and caches it in cache/
  3. renders reports/<strategy>-five-tests.html

A single-strategy report is about that strategy only — no other submission
appears in it, or in its payload. Instead of comparing strategies, every chart
compares the strategy against ITSELF across the three execution bases, which is
where the friction story lives. `all` additionally writes the combined
seven-strategy page, which is the one place other submissions appear.

Everything is recomputed from raw data — no earlier payload is read. Records are
cached, so a second run on the same strategy renders in under a second; a change
to the data, to luck.py, engine.py, trips.py or fivetests.py invalidates the
cache automatically.

Options
  --rung {mid,passive_in_fee,cross_fee}   basis the page opens on (default cross_fee)
  --out DIR                               output directory (default reports/)
  --refresh                               recompute even if the cache looks fresh
  --no-embed-fonts                        link Google Fonts instead of inlining woff2
  --artifact                              also write a wrapper-less copy for publishing
  --combined-only                         with 'all', write only the seven-strategy page
  --quick                                 1,000 Monte Carlo / bootstrap draws, 2 seeds
                                          (for a fast look; never for a published number)
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import engine as E
import fivetests as FT
import render as RD

HERE = Path(__file__).resolve().parent


def parse(argv):
    p = argparse.ArgumentParser(
        prog="pipeline.py", description="Five-tests report for one strategy (or all).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("strategies", nargs="+", metavar="STRATEGY",
                   help="one or more of: " + ", ".join(E.SUBS) + ", or 'all'")
    p.add_argument("--rung", default="cross_fee", choices=FT.RUNGS,
                   help="execution basis the page opens on (default: cross_fee)")
    p.add_argument("--out", default=str(FT.ROOT / "reports"),
                   help="output directory (default: <repo>/reports)")
    p.add_argument("--refresh", action="store_true", help="ignore the cache")
    p.add_argument("--no-embed-fonts", action="store_true",
                   help="link Google Fonts instead of inlining the woff2 files")
    p.add_argument("--artifact", action="store_true",
                   help="also write a wrapper-less copy for publishing as an Artifact")
    p.add_argument("--combined-only", action="store_true",
                   help="with 'all', write only the combined seven-strategy page")
    p.add_argument("--quick", action="store_true",
                   help="1,000 draws and 2 seeds — a fast look, not a publishable number")
    a = p.parse_args(argv[1:])

    keys = list(dict.fromkeys(
        sum([list(E.SUBS) if s == "all" else [s] for s in a.strategies], [])))
    bad = [k for k in keys if k not in E.SUBS]
    if bad:
        p.error(f"unknown strategy {bad}; choose from {', '.join(E.SUBS)} or 'all'")
    a.keys = keys
    a.want_combined = any(s == "all" for s in a.strategies)
    return a


def main(argv=None):
    a = parse(argv or sys.argv)
    t0 = time.time()
    out_dir = Path(a.out)

    # A single-strategy report contains only that strategy, so only the
    # strategies asked for are computed. The combined page needs all seven.
    needed = list(E.SUBS) if (a.want_combined and not a.combined_only) else list(a.keys)
    if a.combined_only:
        needed = list(E.SUBS)
    print(f"five-tests pipeline · subject{'s' if len(a.keys)>1 else ''}: "
          f"{', '.join(a.keys)} · basis {a.rung}"
          + (" · QUICK MODE (1,000 draws)" if a.quick else ""))

    kw = dict(b_boot=1000, b_mc=1000, seeds=FT.SEEDS[:2]) if a.quick else {}
    ctx = None
    records = {}
    for k in needed:
        cached = None if a.refresh else (None if FT.cache_stale(k) else FT.cache_read(k))
        if cached is not None:
            records[k] = cached
            print(f"  {k:13s} from cache")
            continue
        if ctx is None:
            ctx = FT.Context(**kw)
        records[k] = ctx.compute_and_cache(k) if hasattr(ctx, "compute_and_cache") \
            else _compute_and_cache(ctx, k)

    written = []
    if not a.combined_only:
        for k in a.keys:
            p = RD.write(records, k, out_dir, rung=a.rung,
                         embed_fonts=not a.no_embed_fonts)
            written.append(p)
            if a.artifact:
                written.append(RD.write(records, k, out_dir, rung=a.rung,
                                        embed_fonts=False, standalone=False,
                                        suffix="-artifact"))
    if a.want_combined:
        p = RD.write(records, None, out_dir, rung=a.rung,
                     embed_fonts=not a.no_embed_fonts)
        written.append(p)
        if a.artifact:
            written.append(RD.write(records, None, out_dir, rung=a.rung,
                                    embed_fonts=False, standalone=False,
                                    suffix="-artifact"))

    import verdicts as VD
    print(f"\n  {'strategy':14s}{'OVERALL':<9s}" + "".join(f"{n.split()[0][:10]:>11s}"
                                                           for _i, n, _f in VD.TESTS))
    for k in a.keys:
        v = records[k]["verdicts"][a.rung]
        print(f"  {records[k]['name']:14s}{v['overall']:<9s}"
              + "".join(f"{v[i]:>11s}" for i, _n, _f in VD.TESTS))
        for i, n, _f in VD.TESTS:
            print(f"      {n:22s} {v[i]:<8s} {records[k]['why'][a.rung][i]}")
    print()
    for p in written:
        print(f"  wrote {p}  ({p.stat().st_size/1024:.0f} KB)")
    print(f"\ndone in {time.time()-t0:.0f}s")
    return written


def _compute_and_cache(ctx, key):
    rec = FT.compute(ctx, key)
    ctx.cache_write(key, rec)
    return rec


if __name__ == "__main__":
    main()
