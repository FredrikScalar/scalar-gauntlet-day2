#!/usr/bin/env python3
"""One command, one strategy: compute the five tests and write its report.

    python3 pipeline.py --check                # verify deps and data, compute nothing
    python3 pipeline.py windfall               # reports/windfall-five-tests.html
    python3 pipeline.py windfall windfall2     # two individual reports
    python3 pipeline.py all                    # one individual report per strategy

What it does, in order:
  1. loads the market once (book_snapshots.csv, products, registry, blotters)
  2. computes the record for each strategy asked for and caches it in cache/
  3. renders reports/<strategy>-five-tests.html — one file per strategy

**Every report is about one strategy only.** No other submission appears in it,
or in its embedded payload. Instead of comparing strategies with each other,
every chart compares the strategy against ITSELF across the three execution
bases, which is where the friction story lives.

The combined seven-strategy page is opt-in and is the only output where other
submissions appear:

    python3 pipeline.py all --combined         # seven individual + the combined page
    python3 pipeline.py all --combined-only    # only the combined page

Everything is recomputed from raw data — no earlier payload is read. Records are
cached, so a second run on the same strategy renders in under a second; a change
to the data, to luck.py, engine.py, trips.py, verdicts.py or fivetests.py
invalidates the cache automatically.

Requirements: Python 3.9+, numpy, pandas, scipy, PyYAML (see requirements.txt).

Options
  --check                                 preflight only: deps, data, fonts, cache
  --rung {mid,passive_in_fee,cross_fee}   basis the page opens on (default cross_fee)
  --out DIR                               output directory (default <repo>/reports)
  --refresh                               recompute even if the cache looks fresh
  --no-embed-fonts                        link Google Fonts instead of inlining woff2
  --artifact                              also write a wrapper-less copy for publishing
  --combined                              also write the combined seven-strategy page
  --combined-only                         write only the combined seven-strategy page
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
    p.add_argument("strategies", nargs="*", metavar="STRATEGY",
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
    p.add_argument("--combined", action="store_true",
                   help="also write the combined seven-strategy page (needs all seven computed)")
    p.add_argument("--combined-only", action="store_true",
                   help="write only the combined seven-strategy page")
    p.add_argument("--check", action="store_true",
                   help="preflight only: check dependencies, data, fonts and cache, then exit")
    p.add_argument("--quick", action="store_true",
                   help="1,000 draws and 2 seeds — a fast look, not a publishable number")
    a = p.parse_args(argv[1:])

    if a.check:
        a.keys, a.want_combined = [], False
        return a
    if not a.strategies:
        p.error("name at least one strategy (or 'all'), or pass --check")

    keys = list(dict.fromkeys(
        sum([list(E.SUBS) if s == "all" else [s] for s in a.strategies], [])))
    bad = [k for k in keys if k not in E.SUBS]
    if bad:
        p.error(f"unknown strategy {bad}; choose from {', '.join(E.SUBS)} or 'all'")
    a.keys = keys
    # The combined page is opt-in. `all` on its own means one report per
    # strategy, each about that strategy alone.
    a.want_combined = a.combined or a.combined_only
    return a


def preflight() -> int:
    """Report on everything a run needs, without computing anything."""
    ok = True
    print("five-tests pipeline · preflight\n")
    print(f"  python                {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        print("    ^ needs 3.9 or newer"); ok = False
    for mod, floor in (("numpy", "1.22"), ("pandas", "1.4"),
                       ("scipy", "1.8"), ("yaml", "5.4")):
        try:
            m = __import__(mod)
            print(f"  {mod:21s} {getattr(m, '__version__', '?')}   (floor {floor})")
        except ImportError:
            name = "PyYAML" if mod == "yaml" else mod
            print(f"  {mod:21s} MISSING — pip install {name}>={floor}")
            ok = False

    print(f"\n  repo root             {FT.ROOT}")
    for rel in ("data/book_snapshots.csv", "data/products.csv"):
        p = FT.ROOT/rel
        print(f"  {rel:21s} {'%.0f MB' % (p.stat().st_size/1e6) if p.exists() else 'MISSING'}")
        ok &= p.exists()
    missing = [k for k in E.SUBS if not (FT.ROOT/"blotters"/f"{k}-blotter.csv").exists()]
    print(f"  blotters              {len(E.SUBS)-len(missing)}/{len(E.SUBS)} present"
          + (f" — missing {missing}" if missing else ""))
    ok &= not missing
    missing = [k for k in E.SUBS if not (FT.ROOT/"registry"/f"{k}.yaml").exists()]
    print(f"  registry              {len(E.SUBS)-len(missing)}/{len(E.SUBS)} present"
          + (f" — missing {missing}" if missing else ""))
    ok &= not missing

    for f in ("template_one.html", "template_all.html", "report.css",
              "fivetests.py", "verdicts.py", "render.py"):
        if not (HERE/f).exists():
            print(f"  {f:21s} MISSING"); ok = False
    fonts = RD.embedded_fonts()
    print(f"  fonts/                {'9 faces — reports will be fully offline' if fonts else 'not found — reports will link Google Fonts'}")

    fresh = [k for k in E.SUBS if not FT.cache_stale(k) and FT.cache_read(k) is not None]
    print(f"  cache/                {len(fresh)}/{len(E.SUBS)} usable"
          + (f" (stale or absent: {[k for k in E.SUBS if k not in fresh]})" if len(fresh) < len(E.SUBS) else ""))
    if len(fresh) < len(E.SUBS):
        print("                        cold strategies recompute on first use "
              "(~15 s each; pingpong ~4½ min)")

    print("\n  " + ("ready — try: python3 pipeline.py windfall" if ok
                    else "NOT ready — fix the lines marked MISSING above"))
    return 0 if ok else 1


def main(argv=None):
    a = parse(argv or sys.argv)
    if a.check:
        sys.exit(preflight())
    t0 = time.time()
    out_dir = Path(a.out)

    # A single-strategy report contains only that strategy, so only the
    # strategies asked for are computed. The combined page needs all seven.
    needed = list(E.SUBS) if a.want_combined else list(a.keys)
    what = ("the combined seven-strategy page only" if a.combined_only
            else ("individual reports: " + ", ".join(a.keys)
                  + (" + the combined page" if a.combined else "")))
    print(f"five-tests pipeline · {what} · basis {a.rung}"
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
