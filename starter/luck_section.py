"""Luck - section 1 - in the shape `report.py` defines.

`luck/` already computes the five tests and already decides PASS / SUSPECT /
FAIL in `luck/verdicts.py`. Nothing here re-judges any of that; this is the
adapter that hands its answer to `report.run` as a `SectionResult`, so the
luck column stops reading '·'.

The five tests come from one expensive record per submission (Monte Carlo
and bootstrap draws), which `luck/pipeline.py` computes and caches. This
module READS that cache and never computes it, so a report run stays fast
and never silently produces a number nobody published:

    python starter/luck/pipeline.py all      # build the cache (slow, once)
    python starter/report.py                 # reads it

A cached record whose inputs have since changed is still used, but the
section says so in an INFO finding rather than quietly trusting it.

    luck(registry_entry, blotter, market) -> SectionResult
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE / "luck") not in sys.path:
    sys.path.insert(0, str(_HERE / "luck"))

import fivetests as FT                                        # noqa: E402
import verdicts as LV                                         # noqa: E402
from report import Finding, SectionResult                     # noqa: E402

# The execution basis the section is judged on. `cross_fee` is the honest
# one - it pays the spread and the fee - and is what `luck/pipeline.py`
# opens on. (The coin-flip rule scores at mid regardless; see LV.coinflip.)
RUNG = "cross_fee"

# What a SUSPECT on each of the five tests obliges us to do.
CONDITIONS = {
    "coinflip": ("re-run the coin-flip null on more seeds before funding; "
                 "the best-of-10,000 verdict is not stable"),
    "bootstrap": ("size to the lower bound of the resample interval, not to "
                  "the point estimate"),
    "dsr": ("get the true trial count in writing - the deflated Sharpe "
            "breaks near the number declared"),
    "days": ("cap size until the P&L stops depending on a handful of days; "
             "re-review after one more quarter"),
    "baseline": ("hold breadth neutral in review - the edge is close to what "
                 "a signal-free benchmark earns"),
}


def _record(key: str) -> dict:
    rec = FT.cache_read(key)
    if rec is None:
        raise FileNotFoundError(
            f"no luck record cached for {key!r}. Build it first:\n"
            f"    python {Path('starter/luck/pipeline.py')} {key}")
    return rec


def luck(registry_entry: dict, blotter: pd.DataFrame, market: dict,
         tape=None, rung: str = RUNG) -> SectionResult:
    """The section contract: five findings and one verdict.

    `blotter`, `market` and `tape` are unused - luck reads its own cached
    record - but the signature matches every other section so `report.run`
    can call them all the same way.
    """
    key = registry_entry["submission"]
    rec = _record(key)
    scored, why = LV.score(rec, rung)

    findings = [Finding(name=tid, value={"rung": rung, "test": name},
                        verdict=scored[tid], note=why[tid])
                for tid, name, _fn in LV.TESTS]

    if FT.cache_stale(key):
        findings.append(Finding(
            name="cache_freshness", value={"stale": True}, verdict="INFO",
            note="the cached luck record predates its inputs - rebuild with "
                 f"`python starter/luck/pipeline.py {key}` to be sure"))

    return SectionResult("luck", scored["overall"], findings)


def conditions_for(section: SectionResult) -> list[str]:
    """What a SUSPECT luck section obliges us to do."""
    return [CONDITIONS[f.name] for f in section.findings
            if f.verdict == LV.SUSPECT and f.name in CONDITIONS]
