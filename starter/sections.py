"""The friction section, in the shape `report.py` defines.

Three modules, three jobs, and the seam between them is the point:

    friction.py / execution.py   measure - no thresholds, no verdicts
    verdicts.py                  decide  - the only place a line is drawn
    sections.py                  report  - that verdict in the contract shape

What comes out is one `report.SectionResult` named `friction`, one
`report.Finding` per check, and - when the verdict is SUSPECT - the
conditions it must carry, because `report.Report` treats a SUSPECT with an
empty conditions list as an unfinished report.

The other five sections are not built yet, so `Report.verdict` here is the
friction verdict and nothing more. Say so when you quote it.

    python sections.py        # the scoreboard, all seven
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import execution as ex
import friction as fr
import report
import friction_verdicts as vd
from repricer import daily_pnl, load_blotter, load_market, sharpe

FILLS = ("mid", "cross", "sweep")


def headline(crossing: pd.DataFrame,
             daily: dict[str, pd.Series]) -> dict:
    """The numbers `verdicts.assess` reads, named as it expects them.

    `crossing` is a `friction.crossing_check` table, `daily` the mid/cross/
    sweep daily P&L series. One definition, so the HTML report and the
    contract report cannot drift apart on what "the touch" meant.
    """
    return {
        "pnl_mid": float(daily["mid"].sum()),
        "sharpe_mid": sharpe(daily["mid"]),
        "pnl_touch": float(daily["cross"].sum()),
        "sharpe_touch": sharpe(daily["cross"]),
        "pnl_sweep": float(daily["sweep"].sum()),
        "sharpe_sweep": sharpe(daily["sweep"]),
        "edge_mid": float(crossing.loc["edge_eur_per_mwh", "mid"]),
        "half_spread": float(crossing.loc["half_spread_eur_mwh", "mid"]),
    }


def measure(entry: dict, blotter: pd.DataFrame, market: dict,
            tape: ex.Tape) -> tuple[dict, dict]:
    """Everything the checks need, measured: `(headline, absorption stats)`."""
    window = (str(entry["backtest_window"]["start"]),
              str(entry["backtest_window"]["end"]))
    b = fr.attach_book(blotter, market)
    daily = {}
    for fill in FILLS:
        b[f"px_{fill}"] = fr.fill_price(b, fr.FILLS[fill])
        daily[fill] = daily_pnl(b, f"px_{fill}", window=window)
    return (headline(fr.crossing_check(entry, blotter, market), daily),
            ex.absorption_summary(entry, blotter, market, tape))


def to_section(checks: list[vd.Check]) -> report.SectionResult:
    """The checks as the contract's `friction` section.

    Each check's `values` ride along as the Finding's value, so every verdict
    line still points at the numbers that produced it.
    """
    return report.SectionResult(
        section="friction",
        verdict=vd.overall(checks),
        findings=[report.Finding(name=c.name, value=c.values,
                                 verdict=c.verdict, note=c.note)
                  for c in checks])


DATA = Path(__file__).resolve().parent.parent / "data"


def friction(registry_entry: dict, blotter: pd.DataFrame, market: dict,
             tape: ex.Tape | None = None) -> report.SectionResult:
    """The section contract, the same shape `shelf_life.shelf_life` returns.

    The tape is loaded on demand when not supplied, so this matches the
    three-argument section signature `report.run` calls everything with.
    """
    tape = ex.Tape.load(DATA) if tape is None else tape
    daily, stats = measure(registry_entry, blotter, market, tape)
    return to_section(vd.assess(daily, stats))


def conditions_for(section: report.SectionResult) -> list[str]:
    """What a SUSPECT friction section obliges us to do, read off its findings.

    `conditions` works from the checks; this works from the SectionResult, so
    `report.run` can ask without re-running anything.
    """
    return [vd.CONDITIONS[f.name] for f in section.findings
            if f.verdict == vd.SUSPECT and f.name in vd.CONDITIONS]


def run(entry: dict, blotter: pd.DataFrame, market: dict,
        tape: ex.Tape) -> report.Report:
    """One submission's report - friction only, for now."""
    daily, stats = measure(entry, blotter, market, tape)
    checks = vd.assess(daily, stats)
    return report.Report(submission=entry["submission"],
                         sections=[to_section(checks)],
                         conditions=vd.conditions(checks))


def run_all(registry: dict, market: dict, tape: ex.Tape,
            blotter_dir: str | Path = "../blotters"
            ) -> dict[str, report.Report]:
    """The same pipeline over every submission, unchanged."""
    out = {}
    for key, entry in sorted(registry.items()):
        blot = load_blotter(Path(blotter_dir) / f"{key}-blotter.csv")
        out[key] = run(entry, blot, market, tape)
    return out


def scoreboard(reports: dict[str, report.Report]) -> pd.DataFrame:
    """`report.grid`, ordered worst first so the arguments start at the top."""
    g = report.grid(reports)
    return g.sort_values("OVERALL", key=lambda s: s.map(vd.RANK).fillna(-1),
                         kind="stable")


def _text(reports: dict[str, report.Report]) -> str:
    """The scoreboard, then every verdict line and every condition."""
    out = [scoreboard(reports).to_string(), ""]
    for key, r in sorted(reports.items(),
                         key=lambda kv: -vd.RANK[kv[1].verdict]):
        out.append(f"{key}  {r.verdict}")
        for sec in r.sections:
            for f in sec.findings:
                out.append(f"    {f.verdict:<8} {f.name:<10} {f.note}")
        for c in r.conditions:
            out.append(f"    CONDITION  {c}")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "registry"))
    from loader import load_registry            # noqa: E402

    market = load_market(root / "data")
    reports = run_all(load_registry(root / "registry"), market,
                      ex.Tape.load(root / "data"), root / "blotters")
    print(_text(reports))
