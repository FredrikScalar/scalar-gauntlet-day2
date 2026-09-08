"""The build target: an automated validation report.

One pipeline, run unchanged over all seven submissions:

    report = run(registry_entry, blotter, market)
    report.verdict  ->  "PASS" | "SUSPECT" | "FAIL"

    python report.py        # the scoreboard and every verdict line

This file defines the contract and assembles the sections that speak it.
Three do so far — `luck` (luck_section.py), `friction` (sections.py) and
`shelf_life` (shelf_life.py). The grid shows `lineage`, `evidence` and
`warranty` as '·' rather than letting an unbuilt section score a silent
PASS, so read `Report.verdict` as the verdict of the sections that ran.

`luck` reads the record `luck/pipeline.py` caches rather than recomputing
it, so build that cache first:

    python starter/luck/pipeline.py all
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

SECTIONS = ["luck", "lineage", "friction", "shelf_life", "evidence", "warranty"]

VERDICTS = ("PASS", "SUSPECT", "FAIL")


@dataclass
class Finding:
    """One test: a name, a number (or figure path), a verdict, a sentence."""
    name: str
    value: object
    verdict: str = "INFO"          # PASS | SUSPECT | FAIL | INFO
    note: str = ""


@dataclass
class SectionResult:
    section: str                   # one of SECTIONS
    verdict: str                   # PASS | SUSPECT | FAIL
    findings: list[Finding] = field(default_factory=list)


@dataclass
class Report:
    submission: str
    sections: list[SectionResult] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)   # mandatory if SUSPECT

    @property
    def verdict(self) -> str:
        """Aggregation rule: any section FAIL fails the submission; any
        SUSPECT (and no FAIL) makes it SUSPECT — and a SUSPECT report with an
        empty conditions list is an unfinished report."""
        vs = [s.verdict for s in self.sections]
        if "FAIL" in vs:
            return "FAIL"
        if "SUSPECT" in vs:
            return "SUSPECT"
        return "PASS"


def grid(reports: dict[str, Report],
         submissions: list[str] | None = None) -> pd.DataFrame:
    """The scoreboard: one row per submission, one column per section, plus
    the overall verdict. Unfilled cells show '·'. The day fills this in;
    the reveal judges it."""
    subs = submissions if submissions is not None else sorted(reports)
    rows = {}
    for sub in subs:
        row = {s: "·" for s in SECTIONS} | {"OVERALL": "·"}
        if sub in reports:
            r = reports[sub]
            row.update({s.section: s.verdict for s in r.sections})
            row["OVERALL"] = r.verdict
        rows[sub] = row
    return pd.DataFrame(rows).T[SECTIONS + ["OVERALL"]]


# Each built section: the callable returning its SectionResult, and the one
# that reads the conditions a SUSPECT must carry off that result. Sections
# not listed here stay '·' in the grid rather than silently scoring PASS.
def _builders():
    """Imported on call, not at module scope: every section module imports
    this one, so a top-level import would close the cycle."""
    import luck_section as luck_mod
    import sections as friction_mod
    import shelf_life as shelf_mod

    # Every builder takes (entry, blotter, market, tape) so `run` can call
    # them uniformly; luck and shelf-life ignore the tape they are handed.
    return [
        (luck_mod.luck, luck_mod.conditions_for),
        (friction_mod.friction, friction_mod.conditions_for),
        (lambda e, b, m, tape=None: shelf_mod.shelf_life(e, b, m),
         lambda sec: shelf_mod._suggested_conditions(sec.verdict,
                                                     sec.findings)),
    ]


def run(registry_entry: dict, blotter: pd.DataFrame, market: dict,
        tape=None) -> Report:
    """Take one submission's records, return the verdict with evidence.

    One function per section, each returning a SectionResult; this assembles
    them. `luck` and `lineage` do not speak this contract yet, so a report
    from here carries `friction` and `shelf_life` and the grid shows the
    others as unbuilt. Read `Report.verdict` as the verdict of the sections
    that ran, not of all six.

    `tape` is the executed trade tape, which only friction needs; it loads
    from data/ on demand when not supplied.
    """
    sections, conditions = [], []
    for build, conds in _builders():
        sec = build(registry_entry, blotter, market, tape=tape)
        sections.append(sec)
        if sec.verdict == "SUSPECT":
            conditions.extend(conds(sec))

    report = Report(submission=registry_entry["submission"],
                    sections=sections, conditions=conditions)
    # A SUSPECT report with no conditions is an unfinished report; say so
    # loudly here rather than letting it reach a funding decision.
    if report.verdict == "SUSPECT" and not report.conditions:
        raise ValueError(f"{report.submission}: SUSPECT with no conditions - "
                         "a section returned SUSPECT but named no condition")
    return report


def run_all(registry: dict, market: dict, blotter_dir="../blotters",
            tape=None) -> dict[str, Report]:
    """The same pipeline over every submission, unchanged."""
    from pathlib import Path

    from repricer import load_blotter

    out = {}
    for key, entry in sorted(registry.items()):
        out[key] = run(entry, load_blotter(Path(blotter_dir) /
                                           f"{key}-blotter.csv"), market, tape)
    return out


if __name__ == "__main__":
    import sys

    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root / "registry"))
    import execution as _ex                                   # noqa: E402
    from loader import load_registry                          # noqa: E402
    from repricer import load_market                          # noqa: E402

    _reports = run_all(load_registry(_root / "registry"),
                       load_market(_root / "data"),
                       _root / "blotters",
                       _ex.Tape.load(_root / "data"))
    print(grid(_reports).to_string())
    print()
    for _key, _r in sorted(_reports.items()):
        print(f"{_key}  {_r.verdict}")
        for _sec in _r.sections:
            print(f"    {_sec.verdict:<8} {_sec.section}")
            for _f in _sec.findings:
                print(f"        {_f.verdict:<8} {_f.name:<34} {_f.note}")
        for _c in _r.conditions:
            print(f"    CONDITION  {_c}")
        print()
