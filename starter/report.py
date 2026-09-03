"""The build target: an automated validation report.

One pipeline, run unchanged over all seven submissions:

    report = run(registry_entry, blotter, market)
    report.verdict  ->  "PASS" | "SUSPECT" | "FAIL"

This file defines the contract only. The six sections — what each one should
compute — are the six-sections handout; the evidence behind every verdict
line is the point of the exercise. Nothing below does any validation yet.
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


def run(registry_entry: dict, blotter: pd.DataFrame, market: dict) -> Report:
    """Take one submission's records, return the verdict with evidence.

    Suggested shape: one function per section, each returning a
    SectionResult; this function assembles them. Start with the sections
    that are pure record arithmetic and grow from there.
    """
    raise NotImplementedError("this is the hackathon")
