"""Tests for the bridge from `verdicts` into the `report.py` contract.

No market data: the checks are hand-built, so a failure points at the
mapping rather than at the numbers.
"""
from __future__ import annotations

import pandas as pd
import pytest

import report
import sections as sc
import friction_verdicts as vd
from friction_verdicts import FAIL, KEEP, SUSPECT


def _checks(**verdicts) -> list[vd.Check]:
    """One Check per name in `vd.CHECKS`, KEEP unless overridden."""
    return [vd.Check(n, verdicts.get(n, KEEP), f"{n} note", {"n": n})
            for n in vd.CHECKS]


# ---- the section ----------------------------------------------------------

def test_section_is_named_friction():
    assert sc.to_section(_checks()).section == "friction"
    assert "friction" in report.SECTIONS


def test_section_verdict_is_the_worst_check():
    assert sc.to_section(_checks()).verdict == KEEP
    assert sc.to_section(_checks(edge=SUSPECT)).verdict == SUSPECT
    assert sc.to_section(_checks(edge=SUSPECT, absorb=FAIL)).verdict == FAIL


def test_every_check_becomes_a_finding_that_keeps_its_numbers():
    sec = sc.to_section(_checks(touch=FAIL))
    assert [f.name for f in sec.findings] == list(vd.CHECKS)
    assert all(isinstance(f, report.Finding) for f in sec.findings)
    touch = next(f for f in sec.findings if f.name == "touch")
    assert touch.verdict == FAIL
    assert touch.value == {"n": "touch"}      # the evidence rides along


# ---- conditions: a SUSPECT report must carry them -------------------------

def test_a_pass_carries_no_conditions():
    assert vd.conditions(_checks()) == []


def test_each_suspect_check_names_its_condition():
    assert len(vd.conditions(_checks(edge=SUSPECT))) == 1
    assert len(vd.conditions(_checks(edge=SUSPECT, absorb=SUSPECT))) == 2


def test_a_fail_does_not_generate_a_condition():
    """Conditions are what makes a SUSPECT fundable. A FAIL is not."""
    assert vd.conditions(_checks(edge=FAIL)) == []


# `mid` is the one check with no SUSPECT branch: the recorded fill either
# makes money or it does not, and there is no condition that rescues it.
PASS_FAIL_ONLY = {"mid"}


def test_every_check_that_can_be_suspect_has_a_condition():
    """Otherwise `report.Report` would call the result unfinished."""
    for name in set(vd.CHECKS) - PASS_FAIL_ONLY:
        assert vd.conditions(_checks(**{name: SUSPECT})), (
            f"{name} can be SUSPECT but names no condition")


def test_mid_never_returns_suspect():
    """The premise of PASS_FAIL_ONLY, pinned rather than assumed."""
    for pnl in (-1.0, 0.0, 1.0, 1e9):
        assert vd.check_mid(pnl).verdict in (KEEP, FAIL)


def test_conditions_are_declared_for_real_checks_only():
    assert set(vd.CONDITIONS) <= set(vd.CHECKS)


# ---- the report contract --------------------------------------------------

def _report(**verdicts) -> report.Report:
    checks = _checks(**verdicts)
    return report.Report(submission="test", sections=[sc.to_section(checks)],
                         conditions=vd.conditions(checks))


def test_report_verdict_follows_the_friction_section():
    assert _report().verdict == KEEP
    assert _report(absorb=SUSPECT).verdict == SUSPECT
    assert _report(touch=FAIL).verdict == FAIL


def test_a_suspect_report_is_never_left_without_conditions():
    r = _report(absorb=SUSPECT)
    assert r.verdict == SUSPECT and r.conditions


def test_grid_fills_friction_and_leaves_the_rest_unbuilt():
    g = report.grid({"test": _report(edge=SUSPECT)})
    assert g.loc["test", "friction"] == SUSPECT
    assert g.loc["test", "OVERALL"] == SUSPECT
    assert g.loc["test", "luck"] == "·"


def test_scoreboard_puts_the_worst_first():
    board = sc.scoreboard({"good": _report(),
                           "bad": _report(touch=FAIL),
                           "iffy": _report(absorb=SUSPECT)})
    assert list(board.index) == ["good", "iffy", "bad"]


# ---- headline: one definition, shared with the HTML report ----------------

def test_headline_reads_the_crossing_table_and_the_daily_series():
    crossing = pd.DataFrame(
        {"mid": {"edge_eur_per_mwh": 2.0, "half_spread_eur_mwh": 0.4},
         "cross": {"edge_eur_per_mwh": 1.6, "half_spread_eur_mwh": 0.4}})
    idx = pd.date_range("2025-01-01", periods=3, tz="UTC")
    daily = {"mid": pd.Series([10.0, 20.0, 30.0], index=idx),
             "cross": pd.Series([5.0, 10.0, 15.0], index=idx),
             "sweep": pd.Series([4.0, 8.0, 12.0], index=idx)}
    h = sc.headline(crossing, daily)
    assert h["pnl_mid"] == pytest.approx(60.0)
    assert h["pnl_touch"] == pytest.approx(30.0)   # `cross` IS the touch
    assert h["pnl_sweep"] == pytest.approx(24.0)
    assert h["edge_mid"] == pytest.approx(2.0)
    assert h["half_spread"] == pytest.approx(0.4)


def test_headline_supplies_every_key_assess_reads():
    crossing = pd.DataFrame(
        {"mid": {"edge_eur_per_mwh": 2.0, "half_spread_eur_mwh": 0.4}})
    idx = pd.date_range("2025-01-01", periods=2, tz="UTC")
    s = pd.Series([1.0, 2.0], index=idx)
    stats = {"clip_mw": 10.0, "p99_print_mw": 20.0, "median_print_mw": 4.5,
             "median_absorb_min": 10.0, "hold_median_min": 60.0}
    checks = vd.assess(sc.headline(crossing, dict.fromkeys(
        ("mid", "cross", "sweep"), s)), stats)
    assert [c.name for c in checks] == list(vd.CHECKS)
