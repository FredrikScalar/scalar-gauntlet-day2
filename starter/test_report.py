"""Tests for `report.run` - the assembly of sections into one verdict.

The section builders are replaced with stubs, so a failure points at the
assembly rather than at friction or shelf-life.
"""
from __future__ import annotations

import pandas as pd
import pytest

import report
from report import Finding, Report, SectionResult

ENTRY = {"submission": "test"}


def _stub(section: str, verdict: str, conds: list[str] | None = None):
    """A (builder, conditions) pair in the shape `_builders` yields."""
    result = SectionResult(section, verdict,
                           [Finding(f"{section}_finding", 1.0, verdict, "note")])
    return (lambda e, b, m, tape=None: result, lambda sec: list(conds or []))


@pytest.fixture
def builders(monkeypatch):
    """Install stub builders; the test says which sections exist."""
    def install(*pairs):
        monkeypatch.setattr(report, "_builders", lambda: list(pairs))
    return install


# ---- assembly --------------------------------------------------------------

def test_run_collects_every_built_section(builders):
    builders(_stub("friction", "KEEP"), _stub("shelf_life", "KEEP"))
    r = report.run(ENTRY, pd.DataFrame(), {})
    assert [s.section for s in r.sections] == ["friction", "shelf_life"]
    assert r.verdict == "KEEP"
    assert r.submission == "test"


def test_any_fail_fails_the_submission(builders):
    builders(_stub("friction", "FAIL"), _stub("shelf_life", "KEEP"))
    assert report.run(ENTRY, pd.DataFrame(), {}).verdict == "FAIL"


def test_a_fail_outranks_a_suspect(builders):
    builders(_stub("friction", "FAIL"),
             _stub("shelf_life", "SUSPECT", ["watch it"]))
    assert report.run(ENTRY, pd.DataFrame(), {}).verdict == "FAIL"


# ---- conditions ------------------------------------------------------------

def test_conditions_are_gathered_from_the_suspect_sections(builders):
    builders(_stub("friction", "SUSPECT", ["cap the clip"]),
             _stub("shelf_life", "SUSPECT", ["paper-trade a quarter"]))
    r = report.run(ENTRY, pd.DataFrame(), {})
    assert r.conditions == ["cap the clip", "paper-trade a quarter"]


def test_a_passing_section_contributes_no_conditions(builders):
    builders(_stub("friction", "KEEP", ["never asked for"]),
             _stub("shelf_life", "SUSPECT", ["paper-trade a quarter"]))
    assert report.run(ENTRY, pd.DataFrame(), {}).conditions == \
        ["paper-trade a quarter"]


def test_suspect_without_conditions_is_refused(builders):
    """`Report` calls that an unfinished report; `run` refuses to hand one out
    rather than letting it reach a funding decision."""
    builders(_stub("friction", "SUSPECT", []))
    with pytest.raises(ValueError, match="SUSPECT with no conditions"):
        report.run(ENTRY, pd.DataFrame(), {})


def test_a_failing_report_needs_no_conditions(builders):
    """A FAIL is not fundable, so there is nothing to condition."""
    builders(_stub("friction", "FAIL", []))
    assert report.run(ENTRY, pd.DataFrame(), {}).verdict == "FAIL"


# ---- the tape reaches only the section that needs it -----------------------

def test_every_builder_is_called_with_the_tape(builders):
    """Uniform call signature - shelf-life ignores the tape it is handed."""
    seen = {}

    def spy(name):
        def build(e, b, m, tape=None):
            seen[name] = tape
            return SectionResult(name, "KEEP", [])
        return (build, lambda sec: [])

    builders(spy("friction"), spy("shelf_life"))
    report.run(ENTRY, pd.DataFrame(), {}, tape="TAPE")
    assert seen == {"friction": "TAPE", "shelf_life": "TAPE"}


# ---- the grid --------------------------------------------------------------

def test_grid_shows_unbuilt_sections_as_unbuilt(builders):
    builders(_stub("friction", "KEEP"), _stub("shelf_life", "KEEP"))
    g = report.grid({"test": report.run(ENTRY, pd.DataFrame(), {})})
    assert g.loc["test", "friction"] == "KEEP"
    assert g.loc["test", "shelf_life"] == "KEEP"
    for unbuilt in ("luck", "lineage", "evidence", "warranty"):
        assert g.loc["test", unbuilt] == "·"


def test_grid_keeps_the_declared_section_order():
    g = report.grid({"x": Report(submission="x")})
    assert list(g.columns) == report.SECTIONS + ["OVERALL"]


def test_run_all_runs_the_same_pipeline_over_every_submission(builders,
                                                              monkeypatch):
    builders(_stub("friction", "KEEP"), _stub("shelf_life", "KEEP"))
    monkeypatch.setattr("repricer.load_blotter", lambda p: pd.DataFrame())
    registry = {"a": {"submission": "a"}, "b": {"submission": "b"}}
    out = report.run_all(registry, {}, "../blotters")
    assert sorted(out) == ["a", "b"]
    assert all(r.verdict == "KEEP" for r in out.values())
