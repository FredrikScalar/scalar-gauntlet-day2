"""Tests for the adapter that hands luck's answer to `report.run`.

The record and the scoring are luck's; these pin the ADAPTER - that it
reports what luck decided, unchanged, and refuses to invent an answer when
the cache is missing.
"""
from __future__ import annotations

import pandas as pd
import pytest

import luck_section as ls
from report import SectionResult

ENTRY = {"submission": "windfall"}
SCORED = {"coinflip": "KEEP", "bootstrap": "KEEP", "dsr": "SUSPECT",
          "days": "KEEP", "baseline": "KEEP", "overall": "SUSPECT"}
WHY = {k: f"because {k}" for k in SCORED}


@pytest.fixture
def luck_stub(monkeypatch):
    """Stand in for the cached record and luck's own scoring."""
    def install(scored=SCORED, stale=False, record={"schema": "x"}):
        monkeypatch.setattr(ls.FT, "cache_read", lambda k: record)
        monkeypatch.setattr(ls.FT, "cache_stale", lambda k: stale)
        monkeypatch.setattr(ls.LV, "score", lambda rec, rung: (scored, WHY))
    return install


# ---- it reports what luck decided, unchanged ------------------------------

def test_section_is_named_luck(luck_stub):
    luck_stub()
    assert ls.luck(ENTRY, pd.DataFrame(), {}).section == "luck"


def test_verdict_is_lucks_own_overall(luck_stub):
    luck_stub()
    assert ls.luck(ENTRY, pd.DataFrame(), {}).verdict == "SUSPECT"


def test_one_finding_per_luck_test_carrying_its_reason(luck_stub):
    luck_stub()
    sec = ls.luck(ENTRY, pd.DataFrame(), {})
    assert [f.name for f in sec.findings] == [t[0] for t in ls.LV.TESTS]
    dsr = next(f for f in sec.findings if f.name == "dsr")
    assert dsr.verdict == "SUSPECT"
    assert dsr.note == WHY["dsr"]


def test_the_adapter_does_not_re_judge(luck_stub):
    """Whatever luck says is what the section says - no second opinion."""
    luck_stub(scored=dict(SCORED, overall="FAIL"))
    assert ls.luck(ENTRY, pd.DataFrame(), {}).verdict == "FAIL"


def test_it_is_judged_on_the_declared_rung(luck_stub, monkeypatch):
    seen = {}
    monkeypatch.setattr(ls.FT, "cache_read", lambda k: {"schema": "x"})
    monkeypatch.setattr(ls.FT, "cache_stale", lambda k: False)
    monkeypatch.setattr(ls.LV, "score",
                        lambda rec, rung: (seen.setdefault("rung", rung),
                                           (SCORED, WHY))[1])
    ls.luck(ENTRY, pd.DataFrame(), {})
    assert seen["rung"] == ls.RUNG == "cross_fee"


# ---- a stale cache is disclosed, not silently trusted ---------------------

def test_a_fresh_cache_adds_no_extra_finding(luck_stub):
    luck_stub(stale=False)
    names = [f.name for f in ls.luck(ENTRY, pd.DataFrame(), {}).findings]
    assert "cache_freshness" not in names


def test_a_stale_cache_is_disclosed_as_info(luck_stub):
    luck_stub(stale=True)
    sec = ls.luck(ENTRY, pd.DataFrame(), {})
    note = next(f for f in sec.findings if f.name == "cache_freshness")
    assert note.verdict == "INFO"


def test_a_stale_cache_does_not_change_the_verdict(luck_stub):
    """INFO is not a verdict - staleness reports, it does not judge."""
    luck_stub(stale=True)
    assert ls.luck(ENTRY, pd.DataFrame(), {}).verdict == "SUSPECT"


def test_a_missing_cache_refuses_rather_than_guesses(luck_stub):
    luck_stub(record=None)
    with pytest.raises(FileNotFoundError, match="no luck record cached"):
        ls.luck(ENTRY, pd.DataFrame(), {})


# ---- conditions -----------------------------------------------------------

def test_conditions_come_from_the_suspect_tests(luck_stub):
    luck_stub()
    sec = ls.luck(ENTRY, pd.DataFrame(), {})
    assert ls.conditions_for(sec) == [ls.CONDITIONS["dsr"]]


def test_every_luck_test_names_a_condition():
    assert {t[0] for t in ls.LV.TESTS} == set(ls.CONDITIONS)


def test_an_info_finding_never_becomes_a_condition(luck_stub):
    luck_stub(scored=dict(SCORED, dsr="KEEP", overall="KEEP"), stale=True)
    assert ls.conditions_for(ls.luck(ENTRY, pd.DataFrame(), {})) == []


# ---- it fits the shape report.run calls ------------------------------------

def test_it_accepts_the_tape_every_builder_is_handed(luck_stub):
    luck_stub()
    sec = ls.luck(ENTRY, pd.DataFrame(), {}, tape="ignored")
    assert isinstance(sec, SectionResult)
