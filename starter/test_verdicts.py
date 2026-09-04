"""Tests for the threshold layer - the only place a line gets drawn."""
from __future__ import annotations

import numpy as np
import pytest

import verdicts as v
from verdicts import FAIL, KEEP, SUSPECT


def _daily(pnl_mid=1000.0, sh_mid=2.0, pnl_touch=900.0, sh_touch=1.8,
           pnl_sweep=900.0, sh_sweep=1.8, edge_mid=2.0, half_spread=0.4,
           fee=0.1):
    return {"pnl_mid": pnl_mid, "sharpe_mid": sh_mid,
            "pnl_touch": pnl_touch, "sharpe_touch": sh_touch,
            "pnl_sweep": pnl_sweep, "sharpe_sweep": sh_sweep,
            "edge_mid": edge_mid, "half_spread": half_spread, "fee": fee}


def _stats(clip=10.0, p99=20.0, med_print=4.5, absorb=10.0, hold=60.0):
    return {"clip_mw": clip, "p99_print_mw": p99, "median_print_mw": med_print,
            "median_absorb_min": absorb, "hold_median_min": hold}


# ---- mid -------------------------------------------------------------------

def test_mid_must_make_money():
    assert v.check_mid(1.0).verdict == KEEP
    assert v.check_mid(0.0).verdict == FAIL
    assert v.check_mid(-5.0).verdict == FAIL


# ---- touch, and the fail/suspect split -------------------------------------

def test_touch_passes_when_both_metrics_hold_up():
    assert v.check_touch(900, 1.8, 1000, 2.0).verdict == KEEP


def test_touch_fails_on_negative_pnl():
    assert v.check_touch(-1, 1.9, 1000, 2.0).verdict == FAIL


def test_touch_fails_when_both_lose_more_than_half():
    c = v.check_touch(400, 0.9, 1000, 2.0)
    assert c.verdict == FAIL
    assert "both" in c.note


def test_touch_suspect_when_only_sharpe_collapses():
    """P&L above half and positive, Sharpe down by half or more."""
    c = v.check_touch(900, 0.9, 1000, 2.0)
    assert c.verdict == SUSPECT
    assert "Sharpe keeps only" in c.note


def test_touch_suspect_when_only_pnl_collapses():
    c = v.check_touch(400, 1.9, 1000, 2.0)
    assert c.verdict == SUSPECT
    assert "P&L keeps only" in c.note


def test_exactly_half_retained_is_a_drop():
    """`more than half` lost means keeping strictly less than half is bad;
    keeping exactly half is not."""
    assert v.check_touch(500, 1.0, 1000, 2.0).verdict == KEEP
    assert v.check_touch(499, 0.99, 1000, 2.0).verdict == FAIL


# ---- sweep adds a comparison against the touch -----------------------------

def test_sweep_matching_a_passing_touch_passes():
    c = v.check_sweep(900, 1.8, 1000, 2.0, 900, 1.8, KEEP)
    assert c.verdict == KEEP


def test_sweep_suspect_when_it_degrades_against_the_touch():
    """Fine against mid, but half its value is lost between touch and sweep -
    that is a size cost, not a spread cost."""
    c = v.check_sweep(600, 0.8, 1000, 2.0, 900, 1.8, KEEP)
    assert c.verdict == SUSPECT
    assert "too big for the book" in c.note


def test_sweep_inherits_a_suspect_touch_even_when_the_clip_fits():
    """Sweeping cannot be sounder than the fill it is built on."""
    c = v.check_sweep(900, 1.8, 1000, 2.0, 900, 1.8, SUSPECT)
    assert c.verdict == SUSPECT
    assert "touch itself is SUSPECT" in c.note


def test_sweep_size_cost_is_reported_over_the_inherited_touch():
    """Both apply; the size cost is the finding only sweep can make."""
    c = v.check_sweep(600, 0.8, 1000, 2.0, 900, 1.8, SUSPECT)
    assert c.verdict == SUSPECT
    assert "too big for the book" in c.note


def test_sweep_no_longer_goes_suspect_on_the_spread_alone():
    """One metric under 50% of mid is a SPREAD cost - `touch` judges that,
    and re-charging it here counted the same cost twice.

    Sweep is identical to the touch (the clip fits), so there is no size
    cost; Sharpe keeps only 40% of mid, but that loss happened getting to
    the touch, and the touch passed it. Sweep is KEEP.
    """
    c = v.check_sweep(900, 0.8, 1000, 2.0, 900, 0.8, KEEP)
    assert c.verdict == KEEP
    assert v.check_touch(900, 0.8, 1000, 2.0).verdict == SUSPECT  # the old path


def test_sweep_fail_against_mid_is_not_softened_by_the_touch_rule():
    c = v.check_sweep(-10, -1.0, 1000, 2.0, 900, 1.8, KEEP)
    assert c.verdict == FAIL


def test_sweep_records_what_it_kept_against_the_touch():
    c = v.check_sweep(600, 0.9, 1000, 2.0, 900, 1.8, KEEP)
    assert c.values["pnl_vs_touch"] == pytest.approx(600 / 900)
    assert c.values["sharpe_vs_touch"] == pytest.approx(0.5)
    assert c.values["touch_verdict"] == KEEP


def test_assess_feeds_the_touch_verdict_into_sweep():
    """The wiring, not the rule: sweep must see the verdict `touch` reached."""
    checks = v.assess(_daily(pnl_touch=400, sh_touch=0.7), _stats())
    touch = next(c for c in checks if c.name == "touch")
    sweep = next(c for c in checks if c.name == "sweep")
    assert touch.verdict == sweep.values["touch_verdict"]


# ---- absorption ------------------------------------------------------------

def test_clip_size_suspect_above_the_big_print_quantile():
    assert v.check_clip_size(10, 20, 4.5).verdict == KEEP
    assert v.check_clip_size(25, 20, 4.5).verdict == SUSPECT


def test_clip_size_reads_the_quantile_the_thresholds_declare():
    """`assess` must pull the stat `CLIP_QUANTILE` names, not a fixed p95."""
    key = f"p{v.CLIP_QUANTILE * 100:.0f}_print_mw"
    stats = {"clip_mw": 25.0, key: 20.0, "median_print_mw": 4.5,
             "median_absorb_min": 10.0, "hold_median_min": 60.0}
    clip = next(c for c in v.assess(_daily(), stats) if c.name == "clip_size")
    assert clip.verdict == SUSPECT


def test_clip_size_passes_when_the_quantile_is_unknown():
    assert v.check_clip_size(25, np.nan, 4.5).verdict == KEEP


def test_absorption_thresholds():
    assert v.check_absorption(20, 60).verdict == KEEP        # 0.33x
    assert v.check_absorption(40, 60).verdict == SUSPECT     # 0.67x
    assert v.check_absorption(90, 60).verdict == FAIL        # 1.5x


def test_absorption_fails_when_the_tape_never_gets_there():
    assert v.check_absorption(np.nan, 60).verdict == FAIL


# ---- aggregation -----------------------------------------------------------

def test_all_pass_is_a_pass():
    checks = v.assess(_daily(), _stats())
    assert v.overall(checks) == KEEP
    assert v.failing(checks) == []


def test_any_suspect_makes_it_suspect():
    checks = v.assess(_daily(), _stats(absorb=40, hold=60))
    assert v.overall(checks) == SUSPECT
    assert "absorb" in v.failing(checks)
    assert v.sections_failing(checks) == ["market absorption"]


def test_any_fail_makes_it_fail_and_names_the_section():
    checks = v.assess(_daily(pnl_mid=-1), _stats())
    assert v.overall(checks) == FAIL
    assert v.sections_failing(checks) == ["fill mode"]


def test_fail_outranks_suspect():
    checks = v.assess(_daily(pnl_touch=-1), _stats(clip=25, p99=20))
    assert v.overall(checks) == FAIL
    assert "clip_size" not in v.failing(checks)     # only the deciding ones


def test_grid_has_a_column_per_check_and_an_overall():
    g = v.grid({"a": v.assess(_daily(), _stats()),
                "b": v.assess(_daily(pnl_mid=-1), _stats())})
    assert list(g.columns) == list(v.CHECKS) + ["OVERALL"]
    assert g.loc["a", "OVERALL"] == KEEP
    assert g.loc["b", "OVERALL"] == FAIL


# ---- section rollup --------------------------------------------------------

def test_section_verdict_takes_the_worst_check_in_that_section():
    checks = v.assess(_daily(pnl_touch=-1), _stats())
    assert v.section_verdict(checks, "fill mode") == FAIL
    assert v.section_verdict(checks, "market absorption") == KEEP


def test_section_grid_has_a_column_per_section():
    g = v.section_grid({
        "clean": v.assess(_daily(), _stats()),
        "slow": v.assess(_daily(), _stats(absorb=90, hold=60)),
        "broken": v.assess(_daily(pnl_mid=-1), _stats()),
    })
    assert list(g.columns) == ["fill mode", "market absorption", "OVERALL"]
    assert g.loc["clean"].tolist() == [KEEP, KEEP, KEEP]
    assert g.loc["slow"].tolist() == [KEEP, FAIL, FAIL]
    assert g.loc["broken", "fill mode"] == FAIL
    assert g.loc["broken", "market absorption"] == KEEP


def test_section_grid_is_ordered_worst_last():
    g = v.section_grid({"bad": v.assess(_daily(pnl_mid=-1), _stats()),
                        "good": v.assess(_daily(), _stats())})
    assert list(g.index) == ["good", "bad"]


def test_every_check_belongs_to_a_named_section():
    assert set(v.SECTION.values()) == set(v.SECTIONS)


# ---- edge: the gross edge against the cost of earning it -------------------

def test_edge_fails_when_the_edge_is_inside_the_cost():
    """0.40 half-spread + 0.10 fee = 0.50 to trade a 0.45 edge."""
    assert v.check_edge(0.45, 0.40, fee=0.10).verdict == FAIL


def test_edge_suspect_when_friction_takes_most_of_it():
    c = v.check_edge(0.75, 0.40, fee=0.10)          # 1.5x coverage
    assert c.verdict == SUSPECT
    assert c.values["coverage"] == pytest.approx(1.5)


def test_edge_passes_with_room_to_spare():
    c = v.check_edge(2.0, 0.40, fee=0.10)           # 4.0x coverage
    assert c.verdict == KEEP
    assert c.values["edge_net"] == pytest.approx(1.5)


def test_edge_boundaries_are_inclusive_upward():
    """Exactly at a threshold is the kinder verdict."""
    assert v.check_edge(0.50, 0.40, fee=0.10).verdict == SUSPECT   # 1.0x
    assert v.check_edge(1.00, 0.40, fee=0.10).verdict == KEEP      # 2.0x


def test_edge_counts_the_fee_against_the_edge():
    """The fee is the whole reason this check is not a restatement of
    `touch`: the same edge and spread pass at zero fee and fail with one."""
    assert v.check_edge(0.55, 0.50, fee=0.0).verdict == SUSPECT
    assert v.check_edge(0.55, 0.50, fee=0.10).verdict == FAIL


def test_edge_fails_on_a_negative_edge():
    assert v.check_edge(-0.30, 0.45, fee=0.10).verdict == FAIL


def test_edge_at_zero_cost_cannot_be_scored():
    assert v.check_edge(1.0, 0.0, fee=0.0).verdict == FAIL


def test_edge_is_a_fill_mode_check():
    assert v.SECTION["edge"] == "fill mode"
    assert "edge" in v.CHECKS


def test_assess_runs_the_edge_check():
    checks = v.assess(_daily(edge_mid=0.45, half_spread=0.40, fee=0.10),
                      _stats())
    edge = next(c for c in checks if c.name == "edge")
    assert edge.verdict == FAIL
    assert v.overall(checks) == FAIL
    assert "edge" in v.failing(checks)
