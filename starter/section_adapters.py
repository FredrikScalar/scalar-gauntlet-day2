"""Glue for section modules that do not implement `contribute()` themselves.

`report.py` asks each section for (SectionResult, body_html) in one call.
Luck and Shelf-life were written as standalone tools with their own page
generators, so this module adapts them without touching their code. When
either grows its own `contribute()`, delete the adapter here and call it
directly — nothing else changes.

WHY THE RICH PANELS ARE IFRAMED RATHER THAN INLINED
---------------------------------------------------
Four sections were authored independently, and inlining their pages into one
document would break them:

  * Shelf-life pins Plotly 2.35.2; the assembled report loads 4.0.0. Two
    majors on one page means the second load wins and the other section's
    charts break.
  * Luck ships its own 257-line stylesheet and hand-rolled SVG; its bare
    element selectors would restyle every other section.
  * Both carry interactive controls — rung toggles, rolling-window sliders,
    drop-best-days — with their own element ids.

An iframe gives each one its own document: its CSS, its Plotly version and
its buttons all keep working, and no section can break another. The uniform
part — section number, title, the handout's question, verdict badge and the
findings table — is rendered by `report.py` around the frame, so the report
still reads as one document.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from report import Finding, SectionResult, findings_table, panel

_HERE = Path(__file__).resolve().parent
_LUCK = _HERE / "luck"

# Luck's modules import each other by bare name.
if str(_LUCK) not in sys.path:
    sys.path.insert(0, str(_LUCK))

# Luck calls a pass KEEP; report.py's contract is PASS.
_LUCK_VERDICT = {"KEEP": "PASS", "SUSPECT": "SUSPECT", "FAIL": "FAIL"}

# The execution basis Luck's own pipeline defaults to. Friction is a separate
# section, so this is worth knowing: at `cross_fee` the Luck verdict already
# carries execution cost. Its page has the toggle; the reader can switch.
LUCK_RUNG = "cross_fee"


# ---------------------------------------------------------------- Luck
def luck_contribute(registry_entry: dict, blotter: pd.DataFrame, market: dict,
                    out_dir: str | Path | None = "reports",
                    rung: str = LUCK_RUNG) -> tuple[SectionResult, str]:
    """Luck's five tests, from its cached record.

    The record is read from `luck/cache/`, not recomputed: a full compute is
    10,000 Monte Carlo draws plus 10,000 bootstrap resamples per strategy on
    top of loading the book grid. `cache_stale()` is mtime-based, so a fresh
    checkout marks every record stale without the code having changed; the
    cache's own generation timestamp is carried into the findings instead of
    trusting that flag.
    """
    import fivetests as FT
    import render as RD

    key = str(registry_entry.get("submission", "?"))
    rec = FT.cache_read(key)
    if rec is None:                      # no usable cache: compute it properly
        ctx = FT.Context(quiet=True)
        rec = FT.load_or_compute(ctx, key, refresh=True)

    rung = rung if rung in rec.get("rungs_available", []) else "mid"
    verdicts, why = rec["verdicts"][rung], rec["why"][rung]

    findings = [
        Finding(f"luck_{tid}", None, _LUCK_VERDICT.get(verdicts[tid], "INFO"),
                f"{name}: {why[tid]}")
        for tid, name, _fn in _luck_tests()
    ]
    findings.append(Finding("luck_basis", rung, "INFO",
                            f"scored on the {rung} execution basis; record "
                            f"computed {rec.get('generated', 'unknown')}"))

    verdict = _LUCK_VERDICT.get(verdicts["overall"], "INFO")
    res = SectionResult("luck", verdict, findings)
    res.conditions = [f"Luck: {why[tid]}" for tid, _n, _f in _luck_tests()
                      if verdicts[tid] in ("SUSPECT", "FAIL")]

    body = findings_table(res)
    if out_dir is not None:
        page = RD.write({key: rec}, key, Path(out_dir), rung=rung)
        body += panel(page, "Five tests, with the execution-basis toggle")
    return res, body


def _luck_tests():
    import verdicts as V
    return V.TESTS


# ---------------------------------------------------------- Shelf-life
def shelf_life_contribute(registry_entry: dict, blotter: pd.DataFrame,
                          market: dict, out_dir: str | Path | None = "reports"
                          ) -> tuple[SectionResult, str]:
    """Shelf-life already returns a SectionResult; this only adds the page."""
    import shelf_life as SL

    key = str(registry_entry.get("submission", "?"))
    res = SL.shelf_life(registry_entry, blotter, market)
    if not res.conditions and hasattr(SL, "_suggested_conditions"):
        res.conditions = SL._suggested_conditions(res.verdict, res.findings)

    body = findings_table(res)
    if out_dir is not None:
        page = SL.render_html(key, blotter, market, registry_entry,
                              Path(out_dir) / f"{key}-shelf-life.html")
        body += panel(page, "Interactive dashboard — drag the rolling window, "
                            "drop the best days, rebin the regime curves")
    return res, body
