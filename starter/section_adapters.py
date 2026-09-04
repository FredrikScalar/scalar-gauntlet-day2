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

import json
import re
import sys
from pathlib import Path

import pandas as pd

from report import Finding, SectionResult, findings_table, panel

# Injected into each panel page so it can tell the report how tall it is.
# file:// iframes are opaque origins, so the parent cannot read the child's
# scrollHeight directly — the child has to volunteer it. ResizeObserver keeps
# it correct after Plotly draws and after the reader works the controls, so a
# rung toggle or a slider re-flows the report instead of scrolling inside it.
# Applied ONLY when the page is inside the report. Luck sets its ground and
# its three type faces from custom properties, so its own tokens are enough
# to bring it onto the report's white and off IBM Plex; Shelf-life hardcodes
# its ground, hence the explicit body rule. Standalone, neither page is
# touched — opened in its own tab each keeps the design its author chose.
_SANS = "system-ui,-apple-system,'Segoe UI',sans-serif"

# Flatten an embedded panel to the report's own plain look: system-ui on
# white, thin rules, small grey labels, tabular figures — the Evidence
# section's styling, which is the house style for the assembled report.
#
# The distinctive parts of Luck's design are its monospace uppercase labels
# with wide letter-spacing, a 52px condensed masthead, and cards with 12px
# radii and drop shadows. Those are what make it read as a different
# document, so each is neutralised by name. Numbers keep tabular-nums but
# lose the monospace face, which is how Evidence sets its own figures.
_FIT = (
    ":root{"
    "--ground:#fff;--panel:#fff;--panel-2:#fff;--sunk:#fafafa;"
    "--rule:#ececE7;--rule-2:#ececE7;--shadow:none;"
    f"--f-disp:{_SANS};--f-body:{_SANS};--f-mono:{_SANS};"
    "}"
    # ground and metrics
    "html,body{background:#fff!important;font-size:14px!important}"
    ".wrap,.controls-in{max-width:none!important;"
    "padding-left:0!important;padding-right:0!important}"
    # nothing raised off the page
    "*{box-shadow:none!important}"
    ".panel,.vstrip,.defn,.seg,.pill,.chip,.card{border-radius:2px!important}"
    # headings down to the report's scale
    "h1,h2,h3,h4{font-family:inherit!important;letter-spacing:0!important}"
    ".mast{padding:0 0 16px!important}"
    ".mast h1{font-size:19px!important;font-weight:650!important;"
    "margin:6px 0 8px!important}"
    ".mast .lede{font-size:13px!important}"
    ".sec-head h2,h2{font-size:15px!important;font-weight:650!important}"
    "h3{font-size:13px!important}h4{font-size:12.5px!important}"
    ".sec-sub{font-size:12.5px!important;margin-left:0!important}"
    "section{margin-bottom:26px!important}"
    # the mono uppercase label vocabulary, neutralised
    # every uppercase/letter-spaced label in the stylesheet, by name
    ".eyebrow,.vstrip-lbl,.vstrip-t,.sec-num,.defn dt,.ctl>span,.mast .meta,"
    "table.mx thead th,table.dt thead th,.tip .th,.dos-card h6,.kpi span,"
    "details.tv summary,.axis-title,.ser-lab,.tick,.mono,.num,.kpi b,"
    "table.dt td,table.mx td.rung{"
    "font-family:inherit!important;letter-spacing:0!important;"
    "text-transform:none!important}"
    ".eyebrow,.vstrip-lbl,.vstrip-t,.ctl>span,.mast .meta,.tip .th,"
    "table.mx thead th,table.dt thead th,.defn dt,.dos-card h6,.kpi span,"
    "details.tv summary{font-size:11px!important;font-weight:600!important}"
    ".kpi b{font-size:16px!important}"
    ".sec-num{font-size:11px!important}"
    # figures: the report's own treatment
    ".mono,.num,table.mx td{font-variant-numeric:tabular-nums!important}"
)

_AUTOHEIGHT = """
<script>(function(){
  if (parent !== window) {
    var st = document.createElement("style");
    st.textContent = __FIT__;
    (document.head || document.documentElement).appendChild(st);
  }
  var last = 0;
  function send(){
    var h = Math.max(document.documentElement.scrollHeight,
                     document.body ? document.body.scrollHeight : 0);
    if (h && Math.abs(h - last) > 2) {
      last = h;
      try { parent.postMessage({__panelHeight: h}, "*"); } catch (e) {}
    }
  }
  addEventListener("load", function(){
    send(); setTimeout(send, 300); setTimeout(send, 1200); setTimeout(send, 3000);
  });
  addEventListener("resize", send);
  if (window.ResizeObserver) {
    try { new ResizeObserver(send).observe(document.documentElement); } catch (e) {}
  }
  setInterval(send, 1000);
})();</script>
""".replace("__FIT__", json.dumps(_FIT))


def prepare_panel(page: Path) -> Path:
    """Make a generated page fit to embed: light theme, and self-measuring.

    Luck's stylesheet is the only one here that is theme-aware: its default
    :root is light, and it flips to dark under `prefers-color-scheme: dark`.
    Every other page in the report — Shelf-life, Evidence, the leakage page
    and the report shell itself — is light-only, so on a dark-mode machine
    Luck alone renders dark and reads as a different document.

    `data-theme="light"` is that stylesheet's own documented opt-out
    (`:root:not([data-theme="light"])` guards the dark block), so this asks
    the page for a theme it already supports rather than overriding it.
    Delete the theme half of this the day the whole report is theme-aware.

    Both edits are additive and idempotent: the standalone pages stay valid
    on their own, and re-running the pipeline does not stack copies.
    """
    txt = page.read_text(encoding="utf-8")
    out = re.sub(r"<html\b(?![^>]*data-theme)", '<html data-theme="light"',
                 txt, count=1)
    if "__panelHeight" not in out:
        if "</body>" in out:
            out = out.replace("</body>", _AUTOHEIGHT + "</body>", 1)
        else:
            out += _AUTOHEIGHT
    if out != txt:
        page.write_text(out, encoding="utf-8")
    return page


force_light = prepare_panel      # previous name, kept for callers


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
        page = prepare_panel(RD.write({key: rec}, key, Path(out_dir), rung=rung))
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
        page = prepare_panel(SL.render_html(
            key, blotter, market, registry_entry,
            Path(out_dir) / f"{key}-shelf-life.html"))
        body += panel(page, "Interactive dashboard — drag the rolling window, "
                            "drop the best days, rebin the regime curves")
    return res, body
