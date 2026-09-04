"""Turn cached five-tests records into a self-contained HTML report.

One template (template.html) serves both shapes:
  * subject = "windfall"  -> the single-strategy report, peers as context
  * subject = None        -> the combined seven-strategy report
"""
from __future__ import annotations
import base64, json, re, sys
from pathlib import Path
import engine as E
import verdicts as VD

HERE = Path(__file__).resolve().parent
TEMPLATE_ONE = HERE / "template_one.html"     # a single strategy, on its own
TEMPLATE_ALL = HERE / "template_all.html"     # all seven side by side
STYLESHEET = HERE / "report.css"              # shared design system
FONT_DIRS = [HERE / "fonts", HERE.parent.parent / "node_modules" / "@fontsource"]

# The palette slot each strategy owns, so a hue always means the same strategy
# across every report in the set. Order is the validated categorical sequence.
PALETTE_ORDER = ["windfall", "windfall2", "sunspot", "spikecatcher",
                 "pingpong", "bounceback", "blackbox"]

FONT_FACES = [
    ("IBM Plex Sans", 400, "normal", "ibm-plex-sans/files/ibm-plex-sans-latin-400-normal.woff2"),
    ("IBM Plex Sans", 500, "normal", "ibm-plex-sans/files/ibm-plex-sans-latin-500-normal.woff2"),
    ("IBM Plex Sans", 600, "normal", "ibm-plex-sans/files/ibm-plex-sans-latin-600-normal.woff2"),
    ("IBM Plex Sans", 400, "italic", "ibm-plex-sans/files/ibm-plex-sans-latin-400-italic.woff2"),
    ("IBM Plex Sans Condensed", 600, "normal",
     "ibm-plex-sans-condensed/files/ibm-plex-sans-condensed-latin-600-normal.woff2"),
    ("IBM Plex Sans Condensed", 700, "normal",
     "ibm-plex-sans-condensed/files/ibm-plex-sans-condensed-latin-700-normal.woff2"),
    ("IBM Plex Mono", 400, "normal", "ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2"),
    ("IBM Plex Mono", 500, "normal", "ibm-plex-mono/files/ibm-plex-mono-latin-500-normal.woff2"),
    ("IBM Plex Mono", 600, "normal", "ibm-plex-mono/files/ibm-plex-mono-latin-600-normal.woff2"),
]


def embedded_fonts() -> str | None:
    """@font-face rules with the woff2 inlined, so the file works offline.

    Looks for the IBM Plex woff2 files under starter/luck/fonts/ or in an
    @fontsource install. Returns None when they are not present, in which case
    the report falls back to the Google Fonts stylesheet.
    """
    css = []
    for fam, wt, style, rel in FONT_FACES:
        path = next((d/rel for d in FONT_DIRS if (d/rel).exists()), None)
        if path is None:
            flat = next((d/Path(rel).name for d in FONT_DIRS if (d/Path(rel).name).exists()), None)
            path = flat
        if path is None:
            return None
        b64 = base64.b64encode(path.read_bytes()).decode()
        css.append(f'@font-face{{font-family:"{fam}";font-style:{style};font-weight:{wt};'
                   f'font-display:swap;src:url(data:font/woff2;base64,{b64}) format("woff2");}}')
    return "\n".join(css)


def _normalise(rec: dict) -> dict:
    """Field aliases the template reads, kept out of the compute layer."""
    r = dict(rec)
    r["seed"] = rec["seed_stability"]
    r["mvs"] = rec["rungs"]["mid"]["mvs"]
    return r


def _common(rec: dict, rung: str) -> dict:
    return {
        "rung": rung,
        "rungs": rec["rungs_available"],
        "fee": rec["fee"],
        "v_search": rec["v_search"],
        "drift_per_step": rec["drift_per_step"],
        "drift_full": rec["drift_full"],
        "tests": [{"id": tid, "name": name} for tid, name, _ in VD.TESTS],
        "thresholds": {
            "coinflip_z_keep": VD.COINFLIP_Z_KEEP, "coinflip_z_fail": VD.COINFLIP_Z_FAIL,
            "coinflip_p_fail": VD.COINFLIP_P_FAIL,
            "boot_lower_keep": VD.BOOT_LOWER_KEEP, "dsr_mult": VD.DSR_BREAK_KEEP_MULT,
            "days_pct_keep": VD.DAYS_PCT_KEEP, "days_pct_fail": VD.DAYS_PCT_FAIL,
            "ratio_keep": VD.RATIO_KEEP, "ratio_fail": VD.RATIO_FAIL,
        },
    }


def build_payload_one(rec: dict, rung: str) -> dict:
    """A single strategy, on its own. No other submission appears in the payload."""
    p = _common(rec, rung)
    p["sub"] = _normalise(rec)
    return p


def build_payload_all(records: dict, rung: str) -> dict:
    order = [k for k in PALETTE_ORDER if k in records]
    p = _common(records[order[0]], rung)
    p["subject"] = None
    p["order"] = order
    p["palette_order"] = PALETTE_ORDER
    p["subs"] = {k: _normalise(records[k]) for k in order}
    return p


def render(records: dict, subject: str | None, rung="cross_fee",
           embed_fonts=True, standalone=True) -> str:
    """The finished HTML. standalone=False strips the document wrapper for
    publishing as an Artifact (which supplies its own <head>/<body>)."""
    if subject is not None and subject not in records:
        raise KeyError(f"no computed record for {subject}")
    tpl = TEMPLATE_ONE if subject else TEMPLATE_ALL
    html = tpl.read_text(encoding="utf-8").replace(
        "__CSS__", STYLESHEET.read_text(encoding="utf-8"))
    payload = (build_payload_one(records[subject], rung) if subject
               else build_payload_all(records, rung))
    data = json.dumps(payload, separators=(",", ":"))
    if "</script>" in data:
        raise ValueError("payload contains a closing script tag")

    title = (f"{records[subject]['name']} · Five Tests" if subject
             else "Five Tests, Seven Strategies")
    html = html.replace("__TITLE__", title).replace("__DATA__", data)

    if embed_fonts:
        faces = embedded_fonts()
        if faces:
            html = re.sub(r'<link rel="preconnect"[^>]*>\s*', "", html)
            html = re.sub(r'<link rel="stylesheet" href="https://fonts\.googleapis\.com[^>]*>\s*',
                          "", html)
            html = html.replace("<style>\n:root{", "<style>\n" + faces + "\n:root{", 1)
        else:
            print("  (IBM Plex woff2 not found — keeping the Google Fonts link)")

    if not standalone:
        head = re.search(r"<head>(.*?)</head>", html, re.S).group(1)
        body = re.search(r"<body>(.*?)</body>", html, re.S).group(1)
        head = re.sub(r'<meta charset="utf-8">\s*', "", head)
        head = re.sub(r'<meta name="viewport"[^>]*>\s*', "", head)
        html = head.strip() + "\n" + body.strip() + "\n"
    return html


def write(records: dict, subject: str | None, out_dir: Path, rung="cross_fee",
          embed_fonts=True, standalone=True, suffix="") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (subject or "all-strategies") + "-five-tests" + suffix
    p = out_dir / f"{stem}.html"
    p.write_text(render(records, subject, rung, embed_fonts, standalone), encoding="utf-8")
    return p
