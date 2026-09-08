#!/usr/bin/env python3
"""Convenience launcher for the luck pipeline, which lives in `luck/`.

This file used to be a COPY of `luck/pipeline.py` sitting one directory up,
where its imports (`engine`, `fivetests`, `render` — all in `luck/`) could
not resolve, so it failed on import and had drifted behind the original.

It is now a launcher: the pipeline itself has one home. Every argument is
passed straight through, so these are the same command:

    python starter/pipeline_luck.py all
    python starter/luck/pipeline.py all
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

LUCK = Path(__file__).resolve().parent / "luck"

if __name__ == "__main__":
    # `luck/` first: its modules import each other by bare name, and its
    # `verdicts` is not the one `friction_verdicts.py` replaced.
    sys.path.insert(0, str(LUCK))
    sys.argv[0] = str(LUCK / "pipeline.py")
    runpy.run_path(str(LUCK / "pipeline.py"), run_name="__main__")
