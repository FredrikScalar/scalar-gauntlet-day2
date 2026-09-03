"""Load every registry entry: {submission_key: dict}."""
from pathlib import Path

import yaml


def load_registry(path: str | Path = ".") -> dict[str, dict]:
    return {p.stem: yaml.safe_load(p.read_text())
            for p in sorted(Path(path).glob("*.yaml"))}
