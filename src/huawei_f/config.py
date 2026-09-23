"""Configuration loading for reproducible experiments."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9-3.10
    import tomli as tomllib
from pathlib import Path
from typing import Any


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 TOML configuration file."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        return tomllib.load(handle)
