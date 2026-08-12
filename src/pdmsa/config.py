from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML run configuration and resolve paths relative to that file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def resolve_config_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a serialization-safe copy without internal path helper keys."""
    result = copy.deepcopy(config)
    result.pop("_config_path", None)
    result.pop("_config_dir", None)
    return result
