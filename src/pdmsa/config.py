from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath, PureWindowsPath
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
    """Return a serialization-safe configuration without credentials or local paths."""
    result = copy.deepcopy(config)
    result.pop("_config_path", None)
    result.pop("_config_dir", None)

    secret_terms = ("password", "passwd", "secret", "token", "api_key", "private_key")
    path_keys = {
        "manifest",
        "root",
        "output_dir",
        "assignments_file",
        "checkpoint",
        "cache_dir",
    }

    def basename(value: str) -> str:
        normalized = value.replace("\\", "/").rstrip("/")
        return normalized.rsplit("/", 1)[-1] or "<local-path>"

    def is_local_model_path(value: str) -> bool:
        return (
            value.startswith((".", "~"))
            or PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
        )

    def sanitize(value: Any, key: str = "") -> Any:
        lowered = key.lower()
        if any(term in lowered for term in secret_terms):
            return "<redacted>"
        if isinstance(value, dict):
            return {item_key: sanitize(item, str(item_key)) for item_key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [sanitize(item) for item in value]
        if isinstance(value, (str, Path)):
            text = str(value)
            if lowered in path_keys:
                return basename(text)
            if lowered == "name_or_path" and is_local_model_path(text):
                return basename(text)
        return value

    return sanitize(result)
