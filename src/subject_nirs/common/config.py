"""Small, strict YAML configuration helpers."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a mapping from YAML and reject malformed top-level values."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise TypeError(f"Expected a YAML mapping in {config_path}")
    return values


def apply_dataclass_config(config: Any, values: dict[str, Any]) -> Any:
    """Apply validated YAML keys to an existing dataclass instance."""
    if not is_dataclass(config):
        raise TypeError("config must be a dataclass instance")
    allowed = {field.name for field in fields(config)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise KeyError(f"Unknown configuration keys: {', '.join(unknown)}")
    for key, value in values.items():
        setattr(config, key, value)
    validate = getattr(config, "validate", None)
    if callable(validate):
        validate()
    return config
