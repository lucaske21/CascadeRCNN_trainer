from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(RuntimeError):
    pass


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise ConfigError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        current = yaml.safe_load(f) or {}

    base_ref = current.pop("base_config", None)
    if base_ref:
        base_path = (cfg_path.parent / base_ref).resolve()
        base_cfg = load_yaml_config(base_path)
        current = _deep_merge(base_cfg, current)

    return current
