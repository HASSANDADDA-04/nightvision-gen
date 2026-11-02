"""Configuration utilities for NightVision Gen.

Supports hierarchical YAML loading, merging base configs with experiment configs,
environment variable overrides, and dictionary-like / attribute-like access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigDict(dict):
    """Dictionary that allows attribute-style access (e.g. config.model.image_size)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)
            elif isinstance(value, list):
                self[key] = [
                    ConfigDict(item)
                    if isinstance(item, dict) and not isinstance(item, ConfigDict)
                    else item
                    for item in value
                ]

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = ConfigDict(value) if isinstance(value, dict) else value

    def __delattr__(self, item: str) -> None:
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{item}'")

    def to_dict(self) -> dict[str, Any]:
        """Convert back to native recursive dictionary."""
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, ConfigDict) else item for item in value
                ]
            else:
                result[key] = value
        return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file safely from disk.

    Args:
        path: Filepath to YAML configuration.

    Returns:
        Loaded configuration dictionary.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary into base dictionary.

    Args:
        base: Base dictionary.
        override: Override dictionary whose values take precedence.

    Returns:
        New merged dictionary.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: str | Path,
    base_config_path: str | Path | None = None,
) -> ConfigDict:
    """Load configuration from file, optionally merging with a base configuration.

    Args:
        config_path: Path to target configuration YAML.
        base_config_path: Optional path to a base YAML file (e.g. data.yaml or model.yaml).

    Returns:
        ConfigDict with attribute-style access.
    """
    cfg = load_yaml(config_path)

    if base_config_path is not None:
        base_cfg = load_yaml(base_config_path)
        cfg = deep_merge(base_cfg, cfg)

    # Optional merge of data_config path specified inside the yaml
    if "paths" in cfg and "data_config" in cfg["paths"]:
        data_cfg_path = Path(cfg["paths"]["data_config"])
        if not data_cfg_path.is_absolute():
            # Resolve relative to workspace or config location
            workspace_candidate = Path(data_cfg_path)
            if workspace_candidate.exists():
                data_cfg = load_yaml(workspace_candidate)
                cfg = deep_merge(data_cfg, cfg)

    return ConfigDict(cfg)


def save_config(config: ConfigDict | dict[str, Any], output_path: str | Path) -> None:
    """Save configuration to disk as YAML.

    Args:
        config: ConfigDict or dictionary to save.
        output_path: Destination path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dict = config.to_dict() if isinstance(config, ConfigDict) else config
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw_dict, f, default_flow_style=False, sort_keys=False)
