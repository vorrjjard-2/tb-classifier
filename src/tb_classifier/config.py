"""YAML-backed run configuration.

Each experiment is described by a single YAML file under
``experiments/configs/``. Load it with :func:`load_config` to get a typed
:class:`Config` object that can be passed around (or unpacked into
``build_dataloaders`` etc.).

The schema is intentionally narrow for now (data only); add nested
sections (model, training, optim) as the project grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data/TBX11K"
    image_size: int = 512
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class Config:
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)


def _instantiate(cls, raw: dict[str, Any]):
    """Build a (possibly nested) dataclass from a plain dict, rejecting unknown keys."""
    if not is_dataclass(cls):
        return raw
    valid = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(valid)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, f in valid.items():
        if name not in raw:
            continue
        value = raw[name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[name] = _instantiate(f.type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping at the top level")

    data_raw = raw.get("data", {})
    if not isinstance(data_raw, dict):
        raise ValueError("'data' section must be a mapping")
    data_cfg = _instantiate(DataConfig, data_raw)

    return Config(
        name=raw.get("name", path.stem),
        data=data_cfg,
    )
