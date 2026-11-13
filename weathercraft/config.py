"""Project-relative configuration loading."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Mapping
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        value = value.replace("${PROJECT_ROOT}", str(PROJECT_ROOT))
        value = value.replace(
            "${ERA5_ROOT}", os.environ.get("ERA5_ROOT", str(PROJECT_ROOT / "data/era5"))
        )
        value = os.path.expandvars(value)
        if "${" in value:
            raise ValueError(f"Unresolved environment variable: {value}")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and (not path.exists()):
        path = PROJECT_ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    config = _expand(config)
    if "data_config" in config:
        shared = load_config(config.pop("data_config"))
        config["data"] = dict(shared["data"], **config.get("data", {}))
    config["_config_path"] = str(path.resolve())
    return config


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_resolved_config(config: Mapping, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    (output / "config.yaml").write_text(yaml.safe_dump(clean, sort_keys=False), encoding="utf-8")
