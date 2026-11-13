"""Portable paths and validated shared data/model factories."""

import os, json
from pathlib import Path
from weathercraft.config import load_config, write_json as write
from weathercraft.data.cache import token_dataset
from weathercraft.build import build_tokenizer, load_tokenizer_checkpoint, _build_quantizer

ROOT = Path(__file__).resolve().parent
_CONFIG = load_config(os.environ.get("WEATHERCRAFT_CRPS_CONFIG", "configs/crps/one_day.yaml"))
RUN = Path(_CONFIG["output_dir"])
BASE = Path(_CONFIG["checkpoint"])


def config():
    return _CONFIG


def read(p):
    return json.loads(Path(p).read_text())


def days(split):
    return token_dataset(_CONFIG, split)


def codec():
    return _build_quantizer(load_config(_CONFIG["tokenizer_config"])["model"]["quantizer"])


def model():
    from .policy import Policy

    return Policy(codec(), **_CONFIG["model"])


def tokenizer(device):
    c = load_config(_CONFIG["tokenizer_config"])
    m = build_tokenizer(c)
    state = load_tokenizer_checkpoint(m, _CONFIG["tokenizer_checkpoint"], c)
    return m.to(device).eval().requires_grad_(False)
