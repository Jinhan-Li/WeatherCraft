import os, sys, json, datetime as dt
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT
PROJECT = ROOT.parents[2]
ASSETS = Path(os.environ.get("WEATHERCRAFT_GRPO_ASSETS", str(PROJECT / "data/posttrain/assets")))
OCEAN_ASSETS = Path(
    os.environ.get("WEATHERCRAFT_OCEAN_ASSETS", str(PROJECT / "data/posttrain/ocean_assets"))
)
DATA = Path(os.environ.get("WEATHERCRAFT_GRPO_DATA", str(PROJECT / "data")))
PREPARED = Path(
    os.environ.get(
        "WEATHERCRAFT_OCEAN_THRESHOLDS", str(PROJECT / "data/posttrain/ocean_thresholds")
    )
)
OUTPUT = Path(os.environ.get("WEATHERCRAFT_GRPO_OUTPUT", str(PROJECT / "runs/grpo")))
BASE = ASSETS / "weights/base.pt"
TOK = ASSETS / "weights/tokenizer.pt"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + ".tmp")
    q.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")
    q.replace(p)


def bin366(d):
    d = dt.date.fromisoformat(str(d))
    return (dt.date(2000, d.month, d.day) - dt.date(2000, 1, 1)).days


class Fields:

    def __init__(self):
        norm = ASSETS / "normalization"
        s = read(norm / "train_stats.json")
        raw = torch.load(norm / "era5_69_stats.pt", weights_only=False)
        self.names = s["variable_names"]
        assert self.names == raw["variable_names"]
        self.rawmean = np.asarray(raw["mean"], dtype="float32").reshape(69, 1, 1)
        self.rawstd = np.asarray(raw["std"], dtype="float32").reshape(69, 1, 1)
        self.shift = np.asarray(s["cache_mean"], dtype="float32")[:, None, None]
        self.scale = np.asarray(s["cache_std"], dtype="float32")[:, None, None]
        self.mean = self.rawmean + self.rawstd * self.shift
        self.std = self.rawstd * self.scale
        levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        inputs = [f"{v}_{l}" for v in ["z", "q", "t", "u", "v"] for l in levels] + [
            "u10",
            "v10",
            "t2m",
            "msl",
        ]
        self.order = [inputs.index(n) for n in self.names]
        self.years = {}

    def physical(self, date):
        d = np.datetime64(date, "D")
        year = int(str(d)[:4])
        i = int((d - np.datetime64(f"{year}-01-01")).astype(int))
        if year not in self.years:
            self.years[year] = np.load(DATA / f"era5_daily_physical/{year}.npy", mmap_mode="r")
        a = np.asarray(self.years[year][i, self.order, ::-1, :], dtype="float32")[:, :120]
        return np.roll(a, 120, axis=-1).copy()

    def cache(self, date):
        return (self.physical(date) - self.rawmean) / self.rawstd

    def normalized(self, date):
        return (self.cache(date) - self.shift) / self.scale

    def event_fields(self, date):
        x = self.normalized(date)
        return (x * self.std + self.mean)[[7, 65]]


class InitialTokens:

    def __init__(self):
        z = np.load(ASSETS / "reference/initial_tokens.npz")
        self.index = {str(d): i for i, d in enumerate(z["dates"])}
        self.arrays = [z[f"scale{k}"] for k in range(3)]

    def history(self, origin, device):
        d = np.datetime64(origin, "D")
        ii = [self.index[str(d - 1)], self.index[str(d)]]
        return tuple(
            (torch.from_numpy(a[ii].astype("int64"))[None].to(device) for a in self.arrays)
        )


def imports():
    from weathercraft.build import build_tokenizer, load_tokenizer_checkpoint
    from weathercraft.config import load_config
    from .dynamics import StoredAttentionT1
    from weathercraft.data import seasonal_features

    return (
        build_tokenizer,
        load_tokenizer_checkpoint,
        load_config,
        StoredAttentionT1,
        seasonal_features,
    )


def make_models(device, reference=True):
    build, load, config, Dynamics, seasonal = imports()
    c = read(ROOT / "config.json")["tokenizer"]
    t = build(c)
    ck = load(t, TOK, c)
    del ck
    t.to(device).eval().requires_grad_(False)
    m = Dynamics(codec=t.quantizer, **read(ROOT / "config.json")["model"])
    m.mode = "selective_attention"
    assert sum((p.numel() for p in m.parameters() if p.requires_grad)) == 459951604
    ck = torch.load(BASE, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    m.to(device)
    m.codec.requires_grad_(False)
    ref = None
    if reference:
        ref = Dynamics(codec=t.quantizer, **read(ROOT / "config.json")["model"])
        ref.mode = "selective_attention"
        ref.load_state_dict(ck["model"])
        ref.to(device).eval().requires_grad_(False)
    del ck
    return (t, m, ref, seasonal)


def advance(history, pred):
    return tuple((torch.stack((h[:, 1], p), dim=1) for h, p in zip(history, pred)))
