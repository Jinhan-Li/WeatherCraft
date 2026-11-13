"""CPU-only integration checks; no data, checkpoint or GPU allocation required."""

import copy
import importlib
import json
from types import SimpleNamespace
import numpy as np
import torch
from weathercraft.config import load_config
from baseline.deterministic.model import build_model


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    report = {}
    ds = SimpleNamespace(
        variable_names=[str(i) for i in range(69)], latitudes=torch.arange(-90.0, 90.0, 22.5)
    )
    for name in ("simvp", "convlstm", "sfno"):
        c = load_config(f"configs/baseline/{name}.yaml")
        c["model"]["img_size"] = [8, 16]
        if name == "simvp":
            c["model"].update(hidden_spatial=4, hidden_temporal=16, temporal_depth=2)
        elif name == "convlstm":
            c["model"]["hidden_channels"] = [4, 4]
        else:
            c["model"].update(width=4, depth=2)
        m = build_model(c, ds, "cpu")
        x = torch.randn(2, 138, 8, 16, requires_grad=True)
        y = m(x)
        assert y.shape == (2, 69, 8, 16)
        y.square().mean().backward()
        assert torch.isfinite(y).all() and min(x.grad[:, :69].norm(), x.grad[:, 69:].norm()) > 0
        report[name] = "finite forward/backward; both observed days receive gradients"
    from baseline.deterministic.storm_core import STORM

    m = STORM(
        n_var=3, lon=8, lat=16, seq_len=2, pred_len=1, num_layers=1, d_model=16, scales_number_M=1
    )
    x = torch.randn(1, 2, 3, 8, 16, requires_grad=True)
    y = m(x)
    assert y.shape == (1, 1, 3, 8, 16)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    report["storm"] = "finite forward/backward"
    from weathercraft.models.tokenizer.multiscale_fsq import MultiScaleFSQQuantizer
    from weathercraft.posttrain.crps.policy import Policy, distance, rewards

    codec = MultiScaleFSQQuantizer(
        [8, 8, 8, 5, 5, 5], 1, [(1, 2), (2, 4), (4, 8)], [4, 4, 4], [1.0, 1.0, 1.0]
    )
    m = Policy(
        codec,
        depth=1,
        model_dim=32,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        label_smoothing=0.0,
        activation_checkpointing=True,
    ).eval()
    h = tuple(torch.randint(64000, (1, 2, 4, *s)) for s in [(1, 2), (2, 4), (4, 8)])
    days = torch.zeros(2, 1, 2)
    latents, _, _ = m.trajectory(h, days, 1.0, 123)
    latents[0].retain_grad()
    latents[-1].square().mean().backward()
    assert latents[0].grad is not None and latents[0].grad.norm() > 0
    m.zero_grad()
    out = m(h, days, 1.0, [123, 456])
    out.square().mean().backward()
    assert torch.isfinite(out).all() and any(
        p.grad is not None and p.grad.norm() > 0 for p in m.parameters()
    )
    a = torch.tensor([[[[1.0, 3.0], [2.0, 4.0]]]])
    b = a + 2
    t = torch.zeros_like(a)
    lat = torch.tensor([-30.0, 30.0])
    _, loss, _ = rewards(a, b, t, lat)
    torch.testing.assert_close(
        loss, 0.5 * (distance(a, t, lat) + distance(b, t, lat) - distance(a, b, lat))
    )
    report["crps"] = (
        "two-day gradient crosses first-day latent; checkpointed backward finite; fair-CRPS identity"
    )
    from baseline.tyche.model import SwinV2_Tyche
    from baseline.tyche.loss import MeanFlowLoss

    m = SwinV2_Tyche((8, 16), 3, 3, (2, 2), (1, 1), (2, 2), depth=2, dim=32, heads=4, flash=False)
    loss = MeanFlowLoss()(m, torch.randn(2, 3, 8, 16), torch.randn(2, 3, 8, 16))
    loss.backward()
    assert torch.isfinite(loss) and all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )
    report["tyche"] = "MeanFlow JVP and backward finite"
    from weathercraft.evaluation.acc import score, calendar_index

    field = np.arange(24).reshape(3, 2, 4) + 1
    acc, _ = score(field, field, np.zeros_like(field), [-30, 30])
    np.testing.assert_allclose(acc, 1)
    assert calendar_index("2020-03-01") == calendar_index("2021-03-01")
    report["acc"] = "identity score and leap-year month-day mapping passed"
    from weathercraft.models.dynamics.weather import WeatherDynamics
    from weathercraft.models.dynamics.kv_cache import BlockKVCache
    native = WeatherDynamics(codec, depth=1, model_dim=32, num_heads=4, mlp_ratio=2.,
                             dropout=0., drop_path=0., activation_checkpointing=False).eval()
    with torch.no_grad():
        expected = native.generate(h, days[0], mode="greedy")
        actual = BlockKVCache(native).generate(h, days[0], mode="greedy")
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    report["kv_cache"] = "cached and native greedy generation agree"
    import tempfile
    from pathlib import Path
    from weathercraft.data.cache_writer import TokenCacheWriter
    from weathercraft.data.tokens import TokenDayDataset
    with tempfile.TemporaryDirectory() as tmp:
        with TokenCacheWriter(tmp, 2, [(2, 4)], 1, vocab_size=16, split="train") as writer:
            writer.write_batch(0, [np.zeros((2, 1, 2, 4), dtype=np.int64)], [1, 2])
        cache = TokenDayDataset(tmp, "train")
        assert len(cache) == 2 and cache[0]["tokens"][0].shape == (1, 2, 4)
        assert sorted(p.name for p in Path(tmp).iterdir()) == ["days.npy", "metadata.json", "scale_00.npy"]
    report["token_cache"] = "write/read roundtrip with plain metadata"
    from weathercraft.posttrain.grpo.policy import advantages, objective
    adv = advantages(torch.arange(8, dtype=torch.float32))
    lp = torch.zeros(8, 5, requires_grad=True)
    loss, _ = objective(lp, torch.zeros_like(lp), torch.zeros_like(lp), adv)
    loss.backward()
    assert torch.isfinite(lp.grad).all() and lp.grad.norm() > 0
    report["grpo"] = "policy gradient finite"
    for module in (
        "baseline.deterministic.history",
        "baseline.tyche.build",
        "baseline.tyche.loss",
        "baseline.tyche.evaluate",
        "weathercraft.evaluation.forecast",
        "weathercraft.posttrain.crps.training",
        "weathercraft.posttrain.grpo.train",
    ):
        importlib.import_module(module)
    report["imports"] = "passed"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
