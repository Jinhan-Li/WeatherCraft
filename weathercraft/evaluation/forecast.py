"""Ensemble forecast evaluation with per-origin scores and latitude weighting."""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import distributed as dist
from weathercraft.config import load_config, write_json
from weathercraft.build import build_tokenizer, load_tokenizer_checkpoint, build_dynamics
from weathercraft.data.daily import field_dataset, complete_starts
from weathercraft.inference import Forecaster, observed_fields
from weathercraft.evaluation.metrics import scores
from weathercraft.train.utils import setup_distributed, cleanup_distributed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/dynamics.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["valid", "test"], default="test")
    a = p.parse_args()
    c = load_config(a.config)
    device, rank, world, distributed = setup_distributed(42)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    leads = c["evaluation"]["leads"]
    members = c["evaluation"]["ensemble_members"]
    days = field_dataset(c, a.split)
    origins = complete_starts(days, max(leads))
    if members < 2 or len(origins) == 0:
        raise ValueError("Need at least two members and complete origins")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        raise FileExistsError("Use a new output directory")
    tc = load_config(c["tokenizer_config"])
    tok = build_tokenizer(tc)
    ts = load_tokenizer_checkpoint(tok, c["tokenizer_checkpoint"], tc)
    del ts
    model = build_dynamics(c, tok)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(ck["model"], strict=True)
    del ck
    f = Forecaster.__new__(Forecaster)
    f.device = device
    f.tokenizer = tok.to(device).eval().requires_grad_(False)
    f.model = model.to(device).eval().requires_grad_(False)
    rows = []
    persist = []
    lat = days.latitudes.numpy()
    w = np.maximum(np.cos(np.deg2rad(lat.astype("f8"))), 0)
    for i in origins[rank::world]:
        day = int(days.day_ids[i])
        torch.manual_seed(c["evaluation"]["seed"] + day)
        h = f.encode_history(observed_fields(c, a.split, days, i))
        metrics = []
        per = []
        initial = days[i]["field"][0].numpy().astype("f8")
        for lead, samples in f.rollout(
            h, day, leads, members, kv_cache=c["evaluation"].get("kv_cache", False)
        ):
            truth = days[i + lead]["field"][0].numpy()
            metrics.append(scores(samples.cpu().numpy(), truth, lat))
            per.append(
                ((initial - truth) ** 2 * w[None, :, None]).sum((-2, -1))
                / (w.sum() * initial.shape[-1])
            )
        assert np.isfinite(metrics).all()
        rows.append(metrics)
        persist.append(per)
        print("TEST_PROGRESS", rank, len(rows), len(origins[rank::world]), flush=True)
    np.savez_compressed(
        out / f"rank{rank}.npz",
        indices=origins[rank::world],
        metrics=np.asarray(rows),
        persistence=np.asarray(persist),
    )
    if distributed:
        dist.barrier()
    if rank == 0:
        parts = [np.load(out / f"rank{r}.npz") for r in range(world)]
        indices = np.concatenate([v["indices"] for v in parts])
        order = np.argsort(indices)
        assert np.array_equal(indices[order], origins)
        values = np.concatenate([v["metrics"] for v in parts])[order]
        persistence = np.concatenate([v["persistence"] for v in parts])[order]
        np.savez_compressed(
            out / "per_origin_metrics.npz",
            metrics=values,
            persistence_mse=persistence,
            origin_days=days.day_ids[origins],
            leads=leads,
            variables=days.variable_names,
            std=days.std.flatten().numpy(),
        )
        avg = values.mean(0)
        write_json(
            out / "COMPLETE.json",
            dict(
                origins=len(origins),
                members=members,
                leads=leads,
                metric_axes=[
                    "ensemble_mse",
                    "spread_var",
                    "fair_crps",
                    "empirical_crps",
                    "member0_mse",
                ],
                per_lead=[
                    dict(lead=l, rmse=float(np.sqrt(v[0].mean())), fair_crps=float(v[2].mean()))
                    for l, v in zip(leads, avg)
                ],
            ),
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
