"""Daily Tyche: one-step MeanFlow sampling and own-feedback ensemble metrics."""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import distributed as dist
from .build import build_model
from weathercraft.config import load_config, write_json
from weathercraft.data.daily import field_dataset, complete_starts
from weathercraft.evaluation.metrics import scores
from weathercraft.train.utils import setup_distributed, cleanup_distributed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/baseline/tyche.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    c = load_config(a.config)
    device, rank, world, distributed = setup_distributed(42)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_model().to(device).eval().requires_grad_(False)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    days = field_dataset(c, "test")
    origins = complete_starts(days, 15)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        raise FileExistsError(out)
    leads = [1, 3, 5, 10, 15]
    rows = []
    one = torch.ones(8, device=device)
    zero = torch.zeros(8, device=device)
    with torch.inference_mode():
        for i in origins[rank::world]:
            gen = torch.Generator(device=device).manual_seed(20261108 + int(i) * 100)
            field = days[i]["field"].to(device).expand(8, -1, -1, -1).contiguous()
            row = []
            for lead in range(1, 16):
                noise = torch.randn(field.shape, device=device, generator=gen)
                field = noise - model(noise, one, field, r=zero)
                assert torch.isfinite(field).all()
                if lead in leads:
                    row.append(
                        scores(
                            field.cpu().numpy(),
                            days[i + lead]["field"][0].numpy(),
                            days.latitudes.numpy(),
                        )
                    )
            rows.append(row)
            print("TEST_PROGRESS", rank, len(rows), len(origins[rank::world]), flush=True)
    np.savez_compressed(out / f"rank{rank}.npz", indices=origins[rank::world], metrics=rows)
    if distributed:
        dist.barrier()
    if rank == 0:
        parts = [np.load(out / f"rank{r}.npz") for r in range(world)]
        indices = np.concatenate([v["indices"] for v in parts])
        order = np.argsort(indices)
        assert np.array_equal(indices[order], origins)
        values = np.concatenate([v["metrics"] for v in parts])[order]
        avg = values.mean(0)
        np.savez_compressed(
            out / "per_origin_metrics.npz",
            metrics=values,
            origin_days=days.day_ids[origins],
            leads=leads,
            variables=days.variable_names,
            std=days.std.flatten().numpy(),
        )
        write_json(
            out / "COMPLETE.json",
            dict(
                origins=len(origins),
                members=8,
                leads=leads,
                per_lead=[
                    dict(lead=l, rmse=float(np.sqrt(v[0].mean())), fair_crps=float(v[2].mean()))
                    for l, v in zip(leads, avg)
                ],
            ),
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
