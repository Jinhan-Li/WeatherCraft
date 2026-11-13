"""Evaluate reconstruction on a requested split without training or plotting."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, Subset
from weathercraft.build import build_tokenizer, load_tokenizer_checkpoint
from weathercraft.config import PROJECT_ROOT, load_config, write_json
from weathercraft.data.daily import field_dataset, channel_mse
from weathercraft.train.utils import setup_distributed, cleanup_distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tokenizer.yaml")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints/tokenizer.pt"))
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--max-days", type=int)
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "runs/tokenizer_evaluation/results.json")
    )
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    config = load_config(args.config)
    device, rank, world, distributed = setup_distributed(config["seed"])
    model = build_tokenizer(config)
    checkpoint = load_tokenizer_checkpoint(model, args.checkpoint, config)
    selected_epoch = int(checkpoint["epoch"]) + 1
    del checkpoint
    model.to(device).eval()
    days = field_dataset(config, args.split)
    limit = min(len(days), args.max_days) if args.max_days is not None else len(days)
    if limit < 1:
        raise ValueError("--max-days must be positive")
    batches = DataLoader(
        Subset(days, list(range(rank, limit, world))),
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["workers"],
    )
    totals = torch.zeros(70, dtype=torch.float64, device=device)
    with torch.inference_mode():
        for batch in batches:
            fields = batch["field"].to(device)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                prediction = model(fields).reconstruction[:, 0].float()
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite reconstruction")
            totals[:69] += channel_mse(
                prediction, fields[:, 0], days.latitudes.to(device)
            ).double() * len(fields)
            totals[69] += len(fields)
    if distributed:
        dist.all_reduce(totals)
    if rank == 0:
        mse = (totals[:69] / totals[69]).cpu().numpy()
        write_json(
            args.output,
            dict(
                model="weather_tokenizer",
                split=args.split,
                days=int(totals[69]),
                selected_epoch=selected_epoch,
                normalized_rmse=float(np.sqrt(mse.mean())),
                per_variable_physical_rmse=dict(
                    zip(days.variable_names, (np.sqrt(mse) * days.std.reshape(-1).numpy()).tolist())
                ),
            ),
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
