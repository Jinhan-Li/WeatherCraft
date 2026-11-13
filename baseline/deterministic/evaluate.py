"""Matched continuous autoregression for one-day and two-day baselines."""

import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch import distributed as dist
from weathercraft.config import load_config, write_json
from weathercraft.data.daily import complete_starts, channel_mse
from weathercraft.train.utils import setup_distributed, cleanup_distributed
from .data import field_dataset, ContextDays
from .model import build_model
from weathercraft.evaluation.acc import load_climatology, calendar_index, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--output")
    parser.add_argument(
        "--climatology", help="Training-only 366-day cache directory; enables paper ACC"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    device, rank, world, distributed = setup_distributed(config["seed"])
    torch.set_num_threads(2)
    days = field_dataset(config, args.split)
    climate, climate_meta = (
        load_climatology(args.climatology, config) if args.climatology else (None, None)
    )
    leads = config["evaluation"]["leads"]
    origins = complete_starts(days, max(leads))
    checkpoint_path = Path(args.checkpoint or Path(config["train"]["output_dir"]) / "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    model = build_model(config, days, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    selected_epoch = int(checkpoint["epoch"]) + 1
    del checkpoint
    model.eval()
    output = Path(args.output or config["evaluation"]["output_dir"])
    if (output / "results.json").exists():
        raise FileExistsError("Use a new evaluation output directory")
    total = torch.zeros(2, len(leads), 69, dtype=torch.float64, device=device)
    acc_total = torch.zeros(len(leads), 69, dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    latitudes = days.latitudes.to(device)
    local_origins = origins[rank::world]
    batch_size = config["train"]["batch_size"]
    started = time.monotonic()
    for offset in range(0, len(local_origins), batch_size):
        indices = local_origins[offset : offset + batch_size]
        initial = torch.cat([days[i]["field"] for i in indices]).to(device)
        history = (
            torch.cat([days.history(i) for i in indices]).to(device)
            if isinstance(days, ContextDays)
            else initial
        )
        with torch.inference_mode():
            for lead in range(1, max(leads) + 1):
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    prediction = model(history).float()
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(f"Nonfinite forecast at lead {lead}")
                history = (
                    torch.cat((history[:, 69:], prediction), dim=1)
                    if isinstance(days, ContextDays)
                    else prediction
                )
                if lead in leads:
                    truth = torch.cat([days[i + lead]["field"] for i in indices]).to(device)
                    column = leads.index(lead)
                    if climate is not None:
                        for j, i in enumerate(indices):
                            acc, _ = score(
                                prediction[j].cpu().numpy(),
                                truth[j].cpu().numpy(),
                                np.asarray(climate[calendar_index(days.days[i + lead])]),
                                days.latitudes.numpy(),
                            )
                            acc_total[column] += torch.as_tensor(acc, device=device)
                    total[0, column] += channel_mse(prediction, truth, latitudes).double() * len(
                        indices
                    )
                    total[1, column] += channel_mse(initial, truth, latitudes).double() * len(
                        indices
                    )
        count += len(indices)
        print(
            f"EVALUATION_PROGRESS rank={rank} completed={min(offset + batch_size, len(local_origins))}/{len(local_origins)}",
            flush=True,
        )
    if distributed:
        dist.all_reduce(total)
        dist.all_reduce(acc_total)
        dist.all_reduce(count)
    if int(count) != len(origins) or not origins:
        raise RuntimeError("Incomplete evaluation")
    if rank == 0:
        values = (total / count).cpu().numpy()
        standard_deviation = days.std.reshape(-1).numpy()
        result = dict(
            model=config["experiment"],
            split=args.split,
            selected_epoch=selected_epoch,
            initial_days=[str(days.days[i])[:10] for i in origins],
            start_count=len(origins),
            leads=leads,
            seconds=time.monotonic() - started,
            forecast={},
            raw_persistence={},
        )
        for row, label in enumerate(("forecast", "raw_persistence")):
            for column, lead in enumerate(leads):
                rmse = np.sqrt(values[row, column])
                result[label][str(lead)] = dict(
                    normalized_rmse=float(np.sqrt(values[row, column].mean())),
                    per_variable_normalized_rmse=dict(zip(days.variable_names, rmse.tolist())),
                    per_variable_physical_rmse=dict(
                        zip(days.variable_names, (rmse * standard_deviation).tolist())
                    ),
                )
        if climate is not None:
            result["acc_climatology"] = climate_meta
            for column, lead in enumerate(leads):
                acc = (acc_total[column] / count).cpu().numpy()
                result["forecast"][str(lead)]["acc"] = float(acc.mean())
                result["forecast"][str(lead)]["per_variable_acc"] = dict(
                    zip(days.variable_names, acc.tolist())
                )
        write_json(output / "results.json", result)
    cleanup_distributed()


if __name__ == "__main__":
    main()
