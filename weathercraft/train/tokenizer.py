"""Train the four-branch spatial FSQ tokenizer from scratch."""

import argparse
import time
from pathlib import Path
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from weathercraft.build import build_tokenizer
from weathercraft.config import load_config, save_resolved_config, write_json
from weathercraft.data.daily import field_dataset
from weathercraft.data.sampling import FullGlobalBatchSampler
from weathercraft.train.logging import CsvLogger
from weathercraft.train.tokenizer_epoch import atmosphere_metric_groups, run_tokenizer_epoch
from weathercraft.train.utils import (
    setup_distributed,
    cleanup_distributed,
    cosine_schedule,
    save_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tokenizer.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    train = config["train"]
    device, rank, world, distributed = setup_distributed(config["seed"])
    torch.set_num_threads(2)
    if train["batch_size"] * train["grad_accumulation"] * world != train["global_batch"]:
        raise ValueError("batch_size * world_size * grad_accumulation must equal global_batch")
    output = Path(train["output_dir"])
    if (output / "initialization.json").exists():
        raise FileExistsError(
            "Use a new train.output_dir; existing training will not be overwritten"
        )
    tokenizer = build_tokenizer(config).to(device)
    model = (
        DDP(tokenizer, device_ids=[device.index], broadcast_buffers=False)
        if distributed
        else tokenizer
    )
    training = field_dataset(config, "train")
    validation = field_dataset(config, "valid")
    sampler = FullGlobalBatchSampler(training, rank, world, train["global_batch"], config["seed"])
    train_loader = DataLoader(
        training,
        batch_size=train["batch_size"],
        sampler=sampler,
        num_workers=train["workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=train["workers"] > 0,
    )
    valid_loader = DataLoader(
        validation,
        batch_size=train["batch_size"],
        sampler=list(range(rank, len(validation), world)),
        num_workers=train["workers"],
        pin_memory=device.type == "cuda",
    )
    accumulation = train["grad_accumulation"]
    if not len(train_loader) or len(train_loader) % accumulation:
        raise ValueError("Training must contain complete global optimizer batches")
    updates_per_epoch = len(train_loader) // accumulation
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(), lr=train["learning_rate"], weight_decay=train["weight_decay"]
    )
    scheduler = cosine_schedule(
        optimizer,
        updates_per_epoch * train["epochs"],
        train["warmup_steps"],
        train["min_learning_rate"] / train["learning_rate"],
    )
    best = float("inf")
    if rank == 0:
        save_resolved_config(config, output)
        write_json(
            output / "initialization.json",
            dict(
                model="weather_tokenizer",
                parameters=sum((p.numel() for p in tokenizer.parameters())),
                epochs=train["epochs"],
                global_batch=train["global_batch"],
                updates_per_epoch=updates_per_epoch,
                initialization="random_from_scratch",
            ),
        )
        logger = CsvLogger(output / "metrics.csv")
    options = dict(
        latitudes=training.latitudes.to(device),
        device=device,
        precision=train["amp"],
        vq_weight=config["loss"]["vq_weight"],
        variable_metrics=atmosphere_metric_groups(training.variable_names),
    )
    for epoch in range(train["epochs"]):
        sampler.set_epoch(epoch)
        started = time.monotonic()
        train_metrics = run_tokenizer_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_accumulation=accumulation,
            grad_clip=train["grad_clip"],
            progress=rank == 0,
            progress_description=f"Tokenizer epoch {epoch + 1}",
            **options,
        )
        valid_metrics = run_tokenizer_epoch(tokenizer, valid_loader, **options)
        if rank == 0:
            improved = valid_metrics["reconstruction"] < best
            best = min(best, valid_metrics["reconstruction"])
            row = dict(
                epoch=epoch + 1,
                updates=(epoch + 1) * updates_per_epoch,
                seconds=time.monotonic() - started,
                lr=optimizer.param_groups[0]["lr"],
            )
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"valid_{k}": v for k, v in valid_metrics.items()})
            logger.write(row)
            state = dict(
                epoch=epoch,
                best=best,
                model=tokenizer.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                config=config,
            )
            save_checkpoint(state, output / "last.pt")
            if improved:
                save_checkpoint(state, output / "best.pt")
            print(
                f"TOKENIZER_EPOCH_COMPLETE epoch={epoch + 1} valid_mse={valid_metrics['reconstruction']:.8f}",
                flush=True,
            )
        if distributed:
            dist.barrier()
    if rank == 0:
        write_json(
            output / "complete.json",
            dict(
                epochs=train["epochs"],
                best_valid_mse=best,
                updates=train["epochs"] * updates_per_epoch,
            ),
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
