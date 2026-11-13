"""Train two-day latent-history dynamics with standard next-day joint CE."""

import argparse
import time
from pathlib import Path
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from weathercraft.build import build_tokenizer, build_dynamics, load_tokenizer_checkpoint
from weathercraft.config import load_config, save_resolved_config, write_json
from weathercraft.data.cache import cache_metadata, token_dataset
from weathercraft.data.daily import TwoDayTokenPairs
from weathercraft.data.sampling import FullGlobalBatchSampler
from weathercraft.train.dynamics_epoch import run_dynamics_epoch
from weathercraft.train.logging import CsvLogger
from weathercraft.train.utils import (
    setup_distributed,
    cleanup_distributed,
    cosine_schedule,
    save_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dynamics.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    train = config["train"]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(train["output_dir"])
    if (output / "initialization.json").exists():
        raise FileExistsError(
            "Use a new train.output_dir; existing training will not be overwritten"
        )
    tokenizer_config = load_config(config["tokenizer_config"])
    tokenizer = build_tokenizer(tokenizer_config)
    checkpoint = load_tokenizer_checkpoint(
        tokenizer, config["tokenizer_checkpoint"], tokenizer_config
    )
    metadata = cache_metadata(config)
    del checkpoint
    device, rank, world, distributed = setup_distributed(config["seed"])
    if train["batch_size"] * train["grad_accumulation"] * world != train["global_batch"]:
        raise ValueError("batch_size * world_size * grad_accumulation must equal global_batch")
    train_days = token_dataset(config, "train")
    valid_days = token_dataset(config, "valid")
    training = TwoDayTokenPairs(train_days)
    validation = TwoDayTokenPairs(valid_days, train_days)
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
    module = build_dynamics(config, tokenizer).to(device)
    del tokenizer
    parameters = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=train["learning_rate"], weight_decay=train["weight_decay"]
    )
    scheduler = cosine_schedule(
        optimizer,
        updates_per_epoch * train["epochs"],
        train["warmup_steps"],
        train["min_learning_rate"] / train["learning_rate"],
    )
    model = (
        DDP(module, device_ids=[device.index], broadcast_buffers=False) if distributed else module
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and train["amp"] == "fp16")
    best = float("inf")
    if rank == 0:
        save_resolved_config(config, output)
        write_json(
            output / "initialization.json",
            dict(
                model="weather_dynamics",
                history_days=2,
                parameters=sum((p.numel() for p in parameters)),
                epochs=train["epochs"],
                updates_per_epoch=updates_per_epoch,
                global_batch=train["global_batch"],
                initialization="random_from_scratch",
            ),
        )
        logger = CsvLogger(output / "metrics.csv")
    for epoch in range(train["epochs"]):
        sampler.set_epoch(epoch)
        started = time.monotonic()
        training_metrics = run_dynamics_epoch(
            model,
            train_loader,
            device,
            train["amp"],
            optimizer,
            scheduler,
            scaler,
            accumulation,
            train["grad_clip"],
            epoch + 1,
        )
        validation_metrics = run_dynamics_epoch(module, valid_loader, device, train["amp"])
        if rank == 0:
            improved = validation_metrics["loss"] < best
            best = min(best, validation_metrics["loss"])
            row = dict(
                epoch=epoch + 1,
                updates=(epoch + 1) * updates_per_epoch,
                seconds=time.monotonic() - started,
                lr=optimizer.param_groups[0]["lr"],
            )
            row.update({f"train_{k}": v for k, v in training_metrics.items()})
            row.update({f"valid_{k}": v for k, v in validation_metrics.items()})
            logger.write(row)
            state = dict(
                epoch=epoch,
                best=best,
                model=module.state_dict(),
                config=config,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
            )
            save_checkpoint(state, output / "last.pt")
            if improved:
                save_checkpoint(state, output / "best.pt")
            print(
                f"DYNAMICS_EPOCH_COMPLETE epoch={epoch + 1} valid_joint_ce={validation_metrics['loss']:.8f}",
                flush=True,
            )
        if distributed:
            dist.barrier()
    if rank == 0:
        write_json(
            output / "complete.json",
            dict(
                epochs=train["epochs"],
                best_valid_ce=best,
                updates=train["epochs"] * updates_per_epoch,
            ),
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
