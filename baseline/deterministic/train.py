"""Single-step baseline training with exact global batches and clean validation."""

import argparse
from contextlib import nullcontext
import csv
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from weathercraft.config import load_config as read_config, write_json
from weathercraft.data.daily import channel_mse, groups
from weathercraft.train.utils import cosine_schedule, save_checkpoint, setup_distributed
from .data import field_dataset, training_pairs, weighted_loss
from .model import build_model


class ExactSampler(Sampler):

    def __init__(self, length, training, rank, world, seed=42):
        self.length, self.training, self.rank, self.world, self.seed = (
            length,
            training,
            rank,
            world,
            seed,
        )
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.training:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            ids = torch.randperm(self.length, generator=g).tolist()[: self.length // 32 * 32]
        else:
            ids = list(range(self.length))
        return iter(ids[self.rank :: self.world])

    def __len__(self):
        n = self.length // 32 * 32 if self.training else self.length
        return len(range(self.rank, n, self.world))


def loader(days, cfg, training):
    ds = training_pairs(days)
    sampler = ExactSampler(len(ds), training, dist.get_rank(), dist.get_world_size())
    return (
        DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            sampler=sampler,
            num_workers=cfg["workers"],
            pin_memory=True,
            drop_last=training,
            persistent_workers=cfg["workers"] > 0,
        ),
        sampler,
    )


def epoch(model, batches, lat, cfg, optimizer=None, scheduler=None, epoch_index=0, updates=0):
    training = optimizer is not None
    model.train(training)
    accum = cfg["grad_accumulation"]
    totals = torch.zeros(71, device=lat.device, dtype=torch.float64)
    grads = torch.zeros(2, device=lat.device, dtype=torch.float64)
    total_batches = len(batches)
    tail = total_batches % accum
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(batches):
        x, y = (x.to(lat.device, non_blocking=True), y.to(lat.device, non_blocking=True))
        update = (step + 1) % accum == 0 or step + 1 == total_batches
        window = tail if tail and step >= total_batches - tail else accum
        sync = model.no_sync() if training and (not update) else nullcontext()
        with sync, nullcontext() if training else torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x) if training else model.module(x)
            loss = weighted_loss(pred, y, lat)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite normalized loss")
            if training:
                (loss / window).backward()
        if training and update:
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["grad_clip"], error_if_nonfinite=True
            )
            used_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            grads[0] += grad.detach().double()
            grads[1] += 1
            if dist.get_rank() == 0 and (updates == 1 or updates % 50 == 0):
                torch.cuda.synchronize()
                print(
                    f"TRAIN_STEP_OK epoch={epoch_index + 1} update={updates} microbatch={step + 1} MSE={loss.item():.8f} grad={grad.item():.8f} used_lr={used_lr:.8g} next_lr={optimizer.param_groups[0]['lr']:.8g}",
                    flush=True,
                )
        with torch.no_grad():
            totals[:69] += channel_mse(pred, y, lat).double() * len(x)
            totals[69] += len(x)
            totals[70] += loss.detach().double() * len(x)
        if (step + 1) % 50 == 0 and dist.get_rank() == 0:
            print(
                f"{('train' if training else 'valid')} epoch={epoch_index + 1} batch={step + 1}/{total_batches} update={updates} MSE={loss.item():.8f}",
                flush=True,
            )
    dist.all_reduce(totals)
    dist.all_reduce(grads)
    per_channel = (totals[:69] / totals[69]).cpu().tolist()
    return (
        {
            "mse": float(totals[70] / totals[69]),
            "per_channel_mse": per_channel,
            "samples_no_padding": int(totals[69]),
            "grad_norm": float(grads[0] / grads[1].clamp_min(1)),
        },
        updates,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = read_config(args.config)
    tc = cfg["train"]
    torch.set_num_threads(2)
    device, rank, world, _ = setup_distributed(cfg["seed"])
    if world != 8 or tc["batch_size"] * tc["grad_accumulation"] * world != 32:
        raise ValueError("Require eight ranks and global batch32")
    out = Path(tc["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "initialization.json").exists():
        raise RuntimeError("Refusing duplicate training or implicit resume")
    ds = field_dataset(cfg, "train")
    lat = ds.latitudes.to(device)
    model = build_model(cfg, ds, device)
    model = DDP(model, device_ids=[device.index], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"], fused=True
    )
    train_batches, sampler = loader(ds, tc, True)
    valid_ds = field_dataset(cfg, "valid")
    valid_batches, _ = loader(valid_ds, tc, False)
    per_epoch = math.ceil(len(train_batches) / tc["grad_accumulation"])
    scheduler = cosine_schedule(
        optimizer,
        per_epoch * tc["epochs"],
        tc["warmup_steps"],
        tc["min_learning_rate"] / tc["learning_rate"],
    )
    if rank == 0:
        write_json(
            out / "initialization.json",
            {
                "source": "random_from_scratch",
                "pretrained_loaded": False,
                "parameters": sum((p.numel() for p in model.parameters())),
                "updates_per_epoch": per_epoch,
                "total_updates": per_epoch * tc["epochs"],
                "nominal_global_batch": 32,
                "last_global_batch": (
                    len(train_batches) % tc["grad_accumulation"] or tc["grad_accumulation"]
                )
                * tc["batch_size"]
                * world,
                "config": cfg,
            },
        )
        (out / "STAGE").write_text("training", encoding="utf-8")
    dist.barrier()
    best, updates, best_epoch = (float("inf"), 0, 0)
    for ep in range(tc["epochs"]):
        started = time.monotonic()
        sampler.set_epoch(ep)
        tr, updates = epoch(model, train_batches, lat, tc, optimizer, scheduler, ep, updates)
        va, _ = epoch(model, valid_batches, lat, tc, epoch_index=ep)
        if rank == 0:
            improved = va["mse"] < best
            if improved:
                best, best_epoch = (va["mse"], ep + 1)
            state = {
                "epoch": ep,
                "best_epoch": best_epoch,
                "best_valid_mse": best,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "updates": updates,
                "config": cfg,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(),
            }
            save_checkpoint(state, out / "last.pt")
            if improved:
                save_checkpoint(state, out / "best.pt")
            row = {
                "epoch": ep + 1,
                "updates": updates,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.monotonic() - started,
                "train_mse": tr["mse"],
                "valid_mse": va["mse"],
                "grad_norm": tr["grad_norm"],
                "best_epoch": best_epoch,
            }
            for label, metrics in (("train", tr), ("valid", va)):
                for g, ids in groups(ds.variable_names).items():
                    row[f"{label}_{g}_mse"] = float(
                        np.mean(np.asarray(metrics["per_channel_mse"])[ids])
                    )
            with (out / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if ep == 0:
                    w.writeheader()
                w.writerow(row)
            write_json(out / f"valid_epoch{ep + 1:02d}.json", va)
            print("EPOCH_COMPLETE", row, flush=True)
        dist.barrier()
    if rank == 0:
        write_json(
            out / "TRAINING_COMPLETE",
            {"epochs": tc["epochs"], "updates": updates, "best_epoch": best_epoch},
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
