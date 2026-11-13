from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist


def setup_distributed(seed: int = 0) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    return device, rank, world_size, distributed


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
    constant_group_names: tuple[str, ...] = (),
    held_group_schedules: dict[str, tuple[int, float]] | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * progress))

    constant = set(constant_group_names)
    held = held_group_schedules or {}

    def held_multiplier(hold_steps: int, after_ratio: float):
        if hold_steps < 0 or not 0.0 < after_ratio <= 1.0:
            raise ValueError(
                "held group schedules require hold_steps >= 0 and 0 < after_ratio <= 1"
            )
        return lambda step: 1.0 if step < hold_steps else after_ratio

    multipliers = []
    for group in optimizer.param_groups:
        name = group.get("name")
        if name in constant and name in held:
            raise ValueError(f"optimizer group {name!r} cannot be both constant and held")
        if name in constant:
            multipliers.append(lambda _step: 1.0)
        elif name in held:
            multipliers.append(held_multiplier(*held[name]))
        else:
            multipliers.append(multiplier)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multipliers)


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return value
    value = value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / dist.get_world_size()


def save_checkpoint(state: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(target)
