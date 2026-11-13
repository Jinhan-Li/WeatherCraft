"""FP32 token-cache construction, sharded by independent distributed ranks."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, Subset
from weathercraft.build import build_tokenizer, load_tokenizer_checkpoint
from weathercraft.config import load_config, write_json
from weathercraft.data.cache_writer import TokenCacheWriter
from weathercraft.data.daily import field_dataset
from weathercraft.data.tokens import TokenDayDataset
from weathercraft.train.utils import setup_distributed, cleanup_distributed


def cache_metadata(config):
    """Describe the data split and token encoding."""
    tokenizer_config = load_config(config["tokenizer_config"])
    data = config["data"]
    field = Path(data["path"])
    return dict(crop_latitude=data["crop_latitude"], splits=data["splits"], encoding="fp32")


def token_dataset(config, split, metadata=None):
    metadata = metadata or cache_metadata(config)
    days = TokenDayDataset(config["token_cache"], split)
    model = load_config(config["tokenizer_config"])["model"]["quantizer"]
    days.validate_schema(
        scale_shapes=model["scale_shapes"],
        num_codebooks=max(model["codebooks_per_scale"]) * model["slots"],
        vocab_size=int(np.prod(model["levels"])),
    )
    bounds = np.array(config["data"]["splits"][split], dtype="datetime64[D]").astype(np.int64)
    if not np.array_equal(days.day_ids, np.arange(bounds[0], bounds[1] + 1)):
        raise ValueError(f"{split} cache does not match its complete daily split")
    return days


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dynamics.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "valid", "test"],
        default=["train", "valid", "test"],
    )
    args = parser.parse_args()
    config = load_config(args.config)
    device, rank, world, distributed = setup_distributed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer_config = load_config(config["tokenizer_config"])
    tokenizer = build_tokenizer(tokenizer_config)
    checkpoint = load_tokenizer_checkpoint(
        tokenizer, config["tokenizer_checkpoint"], tokenizer_config
    )
    metadata = cache_metadata(config)
    del checkpoint
    tokenizer.to(device).eval().requires_grad_(False)

    def writer(path, count, split):
        return TokenCacheWriter(
            path,
            count,
            tokenizer.quantizer.scale_shapes,
            tokenizer.quantizer.num_codebooks,
            split=split,
            modality="atmosphere",
            vocab_size=tokenizer.quantizer.vocab_size,
            metadata=metadata,
        )

    for split in args.splits:
        target = Path(config["token_cache"]) / split
        if (target / "metadata.json").exists():
            token_dataset(config, split)
            if rank == 0:
                print(f"CACHE_REUSED split={split}", flush=True)
            continue
        days = field_dataset(config, split)
        indices = np.array_split(np.arange(len(days)), world)[rank]
        if not len(indices):
            raise ValueError("More workers than available days")
        part_root = Path(config["token_cache"]) / "parts" / split
        part = part_root / f"rank{rank}"
        if (part / "metadata.json").exists():
            cached = TokenDayDataset(part, split)
            if not np.array_equal(cached.day_ids, days.day_ids[indices]):
                raise ValueError("Cached part belongs to a different partition")
        else:
            offset = 0
            batches = DataLoader(
                Subset(days, indices.tolist()),
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
            with writer(part, len(indices), split) as output, torch.inference_mode():
                for item in batches:
                    encoded = tokenizer.encode(item["field"].to(device))
                    output.write_batch(offset, encoded, item["day"])
                    offset += len(item["day"])
            if offset != len(indices):
                raise RuntimeError("Incomplete encoding")
        if distributed:
            dist.barrier()
        if rank == 0:
            offset = 0
            with writer(target, len(days), split) as output:
                for worker in range(world):
                    chunk = TokenDayDataset(part_root / f"rank{worker}", split)
                    for item in chunk:
                        if int(item["day"]) != int(days.day_ids[offset]):
                            raise ValueError("Token-cache merge changed date order")
                        output.write(offset, item["tokens"], item["day"])
                        offset += 1
            token_dataset(config, split)
            write_json(
                target / "summary.json",
                dict(
                    days=offset, encoding="fp32", first=str(days.days[0]), last=str(days.days[-1])
                ),
            )
            print(f"CACHE_COMPLETE split={split} days={offset}", flush=True)
        if distributed:
            dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
