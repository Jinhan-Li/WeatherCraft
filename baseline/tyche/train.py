"""Distributed MeanFlow training with explicit gradient averaging after JVP."""

import os, json, time, random, traceback
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from .build import build_model
from .loss import MeanFlowLoss
from .data import DailyPairs

ROOT = Path(__file__).resolve().parent
from weathercraft.config import load_config

CFG = load_config(os.environ.get("TYCHE_CONFIG", "configs/baseline/tyche.yaml"))
OUT = Path(CFG["output_dir"])


def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def write(p, v):
    q = p.with_suffix(".tmp")
    q.write_text(json.dumps(v, indent=2))
    q.replace(p)


def save(p, v):
    q = p.with_suffix(".tmp")
    torch.save(v, q)
    q.replace(p)


def rng():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(),
    )


def restore(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state(s["cuda"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 8
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group("gloo")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed(CFG["seed"])
    model = build_model().cuda()
    criterion = MeanFlowLoss().train()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    train = DailyPairs(CFG, "train")
    valid = DailyPairs(CFG, "valid")
    assert len(train) == 14609 and len(valid) == 364
    sampler = DistributedSampler(
        train, num_replicas=world, rank=rank, shuffle=True, seed=CFG["seed"], drop_last=True
    )
    loader = DataLoader(
        train,
        batch_size=2,
        sampler=sampler,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(2000 + rank),
    )
    vloader = DataLoader(
        Subset(valid, list(range(rank, len(valid), world))),
        batch_size=2,
        num_workers=0,
        generator=torch.Generator().manual_seed(3000 + rank),
    )
    epoch0 = 1
    skip = 0
    step = 0
    best = float("inf")
    best_epoch = 0
    latest = OUT / "LATEST.json"
    resuming = os.environ.get("TYCHE_RESUME") == "1"
    if latest.exists() and (not resuming):
        raise FileExistsError("Set TYCHE_RESUME=1 explicitly or use a fresh output_dir")
    if resuming and (not latest.exists()):
        raise FileNotFoundError(latest)
    if resuming:
        ck = torch.load(
            OUT / json.loads(latest.read_text())["file"], map_location="cpu", weights_only=False
        )
        assert ck["config"] == CFG
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        epoch0, skip, step, best, best_epoch = (
            ck["epoch"],
            ck["next_batch"],
            ck["step"],
            ck["best"],
            ck["best_epoch"],
        )
        restore(ck["rng"][rank])
        del ck
    else:
        seed(CFG["seed"] + 10000 * rank)

    def checkpoint(epoch, next_batch):
        states = [None] * world
        dist.all_gather_object(states, rng())
        if rank == 0:
            name = f"checkpoint_{step:07d}.pt"
            save(
                OUT / name,
                dict(
                    model=model.state_dict(),
                    optimizer=opt.state_dict(),
                    rng=states,
                    config=CFG,
                    epoch=epoch,
                    next_batch=next_batch,
                    step=step,
                    best=best,
                    best_epoch=best_epoch,
                ),
            )
            write(latest, dict(file=name, step=step, epoch=epoch, next_batch=next_batch))
            for p in sorted(OUT.glob("checkpoint_*.pt"))[:-2]:
                p.unlink()
        dist.barrier()

    for epoch in range(epoch0, CFG["epochs"] + 1):
        sampler.set_epoch(epoch)
        model.train()
        criterion.train()
        if not (resuming and epoch == epoch0):
            seed(CFG["seed"] + epoch + 10000 * rank)
        tick = time.perf_counter()
        total = 0.0
        count = 0
        for batch, (condition, target) in enumerate(loader):
            if batch < skip:
                continue
            condition = condition.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = criterion(model, target, condition)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad.div_(world)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            step += 1
            values = torch.tensor([loss.item(), time.perf_counter() - start], device="cuda")
            dist.all_reduce(values)
            values /= world
            total += loss.item() * len(condition)
            count += len(condition)
            if rank == 0:
                row = dict(
                    epoch=epoch,
                    batch=batch,
                    step=step,
                    loss=values[0].item(),
                    grad=grad.item(),
                    compute_seconds=values[1].item(),
                    epoch_elapsed=time.perf_counter() - tick,
                )
                with (OUT / "updates.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if step % 10 == 0:
                    print(json.dumps(row), flush=True)
            if step % 100 == 0:
                checkpoint(epoch, batch + 1)
            if step >= int(os.environ.get("TYCHE_STOP_AFTER", "1000000000")):
                checkpoint(epoch, batch + 1)
                write(
                    OUT / f"GATE_rank{rank}.json",
                    dict(
                        step=step, loss=loss.item(), grad=grad.item(), resumed=resuming, finite=True
                    ),
                )
                dist.destroy_process_group()
                return
        skip = 0
        resuming = False
        model.eval()
        criterion.eval()
        seed(CFG["seed"] + 100000 + rank)
        vtotal = 0.0
        vcount = 0
        with torch.enable_grad():
            for condition, target in vloader:
                loss = criterion(model, target.cuda(), condition.cuda())
                assert torch.isfinite(loss)
                vtotal += loss.item() * len(condition)
                vcount += len(condition)
        sums = torch.tensor([total, count, vtotal, vcount], device="cuda", dtype=torch.float64)
        dist.all_reduce(sums)
        val = (sums[2] / sums[3]).item()
        assert sums[3].item() == 364
        if val < best:
            best = val
            best_epoch = epoch
            if rank == 0:
                save(OUT / "best_model.pth", model.state_dict())
        if rank == 0:
            row = dict(
                epoch=epoch,
                step=step,
                train_loss=(sums[0] / sums[1]).item(),
                train_samples_this_segment=int(sums[1]),
                valid_loss=val,
                best=best,
                best_epoch=best_epoch,
                seconds=time.perf_counter() - tick,
            )
            with (OUT / "epochs.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            write(OUT / "SELECTED.json", row)
            print(json.dumps(row), flush=True)
        checkpoint(epoch + 1, 0)
    if rank == 0:
        write(
            OUT / "COMPLETE.json",
            dict(epochs=CFG["epochs"], step=step, best=best, best_epoch=best_epoch),
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (OUT / f"FAILED_rank{os.environ.get('RANK', 'unknown')}.txt").write_text(
            traceback.format_exc()
        )
        raise
