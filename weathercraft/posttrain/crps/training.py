import argparse, os, random, signal, time, contextlib
from . import experiment as e
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from weathercraft.train.utils import setup_distributed, cleanup_distributed
from weathercraft.data.daily import start_history
from weathercraft.data.daily import field_dataset
from weathercraft.data.tokens import seasonal_features
from weathercraft.evaluation.metrics import scores
from .policy import rewards
from torch.utils.checkpoint import checkpoint as activation_checkpoint

STOP = False


def stop(*_):
    global STOP
    STOP = True


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


def save(path, state):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        os.replace(path, path.with_name("previous.pt"))
    os.replace(tmp, path)


def link(source, target):
    tmp = target.with_suffix(".linktmp")
    if tmp.exists():
        tmp.unlink()
    os.link(source, tmp)
    os.replace(tmp, target)


from .selection import origin_indices, evaluate


def main(resume=False):
    device, rank, world, _ = setup_distributed(42)
    assert world == 8
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    cfg = e.config()
    pcfg = cfg["posttraining"]
    out = e.RUN / "train"
    out.mkdir(parents=True, exist_ok=True)
    assert (
        pcfg["lead_days"] in (1, 2)
        and pcfg["relaxation_tau"] == 1.0
        and (pcfg["learning_rate"] == 1e-06)
    )
    m = e.model()
    assert sum((p.numel() for p in m.parameters() if p.requires_grad)) == 459951604
    ck = torch.load(e.BASE, map_location="cpu", weights_only=False, mmap=True)
    m.load_state_dict(ck["model"], strict=True)
    del ck
    m.to(device).eval()
    tok = e.tokenizer(device)
    opt = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad], lr=pcfg["learning_rate"], weight_decay=0.0
    )
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    tr = e.days("train")
    va = e.days("valid")
    ft = field_dataset(cfg, "train")
    fv = field_dataset(cfg, "valid")
    assert np.array_equal(ft.day_ids, tr.day_ids) and np.array_equal(fv.day_ids, va.day_ids)
    selection = origin_indices()
    updates = 0
    history = []
    best = float("inf")
    best_update = 0
    pending_rng = None
    if resume:
        path = out / "last.pt"
        if not path.exists():
            path = out / "previous.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        assert ck["world_size"] == world
        assert (
            ck["config"]["posttraining"] == cfg["posttraining"]
            and ck["config"]["model"] == cfg["model"]
        )
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sch.load_state_dict(ck["scheduler"])
        updates, history, best, best_update = [
            ck[k] for k in ("updates", "history", "best", "best_update")
        ]
        pending_rng = ck["rank_states"][rank]
        del ck
    else:
        assert not (out / "initialization.json").exists(), "Explicit resume required"
    model = DDP(m, device_ids=[device.index], broadcast_buffers=False)
    model.eval()
    if pending_rng:
        restore(pending_rng)
    if rank == 0:
        e.write(
            out / ("RESUMED.json" if resume else "initialization.json"),
            dict(
                updates=updates,
                pretrained_dynamics_loaded=True,
                config=pcfg,
                policy_training_mode=False,
                time=time.time(),
            ),
        )

    def checkpoint():
        states = [None] * world
        dist.all_gather_object(states, rng())
        if rank == 0:
            save(
                out / "last.pt",
                dict(
                    model=m.state_dict(),
                    optimizer=opt.state_dict(),
                    scheduler=sch.state_dict(),
                    rank_states=states,
                    updates=updates,
                    history=history,
                    best=best,
                    best_update=best_update,
                    world_size=world,
                    epoch=0,
                    config=cfg,
                ),
            )
            e.write(
                out / "CHECKPOINT.json",
                dict(updates=updates, best_update=best_update, time=time.time()),
            )
        dist.barrier()

    if updates == 0:
        baseline = evaluate(m, tok, va, tr, fv, selection, "selection_0000", rank, world, device)
        best = baseline["fair_crps"]
        history = [dict(updates=0, validation=baseline["fair_crps"])]
        checkpoint()
        if rank == 0:
            link(out / "last.pt", out / "best.pt")
        dist.barrier()
    start_updates = updates
    limit = pcfg["updates"]
    started = time.monotonic()
    while updates < limit:
        accum = pcfg["gradient_accumulation"]
        indices = (
            np.random.default_rng(3042 + updates)
            .choice(np.arange(1, len(tr) - pcfg["lead_days"]), world * accum, replace=False)
            .reshape(accum, world)
        )
        opt.zero_grad(set_to_none=True)
        aggregate = torch.zeros(4, device=device)
        for micro in range(accum):
            i = int(indices[micro, rank])
            h = start_history(tr, i, device)
            day = torch.stack(
                [
                    seasonal_features(int(tr.day_ids[i]) + lead)[None]
                    for lead in range(1, pcfg["lead_days"] + 1)
                ]
            ).to(device)
            truth = ft[i + pcfg["lead_days"]]["field"].to(device)
            seeds = [1900000000 + updates * 10000 + micro * 100 + rank * 2 + j for j in range(2)]
            with contextlib.nullcontext() if micro == accum - 1 else model.no_sync():
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    latent = model(h, day, pcfg["relaxation_tau"], seeds)
                fields = [
                    activation_checkpoint(
                        lambda z: tok.adapter.unpack(tok.autoencoder.decode(z))[:, 0].float(),
                        v.float(),
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                    for v in latent
                ]
                _, crps, parts = rewards(fields[0], fields[1], truth, ft.latitudes.to(device))
                loss = crps.mean() / accum
                assert torch.isfinite(loss)
                loss.backward()
            aggregate += (
                torch.tensor(
                    [float(crps.detach().mean()), *[float(x.detach().mean()) for x in parts]],
                    device=device,
                )
                / accum
            )
            del fields, latent, crps, parts, loss
        grad = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        sch.step()
        updates += 1
        stats = torch.cat([aggregate, torch.as_tensor(grad, device=device).reshape(1)])
        dist.all_reduce(stats)
        stats /= world
        row = dict(
            updates=updates,
            fair_crps=float(stats[0]),
            error1=float(stats[1]),
            error2=float(stats[2]),
            distance=float(stats[3]),
            grad=float(stats[4]),
            seconds=time.monotonic() - started,
            global_origins=world * accum,
            relaxation_tau=pcfg["relaxation_tau"],
            gradient_estimator="biased_gumbel_ste_full_scale_full_time",
        )
        if rank == 0:
            print("CRPS_TRAIN_STEP_OK", row, flush=True)
            e.write(out / "PROGRESS.json", row)
        if updates == start_updates + 1:
            e.write(
                out / f"FIRST_STEP_rank{rank}.json",
                dict(**row, optimizer_steps=sorted({int(x["step"]) for x in opt.state.values()})),
            )
        improved = False
        if updates in pcfg["selection_steps"]:
            val = evaluate(
                m, tok, va, tr, fv, selection, f"selection_{updates:04d}", rank, world, device
            )
            improved = val["fair_crps"] < best
            if improved:
                best = val["fair_crps"]
                best_update = updates
            row["validation"] = val["fair_crps"]
        history.append(row)
        flag = torch.tensor(int(STOP or (e.RUN / "STOP_REQUEST.json").exists()), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if (
            updates % pcfg["checkpoint_every"] == 0
            or updates == start_updates + 1
            or updates in pcfg["selection_steps"]
            or (updates == limit)
            or flag.item()
        ):
            checkpoint()
        if improved and rank == 0:
            link(out / "last.pt", out / "best.pt")
        dist.barrier()
        if flag.item():
            if rank == 0:
                e.write(out / "INTERRUPTED.json", dict(updates=updates, time=time.time()))
            cleanup_distributed()
            return
    if rank == 0:
        e.write(
            out / "TRAINING_COMPLETE.json",
            dict(updates=updates, best_update=best_update, history=history),
        )
        e.write(
            e.RUN / "COMPLETE.json",
            dict(updates=updates, best_update=best_update, no_test_evaluation=True),
        )
    cleanup_distributed()
