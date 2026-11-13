from .runtime import *
import time, random, contextlib
import torch.distributed as dist
from .policy import gradient_mode, replay, objective, tree_objective


def update_policy(self, trans, row, rr, reuse, micros):
    start = time.monotonic()
    gradient_mode(self.model)
    self.opt.zero_grad(set_to_none=True)
    accum = torch.zeros(4, device=self.device)
    j = 0
    for t in trans:
        count = t["count"]
        weight = 2 / (self.cfg["horizon"] * count)
        for b in range(0, count, 2):
            j += 1
            h = tuple((v[b : b + 2] for v in t["history"]))
            a = tuple((v[b : b + 2] for v in t["target"]))
            s = t["season"][b : b + 2]
            with self.ddp.no_sync() if j < micros else contextlib.nullcontext():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp = replay(self.ddp, h, a, s)
                if self.update == self.resume_update and j == 1:
                    diff = (lp.detach() - t["old"][b : b + 2]).abs()
                    check = dict(
                        mean=float(diff.mean()),
                        p99=float(torch.quantile(diff.flatten(), 0.99)),
                        clip=float(
                            ((lp.detach() - t["old"][b : b + 2]).exp() - 1)
                            .abs()
                            .gt(0.2)
                            .float()
                            .mean()
                        ),
                    )
                    assert (
                        check["mean"] <= 0.01 and check["p99"] <= 0.05 and (check["clip"] < 0.001)
                    ), check
                    write(
                        self.out / f"GPU_CORE_rank{self.rank}.json",
                        dict(
                            passed=True,
                            gradient_replay=check,
                            sample_replay=row["parity"],
                            utc=now(),
                        ),
                    )
                if self.method == "tree":
                    loss, m = tree_objective(
                        lp,
                        t["old"][b : b + 2],
                        t["reference"][b : b + 2],
                        t["adv"][b : b + 2],
                        [v[0].numel() for v in a],
                    )
                else:
                    loss, m = objective(
                        lp, t["old"][b : b + 2], t["reference"][b : b + 2], t["adv"][b : b + 2]
                    )
                assert torch.isfinite(loss)
                (loss * weight).backward()
                accum += m * weight
            if j % 15 == 0:
                self.progress("backward", round=rr + 1, microbatch=j, total=micros, reuse=reuse + 1)
    assert j == micros
    grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
    assert torch.isfinite(grad)
    self.opt.step()
    self.update += 1
    dist.all_reduce(accum)
    accum /= 8
    entry = dict(
        update=self.update,
        round=rr + 1,
        reuse=reuse + 1,
        policy_loss=float(accum[0]),
        kl=float(accum[1]),
        clip=float(accum[2]),
        ratio=float(accum[3]),
        gradient_norm=float(grad),
        seconds=time.monotonic() - start,
        learning_rate=self.opt.param_groups[0]["lr"],
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        utc=now(),
    )
    if self.rank == 0:
        with (self.out / "updates.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(dict(stage="GRPO_UPDATE_OK", method=self.method, **entry)), flush=True)


def train(self):
    if self.rank == 0:
        write(
            self.out / "INITIALIZATION.json",
            dict(
                method=self.method,
                config=self.cfg,
                trainable_parameters=459951604,
                utc=now(),
                resume_update=self.update,
            ),
        )
    if self.task == "ocean":
        self.oracle_evaluate("validation")
    if not self.start_round:
        self.select(self.baseline("validation_base", "validation"))
    elif self.update % 8 == 0 or self.update == 100:
        self.select(self.evaluate(f"validation_update{self.update:03}", "validation"))
    for rr in range(self.start_round, 50):
        begin = time.monotonic()
        origin = self.origins["train"][rr * 8 + self.rank]
        trans, row = self.rollout(
            origin, self.cfg["seed"] + 100000 * (rr + 1) + self.rank, training=True
        )
        micros = sum((t["count"] // 2 for t in trans))
        expected = (
            (25 if self.task == "ocean" else 45)
            if self.method == "tree"
            else 40 if self.task == "ocean" else 60
        )
        assert micros == expected
        assert abs(sum((1 / self.cfg["horizon"] for t in trans)) - 1) < 1e-12
        row.update(microbatches=micros, day_weights=[1 / self.cfg["horizon"]] * self.cfg["horizon"])
        write(self.out / "rollouts" / f"round{rr + 1:03}_rank{self.rank}.json", row)
        for reuse in range(2):
            update_policy(self, trans, row, rr, reuse, micros)
        del trans
        rng = dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state(self.device),
        )
        rng_states = [None] * 8
        dist.all_gather_object(rng_states, rng)
        if self.rank == 0:
            tmp = self.out / "last.pt.tmp"
            torch.save(
                dict(
                    model=self.model.state_dict(),
                    optimizer=self.opt.state_dict(),
                    round=rr + 1,
                    updates=self.update,
                    config=self.cfg,
                    method=self.method,
                    rng_states=rng_states,
                ),
                tmp,
            )
            if (self.out / "last.pt").exists():
                (self.out / "last.pt").replace(self.out / "previous.pt")
            tmp.replace(self.out / "last.pt")
            write(
                self.out / "rounds" / f"round{rr + 1:03}.json",
                dict(
                    updates=self.update,
                    origins=self.origins["train"][rr * 8 : (rr + 1) * 8],
                    rank0_seconds=time.monotonic() - begin,
                    utc=now(),
                ),
            )
        dist.barrier()
        if self.update % 8 == 0 or self.update == 100:
            self.select(self.evaluate(f"validation_update{self.update:03}", "validation"))
    assert self.update == 100
    if self.task == "ocean":
        self.oracle_evaluate("test")
    self.model.load_state_dict(self.ref.state_dict())
    base = self.baseline("test_base", "test")
    ck = torch.load(self.best["path"], map_location="cpu", weights_only=False)
    self.model.load_state_dict(ck["model"])
    del ck
    selected = self.evaluate("test_selected", "test")
    if self.rank == 0:
        write(self.out / "SELECTED_CHECKPOINT.json", self.best)
        write(
            self.out / "RESULTS.json",
            dict(
                passed=True,
                method=self.method,
                updates=100,
                rounds=50,
                train_origins=400,
                validation_origins=16,
                test_origins=128,
                selected=self.best,
                base_test=base,
                selected_test=selected,
                delta_csi=selected["member_csi"] - base["member_csi"],
                elapsed_seconds=time.monotonic() - self.started,
                utc=now(),
            ),
        )
        write(self.out / "COMPLETE.json", dict(passed=True, updates=100, utc=now()))
    dist.barrier()
