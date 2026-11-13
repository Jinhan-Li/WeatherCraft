"""Shared experiment initialization, selection, and resumable state."""

from .runtime import *
import torch.distributed as dist
import time, random
from types import MethodType
from torch.nn.parallel import DistributedDataParallel as DDP
from .land_metrics import Events, LAT
from .ocean_data import OceanData, OceanRuntime
from .land import rollout as land_rollout, aggregate as land_aggregate
from .ocean import ocean_rollout
from .ocean_evaluation import aggregate as ocean_aggregate, oracle_evaluate
from .training import train
import argparse, traceback
from .runtime import OUTPUT, write, now, os


def evaluate(self, label, split):
    path = self.out / "evaluation" / f"{label}.json"
    if path.exists():
        return read(path)
    dates = self.origins[split]
    local = []
    for slot in range(self.rank, len(dates), 8):
        p = self.out / "evaluation" / f"{label}_origin{slot:03}.json"
        if p.exists():
            row = read(p)
            assert row["origin"] == dates[slot]
        else:
            _, row = self.rollout(
                dates[slot],
                self.cfg["seed"] + (9000000 if split == "validation" else 19000000) + slot,
                greedy=split == "test",
            )
            write(p, row)
        local.append(row)
    gathered = [None] * 8
    dist.all_gather_object(gathered, local)
    rows = sorted([r for part in gathered for r in part], key=lambda r: dates.index(r["origin"]))
    assert [r["origin"] for r in rows] == dates
    result = self.aggregate(rows, label)
    if self.rank == 0:
        write(path, result)
    dist.barrier()
    return result


def baseline(self, label, split):
    other = OUTPUT / self.task / "vanilla/evaluation" / f"{label}.json"
    if self.method == "tree" and other.exists():
        v = read(other)
        assert v["origins"] == self.origins[split]
        if self.rank == 0:
            write(self.out / "evaluation" / f"{label}.json", v)
            write(
                self.out / f"REUSED_{label}.json", dict(path=str(other), same_data_model_seeds=True)
            )
        dist.barrier()
        return v
    return self.evaluate(label, split)


def select(self, val):
    if self.rank == 0 and self.update:
        p = self.out / "validation_checkpoints" / f"update{self.update:03}.pt"
        p.parent.mkdir(exist_ok=True)
        if not p.exists():
            tmp = p.with_suffix(".tmp")
            torch.save(dict(model=self.model.state_dict(), updates=self.update), tmp)
            tmp.replace(p)
    if self.best is None or val["member_csi"] > self.best["validation_csi"]:
        path = BASE if self.update == 0 else self.out / "best.pt"
        if self.rank == 0 and self.update:
            tmp = self.out / "best.pt.tmp"
            torch.save(dict(model=self.model.state_dict(), updates=self.update), tmp)
            tmp.replace(path)
        self.best = dict(
            path=str(path),
            updates=self.update,
            validation_csi=val["member_csi"],
            label=val["label"],
            utc=now(),
        )
        if self.rank == 0:
            write(self.out / "SELECTED_CHECKPOINT.json", self.best)
    dist.barrier()


class Experiment:
    train = train
    evaluate = evaluate
    baseline = baseline
    select = select
    oracle_evaluate = oracle_evaluate

    def __init__(self, task, method):
        self.task = task
        self.method = method
        self.rank = int(os.environ["RANK"])
        assert int(os.environ["WORLD_SIZE"]) == 8
        self.out = OUTPUT / task / method
        self.out.mkdir(parents=True, exist_ok=True)
        self.cfg = read(ROOT / "config.json")[task]
        self.origins = read(ROOT / "origins.json")
        self.started = time.monotonic()
        assert BASE.is_file() and TOK.is_file(), "Provide base and tokenizer checkpoints"
        torch.set_num_threads(2)
        torch.cuda.set_device(self.rank)
        self.device = torch.device("cuda", self.rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        dist.init_process_group(self.cfg["backend"], timeout=dt.timedelta(minutes=45))
        self.t, self.model, self.ref, self.seasonal = make_models(self.device)
        self.ddp = DDP(self.model, device_ids=[self.rank], broadcast_buffers=False)
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg["lr"],
            betas=tuple(self.cfg["betas"]),
            weight_decay=0,
        )
        self.fields = Fields()
        self.events = OceanData() if task == "ocean" else Events(self.fields)
        if task == "ocean":
            self.ocean = OceanRuntime(self.device, self.events)
        self.rollout = MethodType(ocean_rollout if task == "ocean" else land_rollout, self)
        self.aggregate = MethodType(ocean_aggregate if task == "ocean" else land_aggregate, self)
        self.update = 0
        self.start_round = 0
        self.best = (
            read(self.out / "SELECTED_CHECKPOINT.json")
            if (self.out / "SELECTED_CHECKPOINT.json").exists()
            else None
        )
        if (self.out / "last.pt").exists():
            ck = torch.load(self.out / "last.pt", map_location="cpu", weights_only=False)
            assert ck["config"] == self.cfg and ck["method"] == method
            self.model.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["optimizer"])
            self.update = ck["updates"]
            self.start_round = ck["round"]
            rng = ck["rng_states"][self.rank]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state(rng["cuda"], self.device)
            del ck
            assert self.update == 2 * self.start_round
        self.resume_update = self.update
        channels = [66, 67, 65, 68] if task == "ocean" else [7, 65]
        self.mu = torch.from_numpy(self.fields.mean[channels]).to(self.device)
        self.sd = torch.from_numpy(self.fields.std[channels]).to(self.device)
        self.w = torch.cos(
            torch.deg2rad(torch.tensor(LAT, device=self.device, dtype=torch.float32))
        ).clamp_min(0)[None, None, :, None]
        self.initial_tokens = InitialTokens()

    def progress(self, stage, **kw):
        write(
            self.out / f"PROGRESS_rank{self.rank}.json",
            dict(utc=now(), stage=stage, update=self.update, **kw),
        )

    @torch.no_grad()
    def initial(self, origin):
        return self.initial_tokens.history(origin, self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["land", "ocean"])
    parser.add_argument("method", choices=["vanilla", "tree"])
    args = parser.parse_args()
    try:
        experiment = Experiment(args.task, args.method)
        experiment.train()
        dist.destroy_process_group()
    except BaseException:
        write(
            OUTPUT
            / args.task
            / args.method
            / f"FAILED_rank{os.environ.get('RANK', 'unknown')}.json",
            dict(utc=now(), error=traceback.format_exc()),
        )
        raise
