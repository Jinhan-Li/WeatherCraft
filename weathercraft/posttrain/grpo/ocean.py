"""Tree or independent weather/ocean rollout."""

from .runtime import *
import time
from .policy import sample, replay, advantages, branch_credit, node_count
from .ocean_metrics import (
    score,
    probability,
    check_cached_ocean_path,
    check_threshold_sensitive_reward,
)


@torch.no_grad()
def ocean_rollout(self, origin, seed, training=False, greedy=False):
    start = time.monotonic()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    self.model.eval()
    day = int(np.datetime64(origin, "D").astype(int))
    tree = training and self.method == "tree"
    ctx = self.events.context(origin)
    initial_ocean = torch.from_numpy(ctx.pop("initial")[None]).to(self.device)
    initial = self.initial(origin)
    history = (
        initial if tree else tuple((x.expand(8, *[-1] * (x.ndim - 1)).clone() for x in initial))
    )
    gh = initial if greedy else None
    gs = initial_ocean
    gp = []
    ge = []
    states = [initial_ocean] * (1 if tree else 8)
    paths = [[] for _ in states]
    errors = [[] for _ in states]
    forcings = [[] for _ in states]
    trans = []
    parity = []
    ocean_steps = 0
    weather_seconds = 0.0
    ocean_seconds = 0.0
    for lead in range(1, 11):
        if tree and lead in (1, 4, 7):
            history = tuple((x.repeat_interleave(2, 0) for x in history))
            states = [s for s in states for _ in range(2)]
            paths = [list(p) for p in paths for _ in range(2)]
            errors = [list(p) for p in errors for _ in range(2)]
            forcings = [list(p) for p in forcings for _ in range(2)]
        count = len(states)
        assert count == (node_count(lead) if tree else 8)
        begin = time.monotonic()
        season = self.seasonal(day + lead)[None].to(self.device).expand(count, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if training:
                parts = [
                    sample(self.model, tuple((x[j : j + 2] for x in history)), season[j : j + 2])
                    for j in range(0, count, 2)
                ]
                pred = tuple((torch.cat([p[0][s] for p in parts]) for s in range(3)))
                slp = torch.cat([p[1] for p in parts])
                decoded = torch.cat(
                    [
                        self.t.decode(tuple((v[j : j + 2] for v in pred)))[:, 0].float()
                        for j in range(0, count, 2)
                    ]
                )
            else:
                pred, slp = sample(self.model, history, season)
                decoded = self.t.decode(pred)[:, 0].float()
            if greedy:
                g = self.model.generate(gh, season[:1], mode="greedy")
                gh = advance(gh, g)
                gd = self.t.decode(g)[:, 0].float()
        assert torch.isfinite(decoded).all()
        if training:
            old = []
            ref = []
            for j in range(0, count, 2):
                h = tuple((x[j : j + 2] for x in history))
                a = tuple((x[j : j + 2] for x in pred))
                s = season[j : j + 2]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    old.append(replay(self.model, h, a, s))
                    ref.append(replay(self.ref, h, a, s))
            old = torch.cat(old)
            ref = torch.cat(ref)
            err = (old - slp).abs().float()
            check = dict(
                lead=lead, mean=float(err.mean()), p99=float(torch.quantile(err.flatten(), 0.99))
            )
            assert check["mean"] <= 0.01 and check["p99"] <= 0.05, check
            parity.append(check)
            trans.append(
                dict(
                    history=history,
                    target=pred,
                    season=season,
                    old=old,
                    reference=ref,
                    count=count,
                    lead=lead,
                )
            )
        f = (decoded[:, [66, 67, 65, 68]] * self.sd + self.mu).cpu().numpy()
        truth = torch.from_numpy(self.fields.normalized(np.datetime64(day + lead, "D"))).to(
            self.device
        )
        mse = (
            (((decoded - truth) ** 2 * self.w).sum((2, 3)) / (self.w.sum() * 240))
            .mean(-1)
            .cpu()
            .tolist()
        )
        if greedy:
            gf = (gd[:, [66, 67, 65, 68]] * self.sd + self.mu)[0].cpu().numpy()
            ge.append(
                float((((gd - truth) ** 2 * self.w).sum((2, 3)) / (self.w.sum() * 240)).mean())
            )
        history = advance(history, pred)
        weather_seconds += time.monotonic() - begin
        begin = time.monotonic()
        next_states = []
        for j in range(count):
            state = self.ocean.step(states[j], f[j])
            next_states.append(state)
            paths[j].append(self.ocean.physical_sst(state, ctx["climate_sst"][lead - 1]))
            errors[j].append(mse[j])
            forcings[j].append(f[j])
            ocean_steps += 1
        states = next_states
        if greedy:
            gs = self.ocean.step(gs, gf)
            gp.append(self.ocean.physical_sst(gs, ctx["climate_sst"][lead - 1]))
        ocean_seconds += time.monotonic() - begin
        self.progress(
            "ocean_rollout",
            origin=origin,
            lead=lead,
            nodes=count,
            training=training,
            unique_ocean_steps=ocean_steps,
        )
    del states, next_states, state, gs, decoded, truth
    predictions = np.asarray(paths)
    members = [score(p, ctx) for p in predictions]
    rewards = torch.tensor([m["reward"] for m in members], device=self.device)
    if greedy:
        members.append(score(np.asarray(gp), ctx))
    row = dict(
        origin=origin,
        seed=seed,
        members=members,
        ensemble=score(predictions.mean(0), ctx),
        persistence=score(ctx["persistence"], ctx),
        probability=probability(predictions, ctx, independent=not tree) if not training else None,
        weather_mse=errors + ([ge] if greedy else []),
        parity=parity,
        reward_std=float(rewards.std(unbiased=False)),
        unique_member_days=ocean_steps,
        scored_cells=ctx["scored_cells"],
        missing_initial_native_wet=ctx["missing_initial_native_wet"],
        weather_seconds=weather_seconds,
        ocean_seconds=ocean_seconds,
    )
    if tree:
        q, adv, std = branch_credit(rewards)
        row["tree_credit"] = dict(
            q={str(k): v.cpu().tolist() for k, v in q.items()},
            advantages={str(k): v.cpu().tolist() for k, v in adv.items()},
            root_std=float(std),
        )
        for t in trans:
            t["adv"] = adv[t["count"]]
        if self.rank == 0 and self.update == 0:
            state = initial_ocean.clone()
            rp = []
            for k, f in enumerate(forcings[0]):
                state = self.ocean.step(state, f)
                rp.append(self.ocean.physical_sst(state, ctx["climate_sst"][k]))
            rp = np.asarray(rp)
            rs = score(rp, ctx)
            check = check_cached_ocean_path(
                rp, predictions[0], ctx["valid"], ctx["weight"], rs["reward"], members[0]["reward"]
            )
            check.update(
                check_threshold_sensitive_reward(
                    rp, predictions[0], ctx["contexts"]["1993_2018"], rs, members[0], float(std)
                )
            )
            row["cache_replay"] = dict(passed=True, extra_ocean_steps=10, **check)
            write(self.out / "CACHE_REPLAY_GPU.json", row["cache_replay"])
        assert ocean_steps == 50
    elif training:
        adv = advantages(rewards)
        row["advantages"] = adv.cpu().tolist()
        for t in trans:
            t["adv"] = adv
        assert ocean_steps == 80
    row["seconds"] = time.monotonic() - start
    return (trans, row)
